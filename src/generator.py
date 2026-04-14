from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _missing_probability(series: pd.Series) -> float:
    return float(series.isna().mean())


def _generate_identifier(column_name: str, row_count: int) -> list[str]:
    prefix = "".join(part[0] for part in column_name.split("_") if part)[:3].upper() or "SYN"
    return [f"{prefix}-{index:05d}" for index in range(1, row_count + 1)]


def _coarsen_geography(series: pd.Series) -> pd.Series:
    def _coarse(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if not text:
            return pd.NA
        if " " in text:
            return text.split()[0][:3].upper()
        return text[:3].upper()

    return series.map(_coarse)


def _group_text(series: pd.Series, keep_top: int = 8) -> pd.Series:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return pd.Series([pd.NA] * len(series), index=series.index)

    counts = non_null.value_counts()
    keepers = set(counts.head(keep_top).index)
    grouped = series.fillna(pd.NA).astype("string")
    grouped = grouped.map(lambda value: value if pd.isna(value) or str(value).strip() in keepers else "Other")
    return grouped


def _group_rare_categories(series: pd.Series, keep_top: int = 6) -> pd.Series:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return pd.Series([pd.NA] * len(series), index=series.index)

    counts = non_null.value_counts()
    keepers = set(counts.head(keep_top).index)
    grouped = series.fillna(pd.NA).astype("string")
    grouped = grouped.map(lambda value: value if pd.isna(value) or str(value).strip() in keepers else "Other")
    return grouped


def _apply_missingness(
    result: pd.Series,
    source_series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    missingness_pattern: str,
    anchor_series: pd.Series | None = None,
) -> pd.Series:
    output = result.copy()
    missing_probability = _missing_probability(source_series)

    if missingness_pattern == "Fill gaps":
        return output

    if missingness_pattern == "Preserve source pattern" and anchor_series is not None:
        missing_mask = anchor_series.isna().reset_index(drop=True)
    else:
        multiplier = 1.0 if missingness_pattern == "Preserve source pattern" else 0.55
        missing_mask = pd.Series(rng.random(row_count) < min(missing_probability * multiplier, 0.98))

    return output.mask(missing_mask.to_numpy(), pd.NA)


def _prepare_anchor_output(series: pd.Series, role: str, control_action: str) -> pd.Series:
    if role == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if role == "date":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if control_action == "Month only":
            return parsed.dt.to_period("M").astype(str).replace("NaT", pd.NA)
        return parsed.dt.strftime("%Y-%m-%d").replace("NaT", pd.NA)
    return series.reset_index(drop=True)


def _blend_with_anchor(
    generated: pd.Series,
    anchor_series: pd.Series,
    rng: np.random.Generator,
    correlation_preservation: int,
    locked_distribution: bool,
) -> pd.Series:
    blend_strength = np.interp(correlation_preservation, [0, 100], [0.0, 0.88])
    if locked_distribution:
        blend_strength = min(0.96, blend_strength + 0.12)
    if blend_strength <= 0:
        return generated

    blend_mask = rng.random(len(generated)) < blend_strength
    blended = generated.copy()
    blended.loc[blend_mask] = anchor_series.loc[blend_mask]
    return blended


def _rare_row_weights(df: pd.DataFrame, metadata: list[dict[str, Any]]) -> np.ndarray:
    if df.empty:
        return np.array([])

    weights = np.ones(len(df), dtype=float)
    for item in metadata:
        if not item["include"]:
            continue

        column = item["column"]
        role = item["data_type"]
        series = df[column]

        if role in {"categorical", "binary"}:
            labels = series.fillna("Missing").astype(str)
            probabilities = labels.value_counts(normalize=True)
            rarity = labels.map(lambda value: 1.0 / max(probabilities.get(value, 1.0), 1e-6))
            baseline = max(float(np.median(rarity)), 1.0)
            weights += np.clip(rarity / baseline - 1.0, 0.0, 2.0)
        elif role == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                low = numeric.quantile(0.05)
                high = numeric.quantile(0.95)
                rare_mask = ((numeric < low) | (numeric > high)).fillna(False).to_numpy()
                weights += rare_mask.astype(float) * 0.45

    weights = np.where(np.isfinite(weights), weights, 1.0)
    total = weights.sum()
    return weights / total if total else np.full(len(df), 1 / len(df))


def _sample_categorical(
    series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    fidelity_priority: int,
    noise_level: int,
    rare_case_retention: int,
    locked_distribution: bool,
) -> pd.Series:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return pd.Series([pd.NA] * row_count)

    probabilities = non_null.value_counts(normalize=True).sort_index()
    smoothing = np.interp(fidelity_priority, [0, 100], [0.35, 0.05])
    smoothing *= np.interp(noise_level, [0, 100], [0.55, 1.55])
    if locked_distribution:
        smoothing *= 0.35
    uniform = np.full(len(probabilities), 1 / len(probabilities))
    smoothed = probabilities.to_numpy() * (1 - smoothing) + uniform * smoothing

    if rare_case_retention > 0 and len(probabilities) > 1:
        rare_threshold = float(probabilities.quantile(0.35))
        rarity_boost = np.where(probabilities.to_numpy() <= rare_threshold, np.interp(rare_case_retention, [0, 100], [1.0, 1.85]), 1.0)
        smoothed = smoothed * rarity_boost

    smoothed = smoothed / smoothed.sum()

    sampled = rng.choice(probabilities.index.to_numpy(), size=row_count, p=smoothed)
    return pd.Series(sampled)


def _sample_numeric(
    column_name: str,
    series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    fidelity_priority: int,
    noise_level: int,
    locked_distribution: bool,
    outlier_strategy: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return pd.Series([pd.NA] * row_count)

    sampled = rng.choice(numeric.to_numpy(), size=row_count, replace=True)
    std = float(numeric.std(ddof=0))
    spread = std if std > 0 else max(float(numeric.max() - numeric.min()) / 6, 1.0)
    noise_scale = np.interp(fidelity_priority, [0, 100], [0.22, 0.04])
    noise_scale *= np.interp(noise_level, [0, 100], [0.45, 1.9])
    if locked_distribution:
        noise_scale *= 0.35
    noise = rng.normal(0, spread * noise_scale, size=row_count)
    synthetic = sampled + noise

    lowered = column_name.lower()
    if outlier_strategy == "Clip extremes":
        lower = float(numeric.quantile(0.05))
        upper = float(numeric.quantile(0.95))
        synthetic = np.clip(synthetic, lower, upper)
    elif outlier_strategy == "Smooth tails":
        low = float(numeric.quantile(0.05))
        high = float(numeric.quantile(0.95))
        lower_mask = synthetic < low
        upper_mask = synthetic > high
        synthetic[lower_mask] = low - (low - synthetic[lower_mask]) * 0.45
        synthetic[upper_mask] = high + (synthetic[upper_mask] - high) * 0.45

    if any(token in lowered for token in ["age", "wait", "stay", "ctas"]):
        synthetic = np.clip(synthetic, 0, None)

    if np.allclose(numeric, np.round(numeric), atol=1e-9):
        synthetic = np.round(synthetic)

    return pd.Series(synthetic)


def _sample_dates(
    series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    fidelity_priority: int,
    control_action: str,
    noise_level: int,
    locked_distribution: bool,
) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
    if parsed.empty:
        return pd.Series([pd.NA] * row_count)

    sampled = rng.choice(parsed.to_numpy(), size=row_count, replace=True)
    jitter_days = int(round(np.interp(fidelity_priority, [0, 100], [10, 2])))
    jitter_days = max(1, int(round(jitter_days * np.interp(noise_level, [0, 100], [0.65, 1.65]))))
    if locked_distribution:
        jitter_days = max(1, int(round(jitter_days * 0.45)))
    jitter = rng.integers(-jitter_days, jitter_days + 1, size=row_count)
    jittered = pd.Series(pd.to_datetime(sampled) + pd.to_timedelta(jitter, unit="D"))

    if control_action == "Month only":
        result = jittered.dt.to_period("M").astype(str)
    else:
        result = jittered.dt.strftime("%Y-%m-%d")
    return result


def generate_synthetic_data(
    df: pd.DataFrame,
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fidelity_priority = int(controls["fidelity_priority"])
    row_count = int(controls["synthetic_rows"])
    valid_columns = {item["column"] for item in metadata if item["include"]}
    locked_columns = {column for column in controls.get("locked_columns", []) if column in valid_columns}
    correlation_preservation = int(controls.get("correlation_preservation", 35))
    rare_case_retention = int(controls.get("rare_case_retention", 30))
    noise_level = int(controls.get("noise_level", 45))
    missingness_pattern = str(controls.get("missingness_pattern", "Preserve source pattern"))
    outlier_strategy = str(controls.get("outlier_strategy", "Preserve tails"))
    generation_preset = str(controls.get("generation_preset", "Balanced"))
    seed = int(controls.get("seed", 42))

    rng = np.random.default_rng(seed)
    synthetic_columns: dict[str, pd.Series] = {}
    excluded_columns: list[str] = []
    uniform_weights = np.full(len(df), 1 / max(len(df), 1))
    rare_weights = _rare_row_weights(df, metadata) if len(df) else np.array([])
    retention_mix = np.interp(rare_case_retention, [0, 100], [0.0, 0.72])
    anchor_weights = uniform_weights if len(df) else np.array([])
    if len(df):
        anchor_weights = uniform_weights * (1 - retention_mix) + rare_weights * retention_mix
        anchor_indices = rng.choice(np.arange(len(df)), size=row_count, replace=True, p=anchor_weights)
    else:
        anchor_indices = np.array([], dtype=int)

    for column_meta in metadata:
        column = column_meta["column"]
        control_action = column_meta.get("control_action", "")
        if not column_meta["include"] or control_action == "Exclude":
            excluded_columns.append(column)
            continue

        role = column_meta["data_type"]
        source_series = df[column]
        working_series = source_series.copy()
        locked_distribution = column in locked_columns

        if control_action == "Coarse geography":
            working_series = _coarsen_geography(working_series)
        elif control_action == "Group text":
            working_series = _group_text(working_series)
        elif control_action == "Group rare categories":
            working_series = _group_rare_categories(working_series)
        elif control_action == "Clip extremes" and role == "numeric":
            numeric_series = pd.to_numeric(working_series, errors="coerce")
            if numeric_series.notna().any():
                lower = numeric_series.quantile(0.05)
                upper = numeric_series.quantile(0.95)
                working_series = numeric_series.clip(lower=lower, upper=upper)

        is_identifier_like = role == "identifier" or column_meta["strategy"] == "new_token"
        if is_identifier_like:
            generated = pd.Series(_generate_identifier(column, row_count))
        elif role == "numeric":
            generated = _sample_numeric(
                column,
                working_series,
                row_count,
                rng,
                fidelity_priority,
                noise_level,
                locked_distribution,
                outlier_strategy,
            )
        elif role == "date":
            generated = _sample_dates(
                working_series,
                row_count,
                rng,
                fidelity_priority,
                control_action,
                noise_level,
                locked_distribution,
            )
        else:
            generated = _sample_categorical(
                working_series,
                row_count,
                rng,
                fidelity_priority,
                noise_level,
                rare_case_retention,
                locked_distribution,
            )

        if not is_identifier_like:
            anchor_source = working_series.iloc[anchor_indices].reset_index(drop=True) if len(anchor_indices) else pd.Series([pd.NA] * row_count)
            anchor_ready = _prepare_anchor_output(anchor_source, role, control_action)
            generated = _blend_with_anchor(generated.reset_index(drop=True), anchor_ready, rng, correlation_preservation, locked_distribution)
            generated = _apply_missingness(generated, working_series, row_count, rng, missingness_pattern, anchor_source)

        synthetic_columns[column] = generated.reset_index(drop=True)

    synthetic_df = pd.DataFrame(synthetic_columns)

    summary = {
        "rows_generated": row_count,
        "columns_generated": len(synthetic_df.columns),
        "excluded_columns": excluded_columns,
        "fidelity_priority": fidelity_priority,
        "locked_columns": sorted(locked_columns),
        "correlation_preservation": correlation_preservation,
        "rare_case_retention": rare_case_retention,
        "noise_level": noise_level,
        "missingness_pattern": missingness_pattern,
        "outlier_strategy": outlier_strategy,
        "generation_preset": generation_preset,
        "noise_mode": "Higher privacy" if fidelity_priority < 40 else "Balanced" if fidelity_priority < 70 else "Higher fidelity",
    }

    return synthetic_df, summary
