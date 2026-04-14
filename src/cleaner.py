from __future__ import annotations

from typing import Any

import pandas as pd


def _standardize_blank_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)
    return cleaned


def _normalize_category_labels(series: pd.Series) -> tuple[pd.Series, int]:
    if series.dtype != object:
        return series, 0

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return series, 0

    normalized = non_null.str.strip().str.lower()
    groups: dict[str, list[str]] = {}
    for original, key in zip(non_null, normalized):
        groups.setdefault(key, []).append(original)

    replacement_map: dict[str, str] = {}
    normalized_changes = 0
    for values in groups.values():
        canonical = pd.Series(values).value_counts().idxmax()
        for value in set(values):
            replacement_map[value] = canonical
            if value != canonical:
                normalized_changes += 1

    updated = series.map(lambda value: replacement_map.get(value, value) if pd.notna(value) else value)
    return updated, normalized_changes


def apply_hygiene_fixes(
    df: pd.DataFrame,
    options: dict[str, bool],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cleaned = _standardize_blank_strings(df)
    actions: list[dict[str, Any]] = []

    if options.get("remove_duplicates", False):
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        removed = before - len(cleaned)
        actions.append(
            {
                "action": "Remove duplicate rows",
                "effect": f"Removed {removed} duplicate rows." if removed else "No duplicate rows were found.",
            }
        )

    if options.get("normalize_categories", False):
        total_changes = 0
        for column in cleaned.columns:
            updated, changes = _normalize_category_labels(cleaned[column])
            cleaned[column] = updated
            total_changes += changes
        actions.append(
            {
                "action": "Normalize category labels",
                "effect": (
                    f"Standardized {total_changes} category value variations."
                    if total_changes
                    else "No category normalization issues were found."
                ),
            }
        )

    if options.get("fix_negative_values", False):
        fixed_columns: list[str] = []
        for column in cleaned.columns:
            lowered = column.lower()
            if any(token in lowered for token in ["age", "wait", "stay", "ctas"]):
                numeric = pd.to_numeric(cleaned[column], errors="coerce")
                negative_mask = numeric < 0
                if bool(negative_mask.fillna(False).any()):
                    cleaned.loc[negative_mask, column] = pd.NA
                    fixed_columns.append(column)
        actions.append(
            {
                "action": "Correct negative operational values",
                "effect": (
                    "Converted negative values to missing in: " + ", ".join(fixed_columns)
                    if fixed_columns
                    else "No invalid negative values were found."
                ),
            }
        )

    return cleaned, actions
