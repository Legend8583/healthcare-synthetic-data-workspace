from __future__ import annotations

from typing import Any

from openai import OpenAI

DEFAULT_CHAT_MODEL = "gpt-5.4-mini"


def build_chat_context(
    source_label: str,
    profile: dict[str, Any],
    hygiene: dict[str, Any],
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
    generation_summary: dict[str, Any] | None,
    validation: dict[str, Any] | None,
) -> str:
    included_columns = [item for item in metadata if item["include"]]
    column_summaries = [
        f"{item['column']} ({item['data_type']}, strategy={item['strategy']}, nullable={item['nullable']})"
        for item in included_columns[:12]
    ]
    hygiene_summaries = [
        f"{issue['severity']} - {issue['column']}: {issue['finding']}"
        for issue in hygiene["issues"][:5]
    ]

    lines = [
        "Application: Healthcare Synthetic Data Copilot",
        "Setting: public Canadian hospital / Southlake Health-style emergency department workflow",
        f"Current dataset: {source_label}",
        f"Rows: {profile['summary']['rows']}, columns: {profile['summary']['columns']}, identifiers detected: {profile['summary']['identifier_columns']}",
        f"Hygiene score: {hygiene['quality_score']}, high issues: {hygiene['severity_counts']['High']}, medium issues: {hygiene['severity_counts']['Medium']}",
        f"Controls: fidelity_priority={controls['fidelity_priority']}, synthetic_rows={controls['synthetic_rows']}, reduce_extreme_waits={controls['reduce_extreme_waits']}, extreme_wait_reduction_pct={controls['extreme_wait_reduction_pct']}",
        "Included metadata columns:",
        *column_summaries,
    ]

    if hygiene_summaries:
        lines.extend(["Top hygiene concerns:", *hygiene_summaries])

    if generation_summary:
        lines.append(
            f"Generation summary: rows_generated={generation_summary['rows_generated']}, columns_generated={generation_summary['columns_generated']}, mode={generation_summary['noise_mode']}"
        )

    if validation:
        lines.append(
            f"Validation summary: overall={validation['overall_score']}, fidelity={validation['fidelity_score']}, privacy={validation['privacy_score']}, verdict={validation['verdict']}"
        )

    lines.extend(
        [
            "Guardrails:",
            "- Explain synthetic data clearly and do not claim formal privacy certification.",
            "- Emphasize transparency: source data -> metadata -> synthetic generation -> validation.",
            "- Position the dataset as fit for prototyping, analysis, testing, education, and sandboxing, not direct clinical decision-making.",
        ]
    )

    return "\n".join(lines)


def generate_chat_reply(
    api_key: str,
    user_message: str,
    chat_history: list[dict[str, str]],
    context: str,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    recent_history = chat_history[-6:]
    history_text = "\n".join(
        f"{item['role'].title()}: {item['content']}" for item in recent_history if item.get("content")
    )
    prompt = f"""
You are the AI copilot inside a healthcare synthetic data workflow.
Be concise, practical, and presentation-friendly.

Application context:
{context}

Recent chat:
{history_text}

Latest user question:
{user_message}

Respond with:
- a direct answer,
- any operational or analytics framing if relevant,
- a caution when the topic touches privacy or clinical use.
""".strip()

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text.strip()


def generate_demo_chat_reply(
    user_message: str,
    profile: dict[str, Any],
    hygiene: dict[str, Any],
    controls: dict[str, Any],
    validation: dict[str, Any] | None,
) -> str:
    message = user_message.lower().strip()
    privacy_label = (
        "higher privacy"
        if controls["fidelity_priority"] < 40
        else "balanced"
        if controls["fidelity_priority"] < 70
        else "higher fidelity"
    )

    if any(keyword in message for keyword in ["hi", "hello", "hey"]):
        return (
            "Hi! I’m running in local demo chat mode, so I can still help explain the workflow, "
            "privacy tradeoffs, and downstream use even without live API credits."
        )

    if any(keyword in message for keyword in ["privacy", "quota", "fidelity", "slider"]):
        return (
            f"The privacy-vs-fidelity slider is currently set to a {privacy_label} posture. "
            "Lower settings add more smoothing and randomness to protect privacy, while higher settings "
            "preserve source-like operational patterns more closely. The important point is that "
            "this tradeoff is visible and adjustable rather than hidden inside a black box."
        )

    if any(keyword in message for keyword in ["analysis", "use", "use case", "readiness", "why"]):
        overall = validation["overall_score"] if validation else "pending"
        return (
            f"This synthetic dataset is designed for downstream analysis use cases such as dashboard prototyping, "
            f"patient-flow analysis, vendor sandbox demos, and analytics pipeline testing. "
            f"The current validation score is {overall}, which helps show that the dataset is not random noise: "
            "it keeps analysis-ready structure while reducing exposure to raw patient-level records."
        )

    if any(keyword in message for keyword in ["hygiene", "issue", "quality", "risk"]):
        return (
            f"The hygiene review found {hygiene['severity_counts']['High']} high-priority and "
            f"{hygiene['severity_counts']['Medium']} medium-priority issues. "
            "That step matters because synthetic data quality depends on understanding missingness, identifiers, "
            "and outliers before generation starts."
        )

    if any(keyword in message for keyword in ["step", "workflow", "how"]):
        return (
            "The workflow is: upload source data, review hygiene, edit metadata, generate synthetic data, "
            "validate fidelity and privacy, then produce an analysis-readiness summary about downstream usefulness."
        )

    return (
        f"Demo chat mode is active. The current dataset has {profile['summary']['rows']} rows and "
        f"{profile['summary']['columns']} columns. I can help explain the six-step workflow, the privacy-vs-fidelity "
        "tradeoff, hygiene findings, or why the synthetic dataset is useful for analysis."
    )
