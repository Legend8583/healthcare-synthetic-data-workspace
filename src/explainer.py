from __future__ import annotations

from typing import Any


def _readiness_label(score: float) -> str:
    if score >= 85:
        return "High analysis readiness"
    if score >= 70:
        return "Promising analysis readiness"
    return "Needs one more iteration"


def build_readiness_briefing(
    profile: dict[str, Any],
    hygiene: dict[str, Any],
    metadata: list[dict[str, Any]],
    generation_summary: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    included_columns = [item for item in metadata if item["include"]]
    top_fidelity_columns = validation["fidelity_table"].head(5)["column"].tolist()
    top_fidelity_text = ", ".join(top_fidelity_columns) if top_fidelity_columns else "the included columns"
    readiness_label = _readiness_label(validation["overall_score"])

    executive_summary = (
        f"{readiness_label}: the app converted {profile['summary']['rows']} source rows into "
        f"{generation_summary['rows_generated']} synthetic rows while keeping the process transparent and measurable. "
        f"The current evidence suggests the dataset is suitable for downstream analysis tasks such as dashboard prototyping, "
        f"workflow simulation, and vendor sandbox testing, with strongest fidelity in {top_fidelity_text}."
    )

    proof_points = [
        {
            "title": "Transparent transformation",
            "body": "Teams can see the full chain from source data to editable metadata to synthetic output to validation, rather than a black-box generator.",
        },
        {
            "title": "Controllable risk posture",
            "body": "Metadata inclusion, generation strategies, and the privacy-vs-fidelity slider make the tradeoff explicit instead of hidden.",
        },
        {
            "title": "Measured utility",
            "body": f"The current run scored {validation['fidelity_score']}/100 on fidelity and {validation['privacy_score']}/100 on privacy, giving an overall score of {validation['overall_score']}/100.",
        },
    ]

    use_cases = [
        {
            "name": "Operational dashboard prototyping",
            "why_it_works": "Analysts can test wait-time, CTAS, admission, and disposition dashboards without using live patient records.",
            "example": "Build and QA an emergency department throughput dashboard before connecting to production data.",
        },
        {
            "name": "Workflow and patient-flow analysis",
            "why_it_works": "The schema preserves encounter structure and operational timing fields, which is enough for queueing and flow experiments.",
            "example": "Compare baseline ED flow with a scenario where extreme wait times are reduced.",
        },
        {
            "name": "Vendor or sandbox environments",
            "why_it_works": "Teams can share realistic healthcare-shaped data with lower re-identification risk than raw source data.",
            "example": "Demo an ED analytics product to hospital stakeholders or external partners.",
        },
        {
            "name": "Analytics and model prototyping",
            "why_it_works": "Data scientists can test feature pipelines, cohort logic, and notebook workflows before requesting sensitive data access.",
            "example": "Prototype predictors for revisit risk or admission likelihood using a non-production dataset.",
        },
        {
            "name": "Training and implementation development",
            "why_it_works": "Developers and clinical informatics teams can practice with realistic fields, missingness, and outlier patterns.",
            "example": "Let implementation teams build and validate an ED operations app in a safe starting environment.",
        },
    ]

    talk_track = [
        "This workflow does not rely on a hidden generator. It exposes each step from raw schema to metadata rules to validation.",
        "The output is useful because it preserves analysis-ready structure and broad operational patterns without reusing direct identifiers.",
        "This means hospitals can prototype dashboards, analytics, vendor integrations, and training workflows earlier and more safely.",
        "It is not a substitute for clinical decision-making data, but it is a strong sandbox for building, testing, and learning.",
    ]

    next_actions = [
        "Tune low-scoring columns in the metadata editor, then rerun generation.",
        "Expand the sample to more departments or time periods for broader synthetic coverage.",
        "Add stricter privacy tests if the project moves toward broader operational use.",
    ]

    return {
        "executive_summary": executive_summary,
        "proof_points": proof_points,
        "use_cases": use_cases,
        "talk_track": talk_track,
        "next_actions": next_actions,
        "readiness_label": readiness_label,
        "included_columns": len(included_columns),
        "high_issues": hygiene["severity_counts"]["High"],
    }
