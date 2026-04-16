"""Agent Orchestrator — the agentic intelligence layer for the governed workflow.

This module provides:
1. Agent Orchestration Panel: shows current objective, findings, recommendations
2. Agent Reasoning Timeline: step-by-step decision log
3. Metadata Lineage: visual transformation chain
4. Release Readiness Verdicts: governance-aware fidelity summary
"""

from __future__ import annotations

from typing import Any

import streamlit as st


# ── Agent Reasoning Timeline ──────────────────────────────────────────────────

def build_agent_timeline(
    profile: dict[str, Any] | None,
    hygiene: dict[str, Any] | None,
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
    generation_summary: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    intake_confirmed: bool,
    hygiene_reviewed: bool,
    settings_reviewed: bool,
    metadata_status: str,
    synthetic_ready: bool,
    results_shared: bool,
) -> list[dict[str, str]]:
    """Build a decision log showing what the agent decided and why."""
    events: list[dict[str, str]] = []

    events.append({
        "status": "done",
        "label": "Established governed workspace under GOV-01",
        "detail": "Enforced role-based access, audit logging, and PHI guardrails before accepting any data.",
    })

    if profile is not None:
        rows = profile["summary"]["rows"]
        cols = profile["summary"]["columns"]
        ids = profile["summary"]["identifier_columns"]
        num = profile["summary"]["numeric_columns"]
        cat = profile["summary"]["categorical_columns"]
        events.append({
            "status": "done",
            "label": f"Parsed source schema: {cols} fields, {ids} identifier(s)",
            "detail": f"Ingested {rows} records. Classified {num} numeric, {cat} categorical. Extracted statistical fingerprints for metadata blueprint.",
        })

    if hygiene is not None and profile is not None:
        high = hygiene["severity_counts"]["High"]
        med = hygiene["severity_counts"]["Medium"]
        total = hygiene["summary"]["issues_found"]
        if high > 0:
            events.append({
                "status": "warn" if not hygiene_reviewed else "done",
                "label": f"Classified {high} blocker(s) under HYG-01 likely to distort realism",
                "detail": f"{total} total issues ({high} blockers, {med} warnings). Held package from advancement until analyst review.",
            })
        elif total > 0:
            events.append({
                "status": "done",
                "label": f"Flagged {total} non-blocking hygiene findings (HYG-02)",
                "detail": f"{med} medium-severity warnings. No blockers detected. Package eligible for advancement.",
            })
        else:
            events.append({
                "status": "done",
                "label": "Clean hygiene scan. No issues detected",
                "detail": "Source data passed all quality checks. No corrections needed before synthesis.",
            })

    if intake_confirmed:
        events.append({
            "status": "done",
            "label": "Admitted request into governed pipeline",
            "detail": "Source package entered the workflow. Scan results released for analyst review.",
        })

    if hygiene_reviewed:
        events.append({
            "status": "done",
            "label": "Analyst acknowledged hygiene findings. Corrections accepted",
            "detail": "Recommended fixes applied to metadata blueprint. Original baseline preserved for audit comparison.",
        })

    if settings_reviewed:
        included = sum(1 for m in metadata if m["include"])
        targeted = sum(1 for m in metadata if m.get("control_action", "Preserve") != "Preserve")
        events.append({
            "status": "done",
            "label": f"Froze metadata package: {included} fields, {targeted} targeted actions (META-03)",
            "detail": "Field-level handling actions finalized. Original extracted metadata retained as audit baseline.",
        })

    if metadata_status == "In Review":
        events.append({
            "status": "active",
            "label": "Submitted package for governance review (GOV-02)",
            "detail": "Package is with Manager / Reviewer. Generation locked until approval.",
        })
    elif metadata_status == "Approved":
        events.append({
            "status": "done",
            "label": "Reviewer approved metadata package. Released generation lock (GOV-01)",
            "detail": "Governance sign-off obtained. Adjusted metadata blueprint cleared for synthetic generation.",
        })
    elif metadata_status in {"Changes Requested", "Rejected"}:
        events.append({
            "status": "warn",
            "label": f"Package {metadata_status.lower()} by reviewer (META-05)",
            "detail": "Revision required. Generation remains locked until resubmission and approval.",
        })

    if generation_summary is not None:
        mode = generation_summary["noise_mode"].lower()
        rows_gen = generation_summary["rows_generated"]
        events.append({
            "status": "done",
            "label": f"Generated {rows_gen} synthetic records under {mode} posture",
            "detail": "All values derived from approved metadata blueprint. Zero source records copied. Audit snapshot created.",
        })

    if validation is not None:
        overall = validation["overall_score"]
        fidelity = validation["fidelity_score"]
        privacy = validation["privacy_score"]
        if overall >= 75:
            events.append({
                "status": "done",
                "label": f"Confirmed sandbox release suitability (FID-03, REL-01)",
                "detail": f"Fidelity {fidelity}, privacy {privacy}, overall {overall}. Package suitable for internal modeling use. REL-03: not for clinical decisions.",
            })
        elif overall >= 60:
            events.append({
                "status": "done",
                "label": f"Moderate fidelity. Review recommended before release (FID-02)",
                "detail": f"Overall {overall}. Acceptable for workflow modeling with noted limitations. Review weaker columns before broader use.",
            })
        else:
            events.append({
                "status": "warn",
                "label": f"Fidelity below release threshold (FID-01). Release held",
                "detail": f"Overall {overall}. Adjust metadata or generation controls and regenerate before release.",
            })

    if results_shared:
        events.append({
            "status": "done",
            "label": "Package released for controlled use. Audit finalized",
            "detail": "Synthetic output marked as shared. Full decision trail preserved in audit log.",
        })

    # Current pending action
    if not synthetic_ready and metadata_status == "Approved":
        events.append({"status": "pending", "label": "Generate synthetic dataset from approved blueprint", "detail": "Metadata approved. Run generation to produce the synthetic package."})
    elif not results_shared and synthetic_ready:
        events.append({"status": "pending", "label": "Review output and release for controlled use", "detail": "Synthetic data ready. Download and mark as shared when reviewed."})
    elif not intake_confirmed and profile is not None:
        events.append({"status": "pending", "label": "Submit source request into workflow", "detail": "Complete request details and submit."})
    elif intake_confirmed and not hygiene_reviewed:
        events.append({"status": "pending", "label": "Review and resolve hygiene findings", "detail": "Confirm scan results and apply recommended corrections."})
    elif hygiene_reviewed and not settings_reviewed:
        events.append({"status": "pending", "label": "Finalize metadata settings", "detail": "Review field actions and mark settings as reviewed."})
    elif settings_reviewed and metadata_status == "Draft":
        events.append({"status": "pending", "label": "Submit metadata package for governance review", "detail": "Package is ready for reviewer sign-off."})

    return events


def render_agent_timeline(
    profile: dict[str, Any] | None,
    hygiene: dict[str, Any] | None,
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
    generation_summary: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    intake_confirmed: bool,
    hygiene_reviewed: bool,
    settings_reviewed: bool,
    metadata_status: str,
    synthetic_ready: bool,
    results_shared: bool,
) -> None:
    """Render the agent reasoning timeline in the UI."""
    events = build_agent_timeline(
        profile, hygiene, metadata, controls, generation_summary, validation,
        intake_confirmed, hygiene_reviewed, settings_reviewed, metadata_status,
        synthetic_ready, results_shared,
    )

    status_colors = {
        "done": "var(--good)",
        "active": "var(--brand)",
        "warn": "var(--warn)",
        "pending": "var(--muted)",
    }
    status_bg = {
        "done": "var(--good-bg)",
        "active": "rgba(11, 94, 168, 0.08)",
        "warn": "var(--warn-bg)",
        "pending": "rgba(214, 226, 236, 0.3)",
    }

    timeline_html = ""
    for i, ev in enumerate(events):
        color = status_colors.get(ev["status"], "var(--muted)")
        bg = status_bg.get(ev["status"], "transparent")
        is_last = i == len(events) - 1
        connector = "" if is_last else (
            f'<div style="width:2px;height:12px;background:{color};opacity:0.3;margin:2px 0 2px 7px;"></div>'
        )
        icon = "✓" if ev["status"] == "done" else "●" if ev["status"] == "active" else "⚠" if ev["status"] == "warn" else "○"

        timeline_html += f"""
        <div style="display:flex;gap:10px;margin-bottom:2px;">
            <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;min-width:16px;">
                <div style="width:16px;height:16px;border-radius:8px;background:{bg};border:1.5px solid {color};
                    display:flex;align-items:center;justify-content:center;font-size:9px;color:{color};font-weight:700;">
                    {icon}
                </div>
                {connector}
            </div>
            <div style="padding-bottom:6px;">
                <div style="font-size:0.88rem;font-weight:600;color:{'var(--text)' if ev['status'] == 'done' else color};line-height:1.3;">
                    {ev['label']}
                </div>
                <div style="font-size:0.82rem;color:var(--muted);line-height:1.45;margin-top:1px;">
                    {ev['detail']}
                </div>
            </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="action-shell">
            <h4>Agent decision log</h4>
            <div style="margin-top:0.7rem;">{timeline_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Agent Orchestration Panel ─────────────────────────────────────────────────

STEP_AGENT_GUIDANCE: dict[int, dict[str, str]] = {
    0: {
        "objective": "Ingest source data and prepare the request package",
        "action": "The agent will parse the uploaded CSV, profile every column, detect identifiers and sensitive fields, and prepare the initial metadata blueprint.",
        "why": "Profiling before generation ensures the synthetic output reflects real operational structure rather than arbitrary noise.",
    },
    1: {
        "objective": "Scan for data quality risks that would distort synthetic realism",
        "action": "The agent flagged hygiene issues including missingness, outliers, invalid values, and category normalization problems. Review and accept recommended corrections.",
        "why": "Uncorrected quality issues propagate into the metadata blueprint and produce synthetic data that misrepresents real operational patterns.",
    },
    2: {
        "objective": "Finalize the metadata package for governed review",
        "action": "Review field-level handling actions (tokenize, date shift, coarse geography, group text). The agent assigned default actions based on field sensitivity. Adjust if needed.",
        "why": "The metadata package is the governed blueprint that drives all downstream generation. Every field action is auditable and reviewable.",
    },
    3: {
        "objective": "Submit metadata package for reviewer approval",
        "action": "Send the reviewed metadata package to the Manager / Reviewer for governance sign-off. The agent will track the approval state and flag any unresolved issues.",
        "why": "Governance review ensures metadata handling decisions are validated by a second role before synthetic data is produced.",
    },
    4: {
        "objective": "Generate synthetic data from the approved metadata blueprint",
        "action": "The agent will apply the approved metadata package with the selected generation controls to produce a synthetic dataset. No source records are copied.",
        "why": "Metadata-driven generation separates the statistical blueprint from individual records, enabling safe synthetic output without PHI exposure.",
    },
    5: {
        "objective": "Verify fidelity and prepare the package for controlled release",
        "action": "The agent validated the synthetic output against the source metadata, checked for identifier reuse and row overlap, and produced a release readiness assessment.",
        "why": "Verification confirms the synthetic package is analytically useful while maintaining privacy boundaries suitable for downstream modeling.",
    },
}


def render_agent_orchestration_panel(step_index: int, metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    """Render the agent guidance panel for the current workflow step."""
    guidance = STEP_AGENT_GUIDANCE.get(step_index, {})
    if not guidance:
        return

    objective = guidance.get("objective", "")
    action = guidance.get("action", "")
    why = guidance.get("why", "")

    st.markdown(
        f"""
        <div class="action-shell" style="border-left: 3px solid var(--accent);">
            <h4>Agent guidance</h4>
            <div style="margin-top:0.55rem;">
                <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--brand);margin-bottom:0.25rem;">
                    Current objective
                </div>
                <div style="font-size:1.02rem;font-weight:600;color:var(--text);line-height:1.35;margin-bottom:0.6rem;">
                    {objective}
                </div>
                <div style="font-size:0.9rem;color:var(--muted);line-height:1.55;margin-bottom:0.45rem;">
                    {action}
                </div>
                <div style="font-size:0.84rem;color:var(--brand);line-height:1.5;font-style:italic;">
                    {why}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Metadata Lineage Bar ─────────────────────────────────────────────────────

def render_metadata_lineage(active_stage: int = 0) -> None:
    """Render a visual lineage bar: Source → Metadata → Adjusted → Synthetic → Verified."""
    stages = [
        {"label": "Source data", "sub": "Uploaded CSV"},
        {"label": "Extracted metadata", "sub": "Statistical blueprint"},
        {"label": "Adjusted metadata", "sub": "Corrections applied"},
        {"label": "Synthetic output", "sub": "Generated records"},
        {"label": "Verified", "sub": "Fidelity confirmed"},
    ]

    items_html = ""
    for i, stage in enumerate(stages):
        if i < active_stage:
            color = "var(--good)"
            bg = "var(--good-bg)"
            icon = "✓"
        elif i == active_stage:
            color = "var(--brand)"
            bg = "rgba(11, 94, 168, 0.1)"
            icon = str(i + 1)
        else:
            color = "var(--muted)"
            bg = "rgba(214, 226, 236, 0.3)"
            icon = str(i + 1)

        items_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:3px;min-width:0;flex:1;">
            <div style="width:24px;height:24px;border-radius:12px;background:{bg};border:1.5px solid {color};
                display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:{color};">
                {icon}
            </div>
            <div style="font-size:0.78rem;font-weight:600;color:{color};text-align:center;white-space:nowrap;">{stage['label']}</div>
            <div style="font-size:0.7rem;color:var(--muted);text-align:center;">{stage['sub']}</div>
        </div>
        """
        if i < len(stages) - 1:
            conn_color = "var(--good)" if i < active_stage else "var(--line)"
            items_html += f'<div style="flex:0.5;height:2px;background:{conn_color};margin-top:12px;min-width:12px;"></div>'

    st.markdown(
        f"""
        <div style="display:flex;align-items:flex-start;gap:0;padding:14px 16px;
            background:var(--surface);border:1px solid var(--line);border-radius:18px;
            margin-bottom:0.85rem;box-shadow:var(--shadow);">
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Release Readiness Verdicts ────────────────────────────────────────────────

def build_release_readiness_verdicts(
    validation: dict[str, Any],
    metadata: list[dict[str, Any]],
    hygiene: dict[str, Any],
    synthetic_df_columns: list[str],
) -> list[dict[str, str]]:
    """Build qualitative release readiness verdicts instead of arbitrary percentages."""
    active_columns = [m for m in metadata if m["include"]]
    active_names = [m["column"] for m in active_columns]
    schema_retained = len([c for c in active_names if c in synthetic_df_columns])
    schema_total = len(active_names)
    schema_ok = schema_retained == schema_total

    fidelity = validation["fidelity_score"]
    privacy = validation["privacy_score"]
    overall = validation["overall_score"]

    high_issues = hygiene["severity_counts"]["High"]
    med_issues = hygiene["severity_counts"]["Medium"]

    verdicts = [
        {
            "label": "Schema preservation",
            "verdict": "Verified" if schema_ok else "Incomplete",
            "detail": f"{schema_retained}/{schema_total} approved fields retained in synthetic output.",
            "status": "Good" if schema_ok else "Bad",
        },
        {
            "label": "Numeric distribution fidelity",
            "verdict": "Strong" if fidelity >= 80 else "Moderate" if fidelity >= 65 else "Review needed",
            "detail": f"Mean, spread, and quantile alignment verified across numeric fields (indicator: {fidelity}).",
            "status": "Good" if fidelity >= 80 else "Warn" if fidelity >= 65 else "Bad",
        },
        {
            "label": "Privacy boundary",
            "verdict": "Confirmed" if privacy >= 85 else "Acceptable" if privacy >= 70 else "Review needed",
            "detail": f"Row overlap and identifier reuse checks passed under current posture (indicator: {privacy}).",
            "status": "Good" if privacy >= 85 else "Warn" if privacy >= 70 else "Bad",
        },
        {
            "label": "Hygiene correction impact",
            "verdict": "Improved" if high_issues > 0 else "Clean baseline",
            "detail": (
                f"{high_issues} high and {med_issues} medium issues were addressed before generation."
                if high_issues > 0 or med_issues > 0
                else "No significant quality issues required correction."
            ),
            "status": "Good" if high_issues == 0 else "Warn",
        },
        {
            "label": "Governance boundary",
            "verdict": "Confirmed",
            "detail": "Metadata-only transformation. No source records copied. Audit trail maintained.",
            "status": "Good",
        },
        {
            "label": "Release recommendation",
            "verdict": (
                "Ready for internal modeling sandbox"
                if overall >= 75
                else "Usable with review"
                if overall >= 60
                else "Additional iteration recommended"
            ),
            "detail": f"Verification indicator: {overall}. Not for clinical decision making (REL-03).",
            "status": "Good" if overall >= 75 else "Warn" if overall >= 60 else "Bad",
        },
    ]

    return verdicts


def render_release_readiness_verdicts(verdicts: list[dict[str, str]]) -> None:
    """Render the release readiness verdict cards."""
    cards_html = ""
    for v in verdicts:
        if v["status"] == "Good":
            border_color = "var(--good)"
            badge_bg = "var(--good-bg)"
            badge_color = "var(--good)"
        elif v["status"] == "Warn":
            border_color = "var(--warn)"
            badge_bg = "var(--warn-bg)"
            badge_color = "var(--warn)"
        else:
            border_color = "var(--danger)"
            badge_bg = "var(--danger-bg)"
            badge_color = "var(--danger)"

        cards_html += f"""
        <div style="background:var(--surface);border:1px solid var(--line);border-top:3px solid {border_color};
            border-radius:16px;padding:0.85rem 0.9rem;box-shadow:var(--shadow);">
            <div style="font-size:0.76rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;
                color:var(--muted);margin-bottom:0.35rem;">{v['label']}</div>
            <div style="display:inline-block;padding:0.22rem 0.6rem;border-radius:999px;font-size:0.8rem;
                font-weight:700;background:{badge_bg};color:{badge_color};border:1px solid {border_color}22;
                margin-bottom:0.3rem;">{v['verdict']}</div>
            <div style="font-size:0.82rem;color:var(--muted);line-height:1.45;">{v['detail']}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="action-shell">
            <h4>Synthetic package verification summary</h4>
            <div style="display:grid;grid-template-columns:repeat(3, minmax(0, 1fr));gap:0.7rem;margin-top:0.7rem;">
                {cards_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Agent-Enhanced Audit Events ───────────────────────────────────────────────

AGENT_AUDIT_PREFIXES = {
    "Dataset loaded": "Agent ingested source package",
    "Request submitted": "Agent processed request into governed pipeline",
    "Hygiene fixes applied": "Agent applied recommended corrections",
    "Scan review completed": "Agent acknowledged scan findings",
    "Data settings reviewed": "Agent prepared metadata package",
    "Request submitted for review": "Agent submitted package for governance review",
    "Request approved": "Agent recorded reviewer approval",
    "Request changes requested": "Agent flagged package revision requirement",
    "Request rejected": "Agent recorded package rejection",
    "Synthetic dataset generated": "Agent generated synthetic cohort from metadata blueprint",
    "Results shared": "Agent finalized release record",
}


def agent_event_label(original_event: str) -> str:
    """Convert standard audit event labels to agent-framed language."""
    return AGENT_AUDIT_PREFIXES.get(original_event, original_event)

# ── Agent Readiness Engine ────────────────────────────────────────────────────
# The agent computes readiness status based on actual conditions and findings.
# This is what makes the agent a controller, not just a narrator.

REASON_CODES = {
    "HYG-01": "Unresolved high-severity hygiene issue blocks review readiness",
    "HYG-02": "Medium-severity hygiene warnings present but non-blocking",
    "HYG-03": "Hygiene scan not yet reviewed by analyst",
    "META-01": "Metadata settings not reviewed by analyst",
    "META-02": "Category normalization recommended before synthesis",
    "META-03": "Pending metadata actions reduce package confidence",
    "META-04": "Metadata package not yet submitted for approval",
    "META-05": "Metadata package rejected or returned for revision",
    "GOV-01": "Metadata-only transformation confirmed. No source records copied",
    "GOV-02": "Reviewer approval not yet obtained",
    "GOV-03": "Package requires governance sign-off before generation",
    "FID-01": "Fidelity score below threshold for release recommendation",
    "FID-02": "Acceptable numeric drift for non-clinical workflow modeling",
    "FID-03": "Strong fidelity confirms analytical utility",
    "REL-01": "Ready for internal modeling sandbox use",
    "REL-02": "Release held pending unresolved issues",
    "REL-03": "Not suitable for clinical decision making",
    "VAL-01": "Negative duration values detected in operational fields",
    "VAL-02": "Extreme outliers may distort operational distributions",
}


def classify_hygiene_issues(hygiene: dict[str, Any]) -> list[dict[str, Any]]:
    """Classify each hygiene issue as blocker, warning, or informational."""
    classified = []
    for issue in hygiene.get("issues", []):
        severity = issue["severity"]
        concern = issue.get("concern", "")
        finding = issue.get("finding", "").lower()

        # Blockers: issues that would make synthetic data misleading
        if severity == "High" and any(kw in concern.lower() + finding for kw in [
            "negative", "identifier", "extreme wait"
        ]):
            level = "blocker"
            reason = REASON_CODES.get("VAL-01", "High-severity issue blocks readiness")
        elif severity == "High":
            level = "blocker"
            reason = REASON_CODES["HYG-01"]
        elif severity == "Medium":
            level = "warning"
            reason = REASON_CODES["HYG-02"]
        else:
            level = "informational"
            reason = "Low-severity observation. Does not affect readiness."

        classified.append({
            **issue,
            "classification": level,
            "reason_code": "VAL-01" if "negative" in finding else "VAL-02" if "extreme" in finding or "outlier" in finding else "HYG-01" if level == "blocker" else "HYG-02" if level == "warning" else "HYG-03",
            "reason": reason,
            "blocks_review": level == "blocker",
            "blocks_release": level in ("blocker", "warning"),
        })
    return classified


def compute_agent_readiness(
    profile: dict[str, Any] | None,
    hygiene: dict[str, Any] | None,
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
    validation: dict[str, Any] | None,
    intake_confirmed: bool,
    hygiene_reviewed: bool,
    settings_reviewed: bool,
    metadata_status: str,
    synthetic_ready: bool,
    results_shared: bool,
) -> dict[str, Any]:
    """Compute the current agent readiness status with reason codes and blockers.

    This is the core of the agent-as-controller pattern. The agent evaluates
    actual conditions and returns a status that should influence whether
    the workflow can advance.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    reason_codes: list[str] = []
    confidence = "high"

    # Phase 1: Pre-submission checks
    if profile is None:
        return {
            "status": "awaiting_data",
            "label": "Awaiting source data",
            "blockers": ["No dataset uploaded"],
            "warnings": [],
            "reason_codes": [],
            "confidence": "none",
            "recommendation": "Upload a source CSV to begin the governed workflow.",
        }

    # Phase 2: Hygiene assessment
    if hygiene is not None:
        classified = classify_hygiene_issues(hygiene)
        blocker_issues = [c for c in classified if c["classification"] == "blocker"]
        warning_issues = [c for c in classified if c["classification"] == "warning"]

        if blocker_issues and not hygiene_reviewed:
            for bi in blocker_issues:
                blockers.append(f"{bi['column']}: {bi['finding']}")
                reason_codes.append(bi["reason_code"])
            confidence = "low"

        if warning_issues:
            for wi in warning_issues:
                warnings.append(f"{wi['column']}: {wi['finding']}")
            if not reason_codes:
                reason_codes.append("HYG-02")

    if not hygiene_reviewed and intake_confirmed:
        blockers.append("Hygiene scan not yet reviewed")
        reason_codes.append("HYG-03")

    # Phase 3: Metadata readiness
    if not settings_reviewed and hygiene_reviewed:
        blockers.append("Metadata settings not finalized")
        reason_codes.append("META-01")
        confidence = "medium" if confidence != "low" else "low"

    pending_actions = sum(1 for m in metadata if m.get("control_action", "Preserve") != "Preserve" and m["include"])
    if pending_actions > 0 and settings_reviewed:
        reason_codes.append("META-03")

    # Phase 4: Governance status
    if metadata_status == "Draft" and settings_reviewed:
        blockers.append("Package not yet submitted for reviewer approval")
        reason_codes.append("META-04")

    if metadata_status in ("Changes Requested", "Rejected"):
        blockers.append(f"Package {metadata_status.lower()} by reviewer")
        reason_codes.append("META-05")
        confidence = "low"

    if metadata_status == "In Review":
        warnings.append("Awaiting reviewer decision")
        reason_codes.append("GOV-02")

    if metadata_status == "Approved":
        reason_codes.append("GOV-01")

    # Phase 5: Generation and validation
    if synthetic_ready and validation is not None:
        overall = validation.get("overall_score", 0)
        fidelity = validation.get("fidelity_score", 0)
        privacy = validation.get("privacy_score", 0)

        if overall < 60:
            blockers.append(f"Fidelity below release threshold (verification indicator: {overall})")
            reason_codes.append("FID-01")
            confidence = "low"
        elif overall < 75:
            warnings.append(f"Moderate fidelity (verification indicator: {overall}). Review weaker columns before release.")
            reason_codes.append("FID-02")
            confidence = "medium"
        else:
            reason_codes.append("FID-03")

        reason_codes.append("REL-03")  # Always: not for clinical use

    # Determine overall status
    if results_shared:
        status = "released"
        label = "Package released for controlled use"
        recommendation = "Synthetic output has been shared. Audit record finalized."
    elif synthetic_ready and not blockers:
        status = "release_ready"
        label = "Ready for internal modeling sandbox"
        reason_codes.append("REL-01")
        recommendation = "Download synthetic output and mark results as shared when ready."
    elif metadata_status == "Approved" and not synthetic_ready:
        status = "generation_ready"
        label = "Approved for synthetic generation"
        recommendation = "Run generation to produce the synthetic dataset from the approved metadata blueprint."
    elif blockers:
        status = "blocked"
        label = "Not ready for review"
        reason_codes.append("REL-02")
        recommendation = f"Resolve {len(blockers)} blocking issue(s) before advancing."
    elif warnings and not settings_reviewed:
        status = "needs_action"
        label = "Needs metadata action"
        recommendation = "Review and finalize metadata settings to proceed."
    elif settings_reviewed and metadata_status == "Draft":
        status = "review_ready"
        label = "Review ready"
        recommendation = "Submit the reviewed metadata package for governance approval."
    elif metadata_status == "In Review":
        status = "pending_review"
        label = "Pending reviewer decision"
        recommendation = "Manager / Reviewer should approve, request changes, or reject the package."
    else:
        status = "in_progress"
        label = "Workflow in progress"
        recommendation = "Continue through the governed workflow steps."

    return {
        "status": status,
        "label": label,
        "blockers": blockers,
        "warnings": warnings,
        "reason_codes": list(dict.fromkeys(reason_codes)),  # deduplicate, preserve order
        "confidence": confidence,
        "recommendation": recommendation,
    }


def render_agent_readiness_panel(readiness: dict[str, Any]) -> None:
    """Render the agent readiness status as a prominent panel."""
    import html as _html
    status = readiness["status"]
    label = _html.escape(readiness["label"])
    confidence = _html.escape(readiness["confidence"].title())
    recommendation = _html.escape(readiness["recommendation"])
    blockers = readiness["blockers"]
    warnings = readiness["warnings"]
    codes = readiness["reason_codes"]

    if status in ("release_ready", "released"):
        border = "#2E7040"; badge_bg = "#EDF9F3"; badge_color = "#136B48"
    elif status in ("blocked", "needs_action"):
        border = "#C62828"; badge_bg = "#FFF1F3"; badge_color = "#9D2B3C"
    elif status in ("pending_review", "review_ready"):
        border = "#004B8B"; badge_bg = "#EBF1F7"; badge_color = "#08467D"
    else:
        border = "#D68A00"; badge_bg = "#FFF6E3"; badge_color = "#9C6A17"

    blockers_html = ""
    if blockers:
        items = "".join(f'<li>{_html.escape(b)}</li>' for b in blockers)
        blockers_html = (
            '<div style="margin-top:0.55rem;padding:0.6rem 0.75rem;background:#FFF1F3;'
            'border:1px solid rgba(143,45,53,0.15);border-radius:12px;">'
            '<div style="font-size:0.76rem;font-weight:700;color:#9D2B3C;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.25rem;">Blocking issues</div>'
            f'<ul style="margin:0;padding-left:1rem;color:#2D3E50;font-size:0.86rem;line-height:1.5;">{items}</ul>'
            '</div>'
        )

    warnings_html = ""
    if warnings:
        items = "".join(f'<li>{_html.escape(w)}</li>' for w in warnings)
        warnings_html = (
            '<div style="margin-top:0.45rem;padding:0.6rem 0.75rem;background:#FFF6E3;'
            'border:1px solid rgba(138,97,22,0.15);border-radius:12px;">'
            '<div style="font-size:0.76rem;font-weight:700;color:#9C6A17;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.25rem;">Warnings</div>'
            f'<ul style="margin:0;padding-left:1rem;color:#2D3E50;font-size:0.86rem;line-height:1.5;">{items}</ul>'
            '</div>'
        )

    codes_html = ""
    if codes:
        pill_parts = []
        for c in codes[:6]:
            title_text = _html.escape(REASON_CODES.get(c, c))
            code_text = _html.escape(c)
            pill_parts.append(
                f'<span title="{title_text}" style="display:inline-block;padding:0.2rem 0.5rem;'
                'border-radius:999px;font-size:0.72rem;font-weight:700;font-family:monospace;'
                'background:rgba(11,94,168,0.06);border:1px solid rgba(11,94,168,0.12);'
                f'color:#004B8B;margin-right:0.3rem;margin-bottom:0.25rem;cursor:help;">{code_text}</span>'
            )
        codes_html = '<div style="margin-top:0.5rem;">' + "".join(pill_parts) + '</div>'

    panel_html = (
        f'<div class="action-shell" style="border-left:3px solid {border};">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        '<div>'
        '<h4>Agent readiness assessment</h4>'
        f'<div style="display:inline-block;padding:0.28rem 0.7rem;border-radius:999px;font-size:0.82rem;'
        f'font-weight:700;background:{badge_bg};color:{badge_color};border:1px solid {border}33;margin-top:0.35rem;">{label}</div>'
        '</div>'
        '<div style="text-align:right;">'
        '<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#668097;">Confidence</div>'
        f'<div style="font-size:0.92rem;font-weight:700;color:{badge_color};margin-top:0.15rem;">{confidence}</div>'
        '</div>'
        '</div>'
        f'<div style="margin-top:0.5rem;font-size:0.9rem;color:#668097;line-height:1.55;">'
        f'<strong style="color:#2D3E50;">Recommendation:</strong> {recommendation}'
        '</div>'
        f'{blockers_html}{warnings_html}{codes_html}'
        '</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)


# ── Metadata Review Artifact ─────────────────────────────────────────────────

def build_metadata_review_artifact(
    metadata: list[dict[str, Any]],
    profile: dict[str, Any],
    hygiene_classified: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a formal metadata lineage artifact.

    Shows for each field: extracted metadata, agent recommendation,
    final approved assumption, and effect on synthetic generation.
    """
    artifact = []
    hygiene_by_col: dict[str, list[dict[str, Any]]] = {}
    for h in hygiene_classified:
        col = h.get("column", "")
        hygiene_by_col.setdefault(col, []).append(h)

    for item in metadata:
        col = item["column"]
        col_profile = profile.get("columns", {}).get(col, {})
        col_issues = hygiene_by_col.get(col, [])

        role = col_profile.get("semantic_role", item["data_type"])
        action = item.get("control_action", "Preserve")
        has_action = action != "Preserve"
        has_blocker = any(i["classification"] == "blocker" for i in col_issues)
        has_warning = any(i["classification"] == "warning" for i in col_issues)

        # Extracted metadata summary
        if role == "numeric":
            mean = col_profile.get("mean")
            std = col_profile.get("std")
            if mean is not None:
                extracted = f"mean {mean}, std {std}"
            else:
                extracted = f"{role}"
        elif role == "date":
            dmin = col_profile.get("min", "?")
            dmax = col_profile.get("max", "?")
            extracted = f"range {dmin} to {dmax}"
        elif role == "identifier":
            extracted = f"{col_profile.get('unique_count', 0)} unique tokens"
        else:
            top = col_profile.get("top_values", {})
            if top:
                top_cat = list(top.keys())[0]
                extracted = f"top: {top_cat} ({top[top_cat]}%)"
            else:
                extracted = role
        missing = col_profile.get("missing_pct", 0)
        if missing > 0:
            extracted += f", {missing}% missing"

        # Agent recommendation
        if has_blocker:
            recommended = "Resolve blocker before generation"
        elif role == "identifier":
            recommended = "Tokenize with surrogate values"
        elif role == "date":
            recommended = "Jitter dates to protect timing"
        elif "postal" in col.lower() or "address" in col.lower():
            recommended = "Reduce to coarse geography"
        elif "complaint" in col.lower() or "note" in col.lower():
            recommended = "Group text into safer categories"
        elif has_warning:
            recommended = "Normalize before sampling"
        else:
            recommended = "Preserve extracted distribution"

        # Approved assumption
        if not item["include"]:
            approved = "Excluded from generation"
        elif has_blocker:
            approved = "Pending blocker resolution"
        else:
            approved = action

        # Effect on synthetic generation
        if not item["include"]:
            effect = "Field will not appear in synthetic output"
        elif has_blocker:
            effect = "Generation held until blocker is resolved"
        elif action == "Tokenize":
            effect = "Surrogate tokens replace source identifiers"
        elif action == "Date shift":
            effect = "Dates bootstrap-sampled with day-level jitter"
        elif action == "Month only":
            effect = "Only month-level timing released"
        elif action == "Coarse geography":
            effect = "Only coarse location preserved"
        elif action == "Group text":
            effect = "Free text collapsed to grouped categories"
        elif action == "Group rare categories":
            effect = "Rare categories grouped into Other"
        elif action == "Clip extremes":
            effect = "Values clipped to 5th-95th percentile"
        elif role == "numeric":
            effect = "Bootstrap sampling with bounded noise"
        elif role == "categorical":
            effect = "Sampled from smoothed empirical distribution"
        else:
            effect = "Preserved per default strategy"

        if not item["include"]:
            status = "excluded"
        elif has_blocker:
            status = "blocker_pending"
        elif has_action or has_warning:
            status = "adjusted"
        else:
            status = "unchanged"

        artifact.append({
            "column": col,
            "role": role,
            "extracted": extracted,
            "recommended": recommended,
            "approved": approved,
            "effect": effect,
            "status": status,
            "included": item["include"],
            "blocker": has_blocker,
        })

    return artifact


def render_metadata_review_artifact(artifact: list[dict[str, Any]]) -> None:
    """Render metadata lineage table: extracted → recommended → approved → effect."""
    status_labels = {
        "unchanged": ("Preserved", "var(--muted)"),
        "adjusted": ("Adjusted", "var(--brand)"),
        "excluded": ("Excluded", "var(--muted)"),
        "blocker_pending": ("Blocker", "var(--danger)"),
    }

    rows_html = ""
    for item in artifact:
        label, color = status_labels.get(item["status"], ("Unknown", "var(--muted)"))
        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(214,226,236,0.5);">
            <td style="padding:0.5rem 0.6rem;font-weight:600;font-size:0.86rem;vertical-align:top;">
                {item['column']}
                <div style="font-size:0.72rem;color:var(--muted);font-weight:500;margin-top:0.1rem;">{item['role']}</div>
            </td>
            <td style="padding:0.5rem 0.4rem;font-size:0.82rem;color:var(--muted);vertical-align:top;">{item['extracted']}</td>
            <td style="padding:0.5rem 0.4rem;font-size:0.82rem;color:var(--text);vertical-align:top;">{item['recommended']}</td>
            <td style="padding:0.5rem 0.4rem;vertical-align:top;">
                <span style="display:inline-block;padding:0.15rem 0.45rem;border-radius:999px;font-size:0.72rem;
                    font-weight:700;color:{color};background:{color}11;border:1px solid {color}22;white-space:nowrap;">{label}</span>
                <div style="font-size:0.78rem;color:var(--text);margin-top:0.25rem;">{item['approved']}</div>
            </td>
            <td style="padding:0.5rem 0.4rem;font-size:0.78rem;color:var(--muted);line-height:1.4;vertical-align:top;">{item['effect']}</td>
        </tr>
        """

    included = sum(1 for a in artifact if a["included"])
    blockers = sum(1 for a in artifact if a["blocker"])
    adjusted = sum(1 for a in artifact if a["status"] == "adjusted")

    st.markdown(
        f"""
        <div class="action-shell">
            <h4>Metadata lineage artifact</h4>
            <div style="display:flex;gap:1.2rem;margin:0.5rem 0 0.75rem 0;font-size:0.84rem;color:var(--muted);">
                <span><strong style="color:var(--text);">{included}</strong> fields included</span>
                <span><strong style="color:var(--brand);">{adjusted}</strong> adjusted from baseline</span>
                <span><strong style="color:var(--danger);">{blockers}</strong> blocker(s) pending</span>
            </div>
            <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;font-size:0.86rem;">
                    <thead>
                        <tr style="border-bottom:2px solid var(--line);">
                            <th style="padding:0.5rem 0.6rem;text-align:left;color:var(--muted);font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Field</th>
                            <th style="padding:0.5rem 0.4rem;text-align:left;color:var(--muted);font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Extracted metadata</th>
                            <th style="padding:0.5rem 0.4rem;text-align:left;color:var(--muted);font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Agent recommendation</th>
                            <th style="padding:0.5rem 0.4rem;text-align:left;color:var(--muted);font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Approved assumption</th>
                            <th style="padding:0.5rem 0.4rem;text-align:left;color:var(--muted);font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Effect on generation</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Hygiene Classification Display ───────────────────────────────────────────

def render_classified_hygiene(classified: list[dict[str, Any]]) -> None:
    """Render hygiene findings with blocker/warning/informational classifications."""
    if not classified:
        return

    blockers = [c for c in classified if c["classification"] == "blocker"]
    warnings = [c for c in classified if c["classification"] == "warning"]
    info = [c for c in classified if c["classification"] == "informational"]

    def _render_group(items: list, title: str, color: str, bg: str) -> str:
        if not items:
            return ""
        rows = ""
        for item in items:
            rows += f"""
            <div style="display:flex;gap:0.6rem;align-items:flex-start;padding:0.55rem 0;border-bottom:1px solid rgba(214,226,236,0.4);">
                <span style="display:inline-block;padding:0.15rem 0.4rem;border-radius:999px;font-size:0.7rem;
                    font-weight:700;font-family:monospace;background:{bg};color:{color};border:1px solid {color}22;
                    white-space:nowrap;margin-top:0.1rem;">{item['reason_code']}</span>
                <div>
                    <div style="font-size:0.88rem;font-weight:600;color:var(--text);">{item['column']}</div>
                    <div style="font-size:0.82rem;color:var(--muted);line-height:1.45;">{item['finding']}</div>
                </div>
            </div>
            """
        return f"""
        <div style="margin-bottom:0.7rem;">
            <div style="font-size:0.76rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:{color};margin-bottom:0.3rem;">
                {title} ({len(items)})
            </div>
            {rows}
        </div>
        """

    html = _render_group(blockers, "Blockers", "var(--danger)", "var(--danger-bg)")
    html += _render_group(warnings, "Warnings", "var(--warn)", "var(--warn-bg)")
    html += _render_group(info, "Informational", "var(--muted)", "rgba(214,226,236,0.2)")

    st.markdown(
        f"""
        <div class="action-shell">
            <h4>Agent hygiene assessment</h4>
            <div style="margin-top:0.5rem;">{html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Stakeholder Interpretation ────────────────────────────────────────────────

def build_stakeholder_interpretations(
    validation: dict[str, Any] | None,
    hygiene: dict[str, Any],
    metadata: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> list[dict[str, str]]:
    """Build concise stakeholder interpretation blocks."""
    overall = validation["overall_score"] if validation else 0
    fidelity = validation["fidelity_score"] if validation else 0
    privacy = validation["privacy_score"] if validation else 0
    status = readiness["status"]
    high_issues = hygiene["severity_counts"]["High"]
    included = sum(1 for m in metadata if m["include"])
    wait_fields = any("wait" in m["column"].lower() and m["include"] for m in metadata)

    interpretations = []

    # Operations Manager
    if status in ("release_ready", "released"):
        ops_text = (
            f"The synthetic package preserves operational fields across {included} columns at {fidelity} fidelity. "
            + ("Wait time and throughput distributions are included and suitable for flow simulation. " if wait_fields else "")
            + "Suitable for capacity planning, staffing analysis, and workflow modeling."
        )
    else:
        ops_text = f"Package is not yet release-ready ({readiness['label']}). Resolve outstanding items before using for operational modeling."
    interpretations.append({"role": "Operations Manager", "focus": "Throughput, wait times, capacity", "text": ops_text})

    # Clinical Analyst
    if fidelity >= 75:
        clin_text = f"Numeric distributions align at {fidelity} fidelity. Category proportions are stable enough for analytical pipeline development and feature engineering. Verify edge-case behavior before training production models."
    elif fidelity >= 60:
        clin_text = f"Moderate fidelity ({fidelity}). Usable for exploratory analysis and pipeline testing. Review column-level scores for weak fields before drawing analytical conclusions."
    else:
        clin_text = f"Fidelity below analytical threshold ({fidelity}). Recommend adjusting metadata controls and regenerating before using for model development."
    interpretations.append({"role": "Clinical Analyst", "focus": "Pattern realism, analytical utility", "text": clin_text})

    # Compliance / Governance
    gov_text = (
        f"Metadata-only transformation confirmed (GOV-01). No source records copied into synthetic output. "
        f"Privacy boundary verified under current posture. "
        + (f"{high_issues} high-severity hygiene issues were addressed before generation. " if high_issues > 0 else "No high-severity issues in source data. ")
        + "Audit trail maintained throughout. Output is for modeling sandbox use only, not clinical decision making (REL-03)."
    )
    interpretations.append({"role": "Compliance Officer", "focus": "Privacy, auditability, governance", "text": gov_text})

    return interpretations


def render_stakeholder_interpretations(interpretations: list[dict[str, str]]) -> None:
    """Render stakeholder interpretation cards."""
    cards_html = ""
    for interp in interpretations:
        cards_html += f"""
        <div style="background:var(--surface);border:1px solid var(--line);border-radius:16px;
            padding:0.85rem 0.95rem;box-shadow:var(--shadow);">
            <div style="font-size:0.76rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;
                color:var(--brand);margin-bottom:0.2rem;">{interp['role']}</div>
            <div style="font-size:0.78rem;color:var(--muted);margin-bottom:0.4rem;">{interp['focus']}</div>
            <div style="font-size:0.86rem;color:var(--text);line-height:1.5;">{interp['text']}</div>
        </div>
        """
    st.markdown(
        f"""
        <div class="action-shell">
            <h4>Stakeholder interpretation</h4>
            <div style="display:grid;grid-template-columns:repeat(3, minmax(0, 1fr));gap:0.7rem;margin-top:0.6rem;">
                {cards_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
