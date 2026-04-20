from __future__ import annotations

import base64
import json
import time
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.cleaner import apply_hygiene_fixes
from src.generator import generate_synthetic_data
from src.hygiene_advisor import review_hygiene
from src.metadata_builder import build_metadata, editor_frame_to_metadata, metadata_to_editor_frame
from src.profiler import profile_dataframe
from src.validator import validate_synthetic_data
from src.agent_orchestrator import (
    render_agent_orchestration_panel,
    render_agent_timeline,
    build_release_readiness_verdicts,
    render_release_readiness_verdicts,
    agent_event_label,
    compute_agent_readiness,
    render_agent_readiness_panel,
    classify_hygiene_issues,
    render_classified_hygiene,
    build_metadata_review_artifact,
    render_metadata_review_artifact,
    build_stakeholder_interpretations,
    render_stakeholder_interpretations,
    render_consolidated_decision_log,
    render_upload_status_panel,
    render_review_package_summary,
    render_synthetic_verification_summary,
)

APP_TITLE = "Southlake Health — Agentic Synthetic Data Workspace"
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data.csv"
LOGO_PATH = Path(__file__).parent / "SL_SouthlakeHealth-logo-rgb.png"

STEP_CONFIG = [
    {
        "title": "Upload Source Data",
        "description": "Register the source dataset and open a governed request.",
        "owner": "Data Analyst",
    },
    {
        "title": "Scan Source Data",
        "description": "Agent profiles fields and classifies hygiene findings.",
        "owner": "System",
    },
    {
        "title": "Review Metadata Assumptions",
        "description": "Confirm approved metadata assumptions before reviewer handoff.",
        "owner": "Data Analyst",
    },
    {
        "title": "Submit for Reviewer Sign-off",
        "description": "Controlled handoff of the metadata-reviewed synthetic request.",
        "owner": "Data Analyst → Manager / Reviewer",
    },
    {
        "title": "Generate Synthetic Package",
        "description": "Synthesize from the approved metadata package. Metadata-only transformation.",
        "owner": "System",
    },
    {
        "title": "Release Synthetic Package",
        "description": "Download the verified synthetic package for controlled distribution.",
        "owner": "Data Analyst + Manager / Reviewer",
    },
    {
        "title": "Analyze Synthetic Data",
        "description": "Explore the released synthetic package. Source data never leaves governance.",
        "owner": "Data Analyst",
    },
]

ROLE_TO_GROUP: dict[str, str] = {
    "Data Analyst": "Analyst workspace",
    "Manager / Reviewer": "Review workspace",
}

ROLE_VISIBLE_STEPS: dict[str, list[int]] = {
    "Data Analyst": [0, 1, 2, 3, 4, 5, 6],
    "Manager / Reviewer": [3, 4, 5, 6],
}

ROLE_CONFIGS: dict[str, dict[str, Any]] = {
    "Data Analyst": {
        "password": "test",
        "summary": "Owns upload, scan review, data settings, request submission, and final download.",
        "permissions": {
            "upload",
            "view_raw",
            "remediate",
            "edit_metadata",
            "submit_metadata",
            "generate",
            "download_results",
            "share_results",
            "rollback",
        },
    },
    "Manager / Reviewer": {
        "password": "test",
        "summary": "Approves or rejects submitted requests and reviews the final synthetic output.",
        "permissions": {"approve_metadata", "view_audit", "review_results"},
    },
}

EMPTY_METADATA_COLUMNS = ["column", "include", "data_type", "strategy", "control_action", "nullable", "notes"]

WORKFLOW_STATE_KEYS = [
    "source_df",
    "source_label",
    "project_purpose",
    "source_file_size",
    "profile",
    "hygiene",
    "metadata_editor_df",
    "controls",
    "intake_confirmed",
    "hygiene_reviewed",
    "settings_reviewed",
    "settings_review_signature",
    "metadata_status",
    "metadata_submitted_by",
    "metadata_submitted_at",
    "metadata_approved_by",
    "metadata_approved_at",
    "metadata_review_note",
    "metadata_reviewed_by",
    "metadata_reviewed_at",
    "last_reviewed_metadata_signature",
    "last_cleaning_actions",
    "current_metadata_package_id",
    "metadata_package_log",
    "metadata_has_unsubmitted_changes",
    "synthetic_df",
    "generation_summary",
    "validation",
    "last_generation_signature",
    "release_status",
    "export_requested_by",
    "export_policy_approved_by",
    "export_approved_by",
    "results_shared_by",
    "results_shared_at",
    "audit_events",
]

SHARED_ROOT_KEYS = [
    "request_registry",
    "active_request_id",
    "next_request_number",
]


@st.cache_resource
def get_shared_workspace_store() -> dict[str, Any]:
    return {"state": {}}


def load_shared_workspace_state() -> None:
    shared_state = get_shared_workspace_store()["state"]
    if not shared_state:
        return
    for key in SHARED_ROOT_KEYS:
        if key in shared_state:
            st.session_state[key] = deepcopy(shared_state[key])

    if st.session_state.get("request_registry"):
        active_id = st.session_state.get("active_request_id") or st.session_state["request_registry"][0]["request_id"]
        record = next((item for item in st.session_state["request_registry"] if item["request_id"] == active_id), None)
        if record is None:
            record = st.session_state["request_registry"][0]
            st.session_state["active_request_id"] = record["request_id"]
        snapshot = record.get("snapshot", {})
        for key in WORKFLOW_STATE_KEYS:
            if key in snapshot:
                st.session_state[key] = deepcopy(snapshot[key])


def persist_shared_workspace_state() -> None:
    sync_active_request_snapshot()
    shared_state = {}
    for key in SHARED_ROOT_KEYS:
        if key in st.session_state:
            shared_state[key] = deepcopy(st.session_state[key])
    get_shared_workspace_store()["state"] = shared_state


def rerun_with_persist() -> None:
    persist_shared_workspace_state()
    st.rerun()


def capture_workflow_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in WORKFLOW_STATE_KEYS:
        if key in st.session_state:
            snapshot[key] = deepcopy(st.session_state[key])
    return snapshot


def request_status_from_snapshot(snapshot: dict[str, Any]) -> str:
    if snapshot.get("source_df") is None or snapshot.get("profile") is None:
        return "Awaiting upload"
    metadata = editor_frame_to_metadata(snapshot.get("metadata_editor_df", pd.DataFrame())) if snapshot.get("metadata_editor_df") is not None else []
    controls = snapshot.get("controls", {})
    metadata_status = snapshot.get("metadata_status", "Draft")
    if not snapshot.get("intake_confirmed"):
        return "Uploaded"
    if not snapshot.get("hygiene_reviewed"):
        return "Scanned"
    if not snapshot.get("settings_reviewed"):
        return "Needs settings review"
    if metadata_status == "Changes Requested":
        return "Changes requested"
    if metadata_status == "Rejected":
        return "Rejected"
    if metadata_status == "In Review":
        return "Pending approval"
    if metadata_status != "Approved":
        return "Reviewed"
    if snapshot.get("synthetic_df") is None:
        return "Approved"
    if snapshot.get("last_generation_signature") and snapshot.get("last_generation_signature") != build_generation_signature(metadata, controls):
        return "Regeneration required"
    if snapshot.get("results_shared_at"):
        return "Shared"
    return "Generated"


def sync_active_request_snapshot() -> None:
    request_id = st.session_state.get("active_request_id")
    registry = st.session_state.get("request_registry", [])
    if not request_id or not registry:
        return
    snapshot = capture_workflow_snapshot()
    for record in registry:
        if record["request_id"] == request_id:
            record["snapshot"] = snapshot
            record["status"] = request_status_from_snapshot(snapshot)
            record["updated_at"] = format_timestamp()
            record["source_label"] = snapshot.get("source_label", record.get("source_label", ""))
            break


def restore_request_workspace(request_id: str) -> None:
    registry = st.session_state.get("request_registry", [])
    record = next((item for item in registry if item["request_id"] == request_id), None)
    if record is None:
        return
    snapshot = deepcopy(record.get("snapshot", {}))
    for key in WORKFLOW_STATE_KEYS:
        if key in snapshot:
            st.session_state[key] = snapshot[key]
    st.session_state.active_request_id = request_id
    st.session_state.pending_active_request_selector = request_id
    st.session_state.uploaded_signature = None


def create_new_request(df: pd.DataFrame, label: str) -> str:
    request_number = int(st.session_state.get("next_request_number", 1))
    request_id = f"REQ-{request_number:03d}"
    st.session_state.next_request_number = request_number + 1
    set_source_dataframe(df, label)
    st.session_state.active_request_id = request_id
    snapshot = capture_workflow_snapshot()
    record = {
        "request_id": request_id,
        "source_label": label,
        "created_by": st.session_state.get("current_role", "System"),
        "created_at": format_timestamp(),
        "updated_at": format_timestamp(),
        "status": request_status_from_snapshot(snapshot),
        "snapshot": snapshot,
    }
    st.session_state.request_registry.insert(0, record)
    st.session_state.pending_active_request_selector = request_id
    persist_shared_workspace_state()
    return request_id


def default_generation_controls(row_count: int = 20) -> dict[str, Any]:
    return {
        "generation_preset": "Balanced",
        "fidelity_priority": 62,
        "synthetic_rows": max(int(row_count), 20),
        "locked_columns": [],
        "correlation_preservation": 40,
        "rare_case_retention": 35,
        "noise_level": 45,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Preserve tails",
        "seed": 42,
    }


def empty_metadata_editor_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_METADATA_COLUMNS)


def has_active_dataset() -> bool:
    return st.session_state.get("source_df") is not None and st.session_state.get("profile") is not None


def reset_workspace_to_empty_request(label: str = "Awaiting dataset upload") -> None:
    st.session_state.source_df = None
    st.session_state.source_label = label
    st.session_state.project_purpose = ""
    st.session_state.source_file_size = None
    st.session_state.profile = None
    st.session_state.hygiene = {
        "issues": [],
        "severity_counts": {"High": 0, "Medium": 0, "Low": 0},
        "quality_score": 100,
        "summary": {"issues_found": 0, "high_priority": 0, "duplicate_rows": 0},
    }
    st.session_state.metadata_editor_df = empty_metadata_editor_frame()
    st.session_state.controls = default_generation_controls()
    st.session_state.intake_confirmed = False
    st.session_state.hygiene_reviewed = False
    st.session_state.settings_reviewed = False
    st.session_state.settings_review_signature = None
    st.session_state.metadata_status = "Draft"
    st.session_state.metadata_submitted_by = None
    st.session_state.metadata_submitted_at = None
    st.session_state.metadata_approved_by = None
    st.session_state.metadata_approved_at = None
    st.session_state.metadata_review_note = None
    st.session_state.metadata_reviewed_by = None
    st.session_state.metadata_reviewed_at = None
    st.session_state.last_reviewed_metadata_signature = None
    st.session_state.last_cleaning_actions = []
    st.session_state.current_metadata_package_id = None
    st.session_state.metadata_package_log = []
    st.session_state.metadata_has_unsubmitted_changes = False
    clear_generation_outputs()
    st.session_state.audit_events = []
    st.session_state.uploaded_signature = None
    st.session_state.current_step = 0


def create_blank_request(label: str = "Awaiting dataset upload") -> str:
    request_number = int(st.session_state.get("next_request_number", 1))
    request_id = f"REQ-{request_number:03d}"
    st.session_state.next_request_number = request_number + 1
    reset_workspace_to_empty_request(label)
    st.session_state.active_request_id = request_id
    snapshot = capture_workflow_snapshot()
    record = {
        "request_id": request_id,
        "source_label": label,
        "created_by": st.session_state.get("current_role", "System"),
        "created_at": format_timestamp(),
        "updated_at": format_timestamp(),
        "status": request_status_from_snapshot(snapshot),
        "snapshot": snapshot,
    }
    st.session_state.request_registry.insert(0, record)
    st.session_state.pending_active_request_selector = request_id
    persist_shared_workspace_state()
    return request_id


def role_with_group(role: str) -> str:
    return f"{role} · {ROLE_TO_GROUP.get(role, 'Users')}"


def clean_dataset_label(label: str) -> str:
    if " • " in label:
        return label.split(" • ", 1)[-1]
    return label


def request_display_label(request_id: str) -> str:
    registry = st.session_state.get("request_registry", [])
    record = next((item for item in registry if item["request_id"] == request_id), None)
    if record is None:
        return request_id
    label = clean_dataset_label(record.get("source_label", "Dataset"))
    return f"{record['request_id']} · {record['status']} · {label}"


def current_dataset_label() -> str:
    return clean_dataset_label(st.session_state.get("source_label", "No dataset uploaded"))


def format_file_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return "Unknown size"
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


def dataset_status_summary() -> tuple[str, str]:
    if not has_active_dataset():
        return "Awaiting upload", "No source dataset is loaded in the current request."
    rows = st.session_state.profile["summary"]["rows"]
    columns = st.session_state.profile["summary"]["columns"]
    size = format_file_size(st.session_state.get("source_file_size"))
    return "Profile ready", f"{current_dataset_label()} · {size} · {rows:,} rows · {columns} columns"


def build_submission_checklist() -> list[tuple[str, bool]]:
    purpose_added = bool(st.session_state.get("project_purpose", "").strip())
    dataset_uploaded = st.session_state.get("source_df") is not None
    preview_generated = has_active_dataset()
    return [
        ("Project purpose added", purpose_added),
        ("Dataset uploaded", dataset_uploaded),
        ("Workspace preview generated", preview_generated),
    ]


def submission_ready() -> bool:
    return all(done for _, done in build_submission_checklist())


def submission_missing_items() -> list[str]:
    return [label for label, done in build_submission_checklist() if not done]


def build_current_request_status_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[tuple[str, str]]:
    package_record = current_review_package_record() or active_metadata_package_record(metadata)
    review_value = metadata_display_status(metadata)
    if package_record is not None:
        review_value = f"{review_value} · {package_record['package_id']}"
    return [
        ("Current request", st.session_state.active_request_id or "Not created"),
        ("Request status", request_status_from_snapshot(capture_workflow_snapshot())),
        ("Review status", review_value),
        ("Final output", effective_release_status(metadata, controls)),
    ]


def build_request_queue_frame() -> pd.DataFrame:
    rows = []
    for record in st.session_state.get("request_registry", []):
        snapshot = record.get("snapshot", {})
        rows.append(
            {
                "Request": record["request_id"],
                "Dataset": clean_dataset_label(record.get("source_label", "")),
                "Purpose": snapshot.get("project_purpose", "") or "Not set",
                "Status": record.get("status", "Draft"),
                "Updated": record.get("updated_at", ""),
                "Created by": record.get("created_by", ""),
            }
        )
    return pd.DataFrame(rows)


def clear_request_queue() -> None:
    st.session_state.request_registry = []
    st.session_state.active_request_id = None
    st.session_state.pending_active_request_selector = None
    st.session_state.next_request_number = 1
    reset_workspace_to_empty_request()
    create_blank_request()


def schedule_request_queue_clear() -> None:
    st.session_state.pending_queue_clear = True
    st.rerun()

GENERATION_PRESETS: dict[str, dict[str, Any]] = {
    "Balanced": {
        "fidelity_priority": 62,
        "correlation_preservation": 40,
        "rare_case_retention": 35,
        "noise_level": 45,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Preserve tails",
    },
    "Privacy-first": {
        "fidelity_priority": 38,
        "correlation_preservation": 25,
        "rare_case_retention": 20,
        "noise_level": 62,
        "missingness_pattern": "Reduce missingness",
        "outlier_strategy": "Smooth tails",
    },
    "Analysis-first": {
        "fidelity_priority": 78,
        "correlation_preservation": 70,
        "rare_case_retention": 58,
        "noise_level": 24,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Preserve tails",
    },
}

METADATA_QUICK_PRESETS: dict[str, str] = {
    "Use approved metadata": "",
    "Tighten sensitive fields": "tighten_phi",
    "Preserve analytics detail": "preserve_analytics",
    "Reset metadata defaults": "reset_defaults",
}


@st.cache_data(show_spinner=False)
def load_logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    suffix = LOGO_PATH.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{encoded}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

            :root {
                --brand: #0b5ea8;
                --brand-deep: #08467d;
                --accent: #19cbc5;
                --bg: #f4f8fb;
                --surface: #ffffff;
                --surface-soft: #f8fbfd;
                --line: #d6e2ec;
                --text: #17324d;
                --muted: #668097;
                --warn: #9c6a17;
                --warn-bg: #fff6e3;
                --danger: #9d2b3c;
                --danger-bg: #fff1f3;
                --good: #136b48;
                --good-bg: #edf9f3;
                --shadow: 0 10px 24px rgba(8, 70, 125, 0.08);
            }

            html {
                color-scheme: light;
            }

            html, body, [class*="css"] {
                font-family: "IBM Plex Sans", sans-serif;
                color: var(--text);
                background: var(--bg);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(11, 94, 168, 0.07), transparent 22%),
                    radial-gradient(circle at top right, rgba(25, 203, 197, 0.08), transparent 16%),
                    linear-gradient(180deg, #f8fbfd 0%, var(--bg) 100%);
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"] {
                background: transparent !important;
            }

            section[data-testid="stSidebar"],
            [data-testid="collapsedControl"] {
                display: none !important;
            }

            .block-container {
                max-width: 1360px;
                padding-top: 1.2rem;
                padding-bottom: 2.8rem;
            }

            .banner {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 1.25rem 1.4rem;
                color: var(--text);
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }

            .banner-kicker {
                display: inline-block;
                background: rgba(11, 94, 168, 0.08);
                border: 1px solid rgba(11, 94, 168, 0.12);
                border-radius: 999px;
                padding: 0.34rem 0.72rem;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                margin-bottom: 0.8rem;
                color: var(--brand);
            }

            .banner h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.12;
                color: var(--text);
            }

            .banner p {
                margin: 0.7rem 0 0 0;
                max-width: 860px;
                line-height: 1.55;
                color: var(--muted);
            }

            .topbar-shell {
                display: flex;
                justify-content: flex-start;
                gap: 1rem;
                align-items: center;
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 1rem 1.2rem;
                box-shadow: var(--shadow);
                margin-bottom: 0.85rem;
            }

            .brand-lockup {
                display: flex;
                align-items: center;
                gap: 1rem;
                min-width: 0;
            }

            .brand-logo {
                width: 240px;
                height: auto;
                object-fit: contain;
                flex: 0 0 auto;
            }

            .login-layout {
                min-height: calc(100vh - 3rem);
                display: flex;
                align-items: center;
            }

            .login-brand-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,253,0.98));
                border: 1px solid var(--line);
                border-radius: 28px;
                padding: 1.8rem 1.85rem;
                box-shadow: var(--shadow);
                min-height: 640px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
            }

            .login-brand-lockup {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }

            .login-logo {
                width: 420px;
                max-width: 100%;
                height: auto;
                object-fit: contain;
                margin-bottom: 0.4rem;
            }

            .login-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                width: fit-content;
                border-radius: 999px;
                padding: 0.32rem 0.72rem;
                background: rgba(11, 94, 168, 0.08);
                color: var(--brand);
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                border: 1px solid rgba(11, 94, 168, 0.12);
            }

            .login-title {
                font-size: 2rem;
                font-weight: 700;
                line-height: 1.08;
                color: var(--text);
                margin: 0;
                max-width: 520px;
            }

            .login-subtitle {
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.55;
                max-width: 520px;
                margin-top: 0.5rem;
            }

            .trust-badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.7rem;
                margin-top: 1.2rem;
            }

            .trust-badge {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.44rem 0.78rem;
                background: var(--surface);
                border: 1px solid var(--line);
                color: var(--text);
                font-size: 0.86rem;
                font-weight: 600;
            }

            .group-strip {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 1.4rem;
            }

            .group-chip {
                background: rgba(11, 94, 168, 0.04);
                border: 1px solid rgba(11, 94, 168, 0.1);
                border-radius: 18px;
                padding: 0.85rem 0.95rem;
            }

            .group-chip-label {
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .group-chip-value {
                color: var(--text);
                font-size: 0.92rem;
                line-height: 1.45;
                font-weight: 600;
            }

            .group-chip-meta {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.5;
                margin-top: 0.35rem;
            }

            .login-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 1.35rem 1.4rem 1.2rem 1.4rem;
                box-shadow: var(--shadow);
                min-height: 640px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
            }

            .login-card-header {
                margin-bottom: 1rem;
            }

            .login-card-title {
                margin: 0;
                color: var(--text);
                font-size: 1.45rem;
                font-weight: 700;
                line-height: 1.15;
            }

            .login-card-subtitle {
                margin-top: 0.45rem;
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.5;
            }

            .environment-tag {
                display: inline-flex;
                align-items: center;
                margin-bottom: 0.8rem;
                border-radius: 999px;
                padding: 0.34rem 0.72rem;
                background: rgba(25, 203, 197, 0.1);
                color: #0d6b69;
                border: 1px solid rgba(25, 203, 197, 0.16);
                font-size: 0.78rem;
                font-weight: 700;
            }

            .login-links {
                display: flex;
                gap: 1rem;
                align-items: center;
                margin-top: 0.5rem;
                margin-bottom: 1rem;
                font-size: 0.9rem;
            }

            .login-links a {
                color: var(--brand);
                text-decoration: none;
                font-weight: 600;
            }

            .security-notice {
                background: rgba(11, 94, 168, 0.04);
                border: 1px solid rgba(11, 94, 168, 0.12);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.9rem;
            }

            .security-notice strong {
                color: var(--text);
                display: block;
                margin-bottom: 0.3rem;
            }

            .login-helper {
                color: var(--muted);
                font-size: 0.84rem;
                line-height: 1.5;
                margin-top: 0.7rem;
            }

            .login-divider {
                height: 1px;
                background: linear-gradient(90deg, rgba(11, 94, 168, 0.14), rgba(11, 94, 168, 0.04));
                margin: 1rem 0;
            }

            .brand-copy {
                min-width: 0;
            }

            .brand-title {
                font-size: 1.85rem;
                font-weight: 700;
                line-height: 1.1;
                color: var(--text);
                margin: 0;
            }

            .brand-subtitle {
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.45;
            }

            .session-rail {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                min-width: 420px;
            }

            .session-item {
                background: var(--surface-soft);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.85rem 0.9rem;
            }

            .session-label {
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .session-value {
                font-size: 1.02rem;
                font-weight: 700;
                color: var(--text);
                line-height: 1.3;
            }

            .session-meta {
                color: var(--muted);
                font-size: 0.84rem;
                margin-top: 0.3rem;
                line-height: 1.4;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.9rem;
                margin-bottom: 1rem;
            }

            .summary-grid.three {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }

            .summary-grid.compact {
                margin-bottom: 0.8rem;
            }

            .login-shell {
                min-height: 100vh;
                display: flex;
                align-items: center;
            }

            .summary-tile {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                box-shadow: var(--shadow);
                min-height: 128px;
            }

            .summary-tile.slim {
                min-height: 108px;
                padding: 0.9rem 0.95rem;
            }

            .summary-label {
                font-size: 0.76rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--brand);
                margin-bottom: 0.42rem;
            }

            .summary-value {
                font-size: 1.5rem;
                font-weight: 700;
                line-height: 1.15;
                color: var(--text);
                margin-bottom: 0.35rem;
            }

            .summary-meta {
                color: var(--muted);
                line-height: 1.45;
                font-size: 0.88rem;
            }

            .action-shell {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 0.95rem 1.05rem;
                box-shadow: var(--shadow);
                margin-bottom: 0.85rem;
            }

            .action-shell h4 {
                margin: 0;
                color: var(--brand-deep);
                font-size: 0.86rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .action-headline {
                color: var(--text);
                font-size: 1.2rem;
                font-weight: 700;
                line-height: 1.35;
                margin: 0.55rem 0 0.35rem 0;
            }

            .action-subtext {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.55;
                margin-bottom: 0.85rem;
            }

            .plain-list {
                margin: 0;
                padding-left: 1rem;
                color: var(--muted);
            }

            .plain-list li {
                margin-bottom: 0.35rem;
                line-height: 1.5;
            }

            .status-lines {
                display: flex;
                flex-direction: column;
                gap: 0.7rem;
            }

            .status-line {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: baseline;
                padding-bottom: 0.55rem;
                border-bottom: 1px solid rgba(214, 226, 236, 0.72);
            }

            .status-line:last-child {
                border-bottom: none;
                padding-bottom: 0;
            }

            .status-line-label {
                color: var(--muted);
                font-size: 0.86rem;
                font-weight: 600;
            }

            .status-line-value {
                color: var(--text);
                font-size: 0.96rem;
                font-weight: 700;
                text-align: right;
            }

            .workflow-progress {
                display: grid;
                grid-template-columns: repeat(7, minmax(0, 1fr));
                gap: 0.55rem;
                margin-bottom: 0.85rem;
            }

            .workflow-progress-step {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.72rem 0.8rem;
                min-height: 88px;
                box-shadow: none;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }

            .workflow-progress-step.complete {
                background: rgba(25, 203, 197, 0.07);
                border-color: rgba(25, 203, 197, 0.22);
            }

            .workflow-progress-step.current {
                background: linear-gradient(180deg, rgba(11, 94, 168, 0.12), rgba(25, 203, 197, 0.05));
                border-color: rgba(11, 94, 168, 0.34);
                box-shadow: 0 10px 22px rgba(11, 94, 168, 0.12);
            }

            .workflow-progress-step.next {
                background: rgba(11, 94, 168, 0.04);
                border-color: rgba(11, 94, 168, 0.18);
            }

            .workflow-progress-step.future {
                background: rgba(255, 255, 255, 0.75);
                border-style: dashed;
                opacity: 0.68;
            }

            .workflow-progress-marker {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 999px;
                background: rgba(11, 94, 168, 0.08);
                color: var(--brand);
                font-weight: 800;
                font-size: 0.92rem;
                margin-bottom: 0.7rem;
            }

            .workflow-progress-step.complete .workflow-progress-marker {
                background: rgba(25, 203, 197, 0.16);
                color: #0d6b69;
            }

            .workflow-progress-step.current .workflow-progress-marker {
                background: var(--brand);
                color: white;
            }

            .workflow-progress-owner {
                color: var(--brand);
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.28rem;
                line-height: 1.25;
            }

            .workflow-progress-title {
                color: var(--brand-deep);
                font-size: 0.94rem;
                font-weight: 800;
                line-height: 1.22;
                margin-bottom: 0.24rem;
            }

            .workflow-progress-state {
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 700;
                margin-top: auto;
            }

            .request-file-card {
                background: linear-gradient(180deg, rgba(11, 94, 168, 0.05), rgba(25, 203, 197, 0.04));
                border: 1px solid rgba(11, 94, 168, 0.16);
                border-radius: 20px;
                padding: 1rem 1.05rem;
                margin-top: 0.9rem;
            }

            .request-file-kicker {
                color: var(--brand);
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.35rem;
            }

            .request-file-title {
                color: var(--text);
                font-size: 1.08rem;
                font-weight: 700;
                line-height: 1.35;
                word-break: break-word;
            }

            .request-file-meta {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.45;
                margin-top: 0.25rem;
            }

            .request-file-stats {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.7rem;
                margin-top: 0.9rem;
            }

            .request-file-stat {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(11, 94, 168, 0.08);
                border-radius: 14px;
                padding: 0.72rem 0.78rem;
            }

            .request-file-stat-label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.22rem;
            }

            .request-file-stat-value {
                color: var(--brand-deep);
                font-size: 0.95rem;
                font-weight: 800;
                line-height: 1.25;
            }

            .upload-callout {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.5;
                margin-top: 0.15rem;
            }

            .checklist {
                display: flex;
                flex-direction: column;
                gap: 0.6rem;
                margin-top: 0.2rem;
            }

            .checklist-row {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .checklist-icon {
                width: 1.5rem;
                height: 1.5rem;
                border-radius: 999px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 0.78rem;
                font-weight: 800;
                flex: 0 0 auto;
                border: 1px solid var(--line);
                background: var(--surface);
                color: var(--muted);
            }

            .checklist-row.done .checklist-icon {
                background: var(--good-bg);
                border-color: rgba(19, 107, 72, 0.16);
                color: var(--good);
            }

            .checklist-row.done {
                color: var(--text);
            }

            .muted-note {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .section-kicker {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .section-shell {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 1rem 1.15rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }

            .section-shell h3 {
                margin: 0;
                font-size: 1.28rem;
                color: var(--text);
            }

            .section-shell p {
                margin: 0.45rem 0 0 0;
                color: var(--muted);
                line-height: 1.6;
            }

            .st-key-request_details_panel,
            .st-key-upload_task_panel,
            .st-key-submission_readiness_panel,
            .st-key-intake_summary_panel,
            .st-key-hygiene_summary_panel,
            .st-key-findings_remediation_panel,
            .st-key-remediation_preview_panel,
            .st-key-scan_context_panel,
            .st-key-last_remediation_panel {
                background: var(--surface) !important;
                border: 1px solid var(--line) !important;
                border-radius: 20px !important;
                box-shadow: var(--shadow) !important;
                padding: 1rem 1.15rem !important;
            }

            .st-key-request_details_panel {
                padding-bottom: 1.35rem !important;
            }

            .st-key-upload_task_panel {
                padding-bottom: 0.9rem !important;
            }

            .st-key-intake_summary_panel {
                padding-bottom: 1.45rem !important;
            }

            .st-key-hygiene_summary_panel {
                padding-bottom: 1.4rem !important;
            }

            .st-key-findings_remediation_panel {
                padding-bottom: 1.2rem !important;
            }

            .st-key-remediation_preview_panel {
                padding-bottom: 1.1rem !important;
            }

            .st-key-scan_context_panel {
                padding-bottom: 1rem !important;
            }

            .st-key-request_details_panel > div,
            .st-key-upload_task_panel > div,
            .st-key-submission_readiness_panel > div,
            .st-key-intake_summary_panel > div,
            .st-key-hygiene_summary_panel > div,
            .st-key-findings_remediation_panel > div,
            .st-key-remediation_preview_panel > div,
            .st-key-scan_context_panel > div,
            .st-key-last_remediation_panel > div {
                background: transparent !important;
            }

            .st-key-request_details_panel [data-testid="stTextInput"] {
                margin-bottom: 0.45rem !important;
            }

            .pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.28rem 0.62rem;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                margin-right: 0.4rem;
                margin-bottom: 0.35rem;
                border: 1px solid var(--line);
                background: var(--surface-soft);
                color: var(--brand-deep);
            }

            .state-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1rem 1.05rem;
                box-shadow: var(--shadow);
                min-height: 148px;
            }

            .state-card h4 {
                margin: 0;
                font-size: 0.88rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .state-value {
                font-size: 1.65rem;
                font-weight: 700;
                color: var(--brand);
                margin-top: 0.55rem;
                margin-bottom: 0.25rem;
                line-height: 1.15;
            }

            .state-text {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.94rem;
            }

            .status-good,
            .status-warn,
            .status-bad {
                display: inline-block;
                margin-top: 0.55rem;
                border-radius: 999px;
                padding: 0.22rem 0.6rem;
                font-size: 0.74rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .status-good {
                background: var(--good-bg);
                color: var(--good);
                border: 1px solid rgba(22, 112, 74, 0.22);
            }

            .status-warn {
                background: var(--warn-bg);
                color: var(--warn);
                border: 1px solid rgba(138, 97, 22, 0.22);
            }

            .status-bad {
                background: var(--danger-bg);
                color: var(--danger);
                border: 1px solid rgba(143, 45, 53, 0.22);
            }

            .step-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 0.95rem 0.95rem 0.85rem 0.95rem;
                box-shadow: var(--shadow);
                min-height: 156px;
            }

            .step-card.active {
                border-color: rgba(28, 216, 211, 0.5);
                box-shadow: 0 16px 32px rgba(15, 79, 149, 0.12);
            }

            .step-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 14px;
                background: rgba(15, 79, 149, 0.08);
                color: var(--brand);
                font-weight: 700;
                margin-bottom: 0.6rem;
            }

            .step-title {
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.35rem;
            }

            .step-description {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.9rem;
                min-height: 98px;
                margin-bottom: 0.7rem;
            }

            .step-action-note {
                color: var(--muted);
                font-size: 0.8rem;
                font-weight: 600;
                margin-top: 0.15rem;
            }

            .note-card {
                background: var(--surface-soft);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                line-height: 1.6;
                color: var(--muted);
                margin-bottom: 0.75rem;
            }

            .role-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: var(--shadow);
                min-height: 124px;
            }

            .role-card strong {
                display: block;
                color: var(--text);
                margin-bottom: 0.35rem;
            }

            .approval-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem;
                box-shadow: var(--shadow);
                min-height: 182px;
            }

            .approval-level {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.4rem;
            }

            .approval-card h4 {
                margin: 0 0 0.35rem 0;
                color: var(--text);
                font-size: 1.02rem;
            }

            .approval-owner {
                color: var(--muted);
                font-size: 0.88rem;
                margin-bottom: 0.55rem;
            }

            .approval-detail {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.9rem;
                margin-top: 0.6rem;
            }

            .bullet-list {
                margin: 0.2rem 0 0 0;
                padding-left: 1rem;
                color: var(--muted);
            }

            .bullet-list li {
                margin-bottom: 0.32rem;
                line-height: 1.5;
            }

            .mini-label {
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .steps-wrap {
                margin-top: 0.35rem;
            }

            .step-pill-compact {
                display: inline-block;
                margin-right: 0.32rem;
                margin-bottom: 0.38rem;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                border: 1px solid rgba(11, 94, 168, 0.14);
                background: rgba(11, 94, 168, 0.06);
                color: var(--brand);
                font-size: 0.78rem;
                font-weight: 700;
            }

            .step-nav-meta {
                text-align: center;
                margin-top: 0.15rem;
            }

            .step-nav-owner {
                color: var(--brand-deep);
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .step-nav-status {
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 700;
                margin-top: 0.12rem;
            }

            .workflow-step-card,
            .status-flow-row {
                display: none;
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.7rem 0.85rem;
                box-shadow: var(--shadow);
                min-height: 116px;
            }

            div[data-testid="stMetric"] * {
                color: var(--text) !important;
            }

            div.stButton > button,
            div.stDownloadButton > button {
                min-height: 2.8rem;
                border-radius: 12px;
                border: 1px solid rgba(11, 94, 168, 0.18);
                background: var(--surface) !important;
                color: var(--text) !important;
                font-weight: 700;
                box-shadow: none !important;
            }

            div.stButton > button[kind="primary"] {
                background: var(--brand) !important;
                color: #ffffff !important;
                border: 1px solid rgba(8, 70, 125, 0.28) !important;
            }

            div[data-testid="stTextInput"] label p,
            div[data-testid="stSelectbox"] label p,
            div[data-testid="stCheckbox"] label p {
                color: var(--text) !important;
                font-weight: 600 !important;
            }

            div[data-testid="stTextInput"] input {
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
            }

            div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
                min-height: 3rem !important;
                border-radius: 14px !important;
                border: 1px solid var(--line) !important;
                background: #fbfdff !important;
            }

            div[data-baseweb="button-group"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.35rem;
                box-shadow: var(--shadow);
                margin-bottom: 0.45rem;
            }

            div[data-baseweb="button-group"] button {
                border-radius: 12px !important;
                border: 1px solid transparent !important;
                color: var(--brand-deep) !important;
                font-weight: 800 !important;
                min-height: 3.35rem !important;
                font-size: 1.02rem !important;
                padding: 0.8rem 0.95rem !important;
            }

            div[data-baseweb="button-group"] button[aria-pressed="true"] {
                background: linear-gradient(180deg, var(--brand), var(--brand-deep)) !important;
                color: #ffffff !important;
                box-shadow: 0 10px 20px rgba(11, 94, 168, 0.18) !important;
            }

            div[data-baseweb="button-group"] button:hover {
                background: rgba(11, 94, 168, 0.08) !important;
            }

            div[data-baseweb="select"] > div,
            div[data-testid="stTextArea"] textarea {
                border-radius: 14px;
                background: var(--surface) !important;
                color: var(--text) !important;
                -webkit-text-fill-color: var(--text) !important;
                border: 1px solid var(--line) !important;
            }

            /* Universal text field fix:
               BaseWeb clips borders when height/border are applied to the inner input.
               Put the frame on the wrapper and let the input stay borderless. */
            div[data-testid="stTextInput"] [data-baseweb="input"],
            div[data-testid="stNumberInput"] [data-baseweb="input"] {
                min-height: 3.15rem !important;
                border-radius: 14px !important;
                border: 0 !important;
                background: var(--surface) !important;
                box-shadow: inset 0 0 0 1px var(--line) !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
                background-clip: padding-box !important;
            }

            div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
            div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
                box-shadow:
                    inset 0 0 0 1px rgba(11, 94, 168, 0.62),
                    0 0 0 2px rgba(11, 94, 168, 0.1) !important;
            }

            div[data-testid="stTextInput"] [data-baseweb="input"] > div,
            div[data-testid="stNumberInput"] [data-baseweb="input"] > div {
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
                min-height: calc(3.15rem - 2px) !important;
                display: flex !important;
                align-items: center !important;
            }

            div[data-testid="stTextInput"] input,
            div[data-testid="stNumberInput"] input {
                min-height: auto !important;
                height: auto !important;
                border: 0 !important;
                border-radius: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
                padding: 0 0.9rem !important;
                line-height: 1.35 !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stFileUploader"] section {
                background: var(--surface) !important;
                border: 1px solid var(--line) !important;
                border-radius: 18px !important;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                background: var(--surface) !important;
                border-radius: 18px;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff !important;
                border: 1px solid var(--line) !important;
                border-radius: 18px;
                box-shadow: var(--shadow);
                padding: 0.35rem 0.45rem 0.7rem 0.45rem;
            }
            /* Ensure ALL inner blocks including nested containers are transparent (no blue tint) */
            html body div[data-testid="stVerticalBlockBorderWrapper"],
            html body div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"],
            html body div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
                background: #ffffff !important;
            }
            html body div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="element-container"],
            html body div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] {
                background: transparent !important;
            }

            div[data-testid="stDataFrame"] *,
            div[data-testid="stTable"] * {
                color: var(--text) !important;
            }

            .control-card-kicker {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.3rem;
            }

            .control-card-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.28rem;
            }

            .control-card-text {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.93rem;
                margin-bottom: 0.9rem;
            }

            .control-compact-note {
                color: var(--muted);
                line-height: 1.5;
                font-size: 0.9rem;
                margin-top: 0.2rem;
            }

            .generate-action-card {
                background: linear-gradient(180deg, rgba(15, 79, 149, 0.05), rgba(28, 216, 211, 0.04)), var(--surface);
                border: 1px solid rgba(15, 79, 149, 0.16);
                border-radius: 20px;
                padding: 1rem 1rem 0.35rem 1rem;
                margin-bottom: 0.8rem;
            }

            .generate-action-title {
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.28rem;
            }

            .generate-action-text {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.93rem;
                margin-bottom: 0.75rem;
            }

            @media (max-width: 1100px) {
                .topbar-shell {
                    flex-direction: column;
                }

                .session-rail,
                .summary-grid,
                .summary-grid.three,
                .group-strip,
                .workflow-progress {
                    grid-template-columns: 1fr;
                    min-width: 0;
                }

                .brand-lockup {
                    flex-direction: column;
                    align-items: flex-start;
                }

                .brand-logo {
                    width: 240px;
                }

                .login-brand-card,
                .login-card {
                    min-height: auto;
                }

                .login-links {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.5rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes)).replace(r"^\s*$", pd.NA, regex=True)


@st.cache_data(show_spinner=False)
def load_sample_dataframe() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH).replace(r"^\s*$", pd.NA, regex=True)


def build_generation_signature(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> str:
    return json.dumps({"metadata": metadata, "controls": controls}, sort_keys=True, default=str)


def build_metadata_signature(metadata: list[dict[str, Any]]) -> str:
    return json.dumps(metadata, sort_keys=True, default=str)


def has_permission(permission: str) -> bool:
    role = st.session_state.get("current_role")
    if not role:
        return False
    return permission in ROLE_CONFIGS[role]["permissions"]


def current_role_summary() -> str:
    role = st.session_state.get("current_role", "")
    return ROLE_CONFIGS.get(role, {}).get("summary", "")


def current_role_group() -> str:
    role = st.session_state.get("current_role", "")
    return ROLE_TO_GROUP.get(role, "Users")


def visible_steps_for_role(role: str | None = None) -> list[int]:
    active_role = role or st.session_state.get("current_role")
    if not active_role:
        return list(range(len(STEP_CONFIG)))
    return ROLE_VISIBLE_STEPS.get(active_role, list(range(len(STEP_CONFIG))))


def format_timestamp() -> str:
    return datetime.now().strftime("%H:%M")


def record_audit_event(event: str, detail: str, status: str = "Logged") -> None:
    if "audit_events" not in st.session_state:
        st.session_state.audit_events = []
    st.session_state.audit_events.insert(
        0,
        {
            "time": format_timestamp(),
            "actor": st.session_state.get("current_role", "System"),
            "event": agent_event_label(event),
            "detail": detail,
            "status": status,
        },
    )
    st.session_state.audit_events = st.session_state.audit_events[:30]
    persist_shared_workspace_state()


def clear_generation_outputs() -> None:
    st.session_state.synthetic_df = None
    st.session_state.generation_summary = None
    st.session_state.validation = None
    st.session_state.last_generation_signature = None
    st.session_state.release_status = "Not ready"
    st.session_state.export_requested_by = None
    st.session_state.export_policy_approved_by = None
    st.session_state.export_approved_by = None
    st.session_state.results_shared_by = None
    st.session_state.results_shared_at = None


def set_source_dataframe(df: pd.DataFrame, label: str) -> None:
    existing_file_size = st.session_state.get("source_file_size")
    profile = profile_dataframe(df)
    hygiene = review_hygiene(df, profile)
    metadata = build_metadata(df, profile)

    st.session_state.source_df = df
    st.session_state.source_label = label
    st.session_state.source_file_size = existing_file_size
    st.session_state.profile = profile
    st.session_state.hygiene = hygiene
    st.session_state.metadata_editor_df = metadata_to_editor_frame(metadata)
    st.session_state.controls = default_generation_controls(len(df))
    st.session_state.intake_confirmed = False
    st.session_state.hygiene_reviewed = False
    st.session_state.settings_reviewed = False
    st.session_state.settings_review_signature = None
    st.session_state.metadata_status = "Draft"
    st.session_state.metadata_submitted_by = None
    st.session_state.metadata_submitted_at = None
    st.session_state.metadata_approved_by = None
    st.session_state.metadata_approved_at = None
    st.session_state.metadata_review_note = None
    st.session_state.metadata_reviewed_by = None
    st.session_state.metadata_reviewed_at = None
    st.session_state.last_reviewed_metadata_signature = None
    st.session_state.last_cleaning_actions = []
    st.session_state.current_metadata_package_id = None
    st.session_state.metadata_package_log = []
    st.session_state.metadata_has_unsubmitted_changes = False
    clear_generation_outputs()
    st.session_state.current_step = 0
    record_audit_event("Dataset loaded", label, status="Logged")


def initialize_app_state() -> None:
    defaults = {
        "authenticated": False,
        "current_role": None,
        "current_user_email": None,
        "project_purpose": "",
        "source_file_size": None,
        "shared_workspace_loaded": False,
        "current_step": 0,
        "audit_events": [],
        "release_status": "Not ready",
        "export_requested_by": None,
        "export_policy_approved_by": None,
        "export_approved_by": None,
        "results_shared_by": None,
        "results_shared_at": None,
        "uploaded_signature": None,
        "metadata_submitted_at": None,
        "metadata_approved_at": None,
        "metadata_review_note": None,
        "metadata_reviewed_by": None,
        "metadata_reviewed_at": None,
        "current_metadata_package_id": None,
        "metadata_package_log": [],
        "metadata_has_unsubmitted_changes": False,
        "settings_reviewed": False,
        "settings_review_signature": None,
        "request_registry": [],
        "active_request_id": None,
        "pending_active_request_selector": None,
        "pending_queue_clear": False,
        "next_request_number": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.shared_workspace_loaded:
        load_shared_workspace_state()
        st.session_state.shared_workspace_loaded = True


def ensure_dataset_loaded() -> None:
    if not st.session_state.get("request_registry"):
        create_blank_request()
    elif "source_df" not in st.session_state and st.session_state.get("active_request_id"):
        restore_request_workspace(st.session_state["active_request_id"])


def process_pending_workspace_actions() -> None:
    if st.session_state.get("pending_queue_clear"):
        st.session_state.pending_queue_clear = False
        clear_request_queue()


def effective_release_status(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> str:
    if not has_active_dataset():
        return "Awaiting dataset"
    if not st.session_state.intake_confirmed:
        return "Uploaded"
    if not st.session_state.hygiene_reviewed:
        return "Scanned"
    if not st.session_state.settings_reviewed:
        return "Reviewed"
    if st.session_state.metadata_status == "In Review":
        return "Pending approval"
    if st.session_state.metadata_status == "Changes Requested":
        return "Changes requested"
    if st.session_state.metadata_status == "Rejected":
        return "Rejected"
    if st.session_state.metadata_status != "Approved":
        return "Reviewed"
    if st.session_state.synthetic_df is None:
        return "Approved"
    if has_stale_generation(metadata, controls):
        return "Regeneration required"
    if st.session_state.results_shared_at:
        return "Shared"
    return "Generated"


def metadata_sensitivity(item: dict[str, Any]) -> str:
    lowered = item["column"].lower()
    if item["data_type"] == "identifier":
        return "Restricted"
    if "postal" in lowered or item["data_type"] == "date":
        return "Sensitive"
    if "complaint" in lowered or "note" in lowered or "text" in lowered:
        return "Sensitive"
    return "Operational"


def metadata_owner(item: dict[str, Any]) -> str:
    sensitivity = metadata_sensitivity(item)
    if sensitivity in {"Restricted", "Sensitive"}:
        return "Manager / Reviewer"
    return "Data Analyst"


def metadata_handling(item: dict[str, Any]) -> str:
    lowered = item["column"].lower()
    if not item["include"]:
        return "Excluded from generation."
    action = item.get("control_action", "")
    if action == "Tokenize" or item["data_type"] == "identifier":
        return "Replace with surrogate token before synthetic export."
    if action == "Month only":
        return "Release month-level timing only instead of exact dates."
    if action == "Date shift" or item["data_type"] == "date":
        return "Apply controlled date jitter before release."
    if action == "Coarse geography" or "postal" in lowered:
        return "Keep only coarse geography in synthetic output."
    if action == "Group text":
        return "Collapse free text into safer grouped categories before sampling."
    if action == "Group rare categories":
        return "Keep major categories and group low-frequency labels into 'Other'."
    if action == "Clip extremes":
        return "Clip extreme numeric values before sampling to reduce outlier leakage."
    if "complaint" in lowered:
        return "Normalize or group text before sampling."
    if item["strategy"] == "sample_plus_noise":
        return "Sample source behavior with bounded noise."
    return "Sample approved categories while preserving broad distribution."


def metadata_status_for_row(item: dict[str, Any]) -> str:
    if not item["include"]:
        return "Excluded"
    if st.session_state.metadata_status == "Approved":
        return "Approved"
    if st.session_state.metadata_status == "In Review":
        return "In review"
    if st.session_state.metadata_status == "Changes Requested":
        return "Changes requested"
    if st.session_state.metadata_status == "Rejected":
        return "Rejected"
    return "Draft"


def build_metadata_review_frame(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        rows.append(
            {
                "Field": item["column"],
                "Field type": item["data_type"].replace("_", " ").title(),
                "Sensitivity": metadata_sensitivity(item),
                "Field action": item.get("control_action", "Preserve"),
                "Generation rule": item["strategy"].replace("_", " ").title(),
                "Owner": metadata_owner(item),
                "Status": metadata_status_for_row(item),
                "Handling": metadata_handling(item),
            }
        )
    return pd.DataFrame(rows)


def build_phi_detection_frame(profile: dict[str, Any], metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        column = item["column"]
        details = profile["columns"][column]
        lowered = column.lower()
        if details["semantic_role"] == "identifier":
            rows.append(
                {
                    "Field": column,
                    "Detection": "Direct identifier",
                    "Why flagged": "Encounter-level ID pattern detected.",
                    "Control point": "Restricted to raw workspace; replaced with surrogate tokens after approval.",
                }
            )
        elif details["semantic_role"] == "date":
            rows.append(
                {
                    "Field": column,
                    "Detection": "Date or timing field",
                    "Why flagged": "Exact encounter timing can support re-identification when combined with other fields.",
                    "Control point": "Dates are jittered after governance approval and before synthetic release.",
                }
            )
        elif "postal" in lowered:
            rows.append(
                {
                    "Field": column,
                    "Detection": "Geographic quasi-identifier",
                    "Why flagged": "Location fields increase linkage risk when combined with dates or demographics.",
                    "Control point": "Only coarse geography is preserved in the synthetic dataset.",
                }
            )
        elif "complaint" in lowered or "note" in lowered or "text" in lowered:
            rows.append(
                {
                    "Field": column,
                    "Detection": "Free-text clinical context",
                    "Why flagged": "Free text can contain inconsistent or unexpectedly identifying detail.",
                    "Control point": "Normalize or group before sampling and keep under governance review.",
                }
            )

    if not rows:
        rows.append(
            {
                "Field": "No high-risk field detected",
                "Detection": "Operational schema",
                "Why flagged": "This sample contains no obvious direct PHI columns beyond identifiers.",
                "Control point": "Continue with metadata review and validation.",
            }
        )

    return pd.DataFrame(rows)


def build_missingness_strategy_frame(profile: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for column, details in profile["columns"].items():
        missing_pct = float(details["missing_pct"])
        if missing_pct <= 0:
            continue

        role = details["semantic_role"]
        if missing_pct >= 25:
            strategy = "Escalate before approval"
            rationale = "High missingness can distort downstream analysis and should be reviewed explicitly."
        elif role in {"categorical", "binary"}:
            strategy = "Preserve as explicit missing / unknown"
            rationale = "Keeping gaps visible helps analysts understand sparse operational coding."
        elif role == "numeric":
            strategy = "Preserve pattern unless business rule exists"
            rationale = "Avoid silent filling unless the data owner agrees on an imputation rule."
        elif role == "date":
            strategy = "Preserve timing gaps"
            rationale = "Missing dates can be operationally meaningful and should stay visible in validation."
        else:
            strategy = "Review and document"
            rationale = "Track the gap and decide handling during metadata review."

        rows.append(
            {
                "Field": column,
                "Missing %": missing_pct,
                "Recommended strategy": strategy,
                "Why": rationale,
            }
        )

    if not rows:
        rows.append(
            {
                "Field": "No incomplete field",
                "Missing %": 0.0,
                "Recommended strategy": "No action required",
                "Why": "The current dataset does not contain missing values.",
            }
        )

    return pd.DataFrame(rows).sort_values(by="Missing %", ascending=False, ignore_index=True)


def build_hygiene_option_defaults(hygiene: dict[str, Any]) -> dict[str, bool]:
    concerns = {issue["concern"] for issue in hygiene.get("issues", [])}
    findings = " ".join(issue["finding"].lower() for issue in hygiene.get("issues", []))
    return {
        "standardize_blank_strings": "Missingness" in concerns,
        "remove_duplicates": "Duplicate rows" in concerns,
        "normalize_categories": "Category normalization" in concerns,
        "fill_operational_gaps": "Missingness" in concerns,
        "fix_negative_values": "Negative values" in concerns,
        "repair_invalid_dates": "Invalid dates" in concerns,
        "cap_numeric_extremes": "Outliers" in concerns or "extreme" in findings,
        "group_rare_categories": False,
    }


def summarize_dataframe_change(original_df: pd.DataFrame, updated_df: pd.DataFrame) -> dict[str, int]:
    common_rows = min(len(original_df), len(updated_df))
    common_columns = [column for column in original_df.columns if column in updated_df.columns]
    changed_cells = 0
    changed_columns = 0
    if common_rows and common_columns:
        original_view = original_df.loc[: common_rows - 1, common_columns].reset_index(drop=True).astype("string").fillna("<NA>")
        updated_view = updated_df.loc[: common_rows - 1, common_columns].reset_index(drop=True).astype("string").fillna("<NA>")
        differences = original_view.ne(updated_view)
        changed_cells = int(differences.sum().sum()) + abs(len(original_df) - len(updated_df)) * len(common_columns)
        changed_columns = int(differences.any(axis=0).sum())
    return {
        "rows_before": int(len(original_df)),
        "rows_after": int(len(updated_df)),
        "changed_cells": int(changed_cells),
        "changed_columns": int(changed_columns),
        "missing_before": int(original_df.isna().sum().sum()),
        "missing_after": int(updated_df.isna().sum().sum()),
    }


def summarize_metadata_package(metadata: list[dict[str, Any]]) -> dict[str, int]:
    included = sum(1 for item in metadata if item["include"])
    excluded = sum(1 for item in metadata if not item["include"])
    restricted = sum(1 for item in metadata if item["include"] and metadata_sensitivity(item) == "Restricted")
    sensitive = sum(1 for item in metadata if item["include"] and metadata_sensitivity(item) == "Sensitive")
    targeted = sum(1 for item in metadata if item["include"] and item.get("control_action", "Preserve") != "Preserve")
    return {
        "included_fields": included,
        "excluded_fields": excluded,
        "restricted_fields": restricted,
        "sensitive_fields": sensitive,
        "targeted_actions": targeted,
    }


def register_metadata_submission(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    package_id = f"PKG-{datetime.now().strftime('%H%M%S')}-{len(st.session_state.metadata_package_log) + 1:02d}"
    record = {
        "package_id": package_id,
        "signature": build_metadata_signature(metadata),
        "snapshot": metadata,
        "summary": summarize_metadata_package(metadata),
        "submitted_by": st.session_state.current_role,
        "submitted_at": format_timestamp(),
        "approved_by": None,
        "approved_at": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "review_note": None,
        "status": "In Review",
    }
    st.session_state.metadata_package_log.insert(0, record)
    st.session_state.current_metadata_package_id = package_id
    return record


def register_metadata_approval(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    signature = build_metadata_signature(metadata)
    target_id = st.session_state.get("current_metadata_package_id")
    for record in st.session_state.metadata_package_log:
        if (target_id and record["package_id"] == target_id) or record["signature"] == signature:
            record["approved_by"] = st.session_state.current_role
            record["approved_at"] = format_timestamp()
            record["reviewed_by"] = st.session_state.current_role
            record["reviewed_at"] = format_timestamp()
            record["status"] = "Approved"
            st.session_state.current_metadata_package_id = record["package_id"]
            return record

    record = register_metadata_submission(metadata)
    record["approved_by"] = st.session_state.current_role
    record["approved_at"] = format_timestamp()
    record["reviewed_by"] = st.session_state.current_role
    record["reviewed_at"] = format_timestamp()
    record["status"] = "Approved"
    return record


def register_metadata_feedback(metadata: list[dict[str, Any]], outcome: str, note: str) -> dict[str, Any] | None:
    signature = build_metadata_signature(metadata)
    target_id = st.session_state.get("current_metadata_package_id")
    for record in st.session_state.metadata_package_log:
        if (target_id and record["package_id"] == target_id) or record["signature"] == signature:
            record["status"] = outcome
            record["reviewed_by"] = st.session_state.current_role
            record["reviewed_at"] = format_timestamp()
            record["review_note"] = note.strip()
            return record
    return None


def active_metadata_package_record(metadata: list[dict[str, Any]]) -> dict[str, Any] | None:
    target_id = st.session_state.get("current_metadata_package_id")
    if target_id:
        for record in st.session_state.metadata_package_log:
            if record["package_id"] == target_id:
                return record

    signature = build_metadata_signature(metadata)
    for record in st.session_state.metadata_package_log:
        if record["signature"] == signature:
            return record
    return st.session_state.metadata_package_log[0] if st.session_state.metadata_package_log else None


def current_review_package_record() -> dict[str, Any] | None:
    target_id = st.session_state.get("current_metadata_package_id")
    if target_id:
        for record in st.session_state.metadata_package_log:
            if record["package_id"] == target_id and record["status"] == "In Review":
                return record
    for record in st.session_state.metadata_package_log:
        if record["status"] == "In Review":
            return record
    return None


def build_metadata_package_log_frame() -> pd.DataFrame:
    rows = []
    for record in st.session_state.metadata_package_log:
        summary = record["summary"]
        rows.append(
            {
                "Package": record["package_id"],
                "Status": record["status"],
                "Submitted by": record["submitted_by"],
                "Submitted at": record["submitted_at"],
                "Approved by": record["approved_by"] or "Not approved",
                "Approved at": record["approved_at"] or "Not approved",
                "Reviewed by": record.get("reviewed_by") or "Not reviewed",
                "Review note": record.get("review_note") or "None",
                "Included fields": summary["included_fields"],
                "Restricted": summary["restricted_fields"],
                "Targeted actions": summary["targeted_actions"],
            }
        )
    return pd.DataFrame(rows)


def has_unsubmitted_metadata_changes(metadata: list[dict[str, Any]]) -> bool:
    reviewed_signature = st.session_state.get("last_reviewed_metadata_signature")
    if not reviewed_signature:
        return False
    return build_metadata_signature(metadata) != reviewed_signature


def metadata_display_status(metadata: list[dict[str, Any]]) -> str:
    if has_unsubmitted_metadata_changes(metadata):
        return "Draft changes"
    return st.session_state.metadata_status


def build_work_in_progress_frame(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> pd.DataFrame:
    active_package = active_metadata_package_record(metadata)
    review_package = current_review_package_record()
    if not has_active_dataset():
        request_status = "Awaiting upload"
        request_owner = "Data Analyst"
        request_note = "No source dataset is in the workspace yet."
    else:
        request_status = "Uploaded" if st.session_state.intake_confirmed else "Draft"
        request_owner = "Data Analyst"
        request_note = current_dataset_label()

    if review_package is not None:
        review_status = f"{review_package['package_id']} pending"
        review_owner = "Manager / Reviewer"
        review_note = f"Submitted by {review_package['submitted_by']} at {review_package['submitted_at']}."
    elif active_package is not None and active_package["status"] == "Approved":
        review_status = f"{active_package['package_id']} approved"
        review_owner = "Manager / Reviewer"
        review_note = f"Approved by {active_package['approved_by']} at {active_package['approved_at']}."
    elif active_package is not None and active_package["status"] in {"Changes Requested", "Rejected"}:
        review_status = f"{active_package['package_id']} · {active_package['status']}"
        review_owner = "Data Analyst"
        review_note = active_package.get("review_note") or "Revision is required before resubmission."
    elif st.session_state.settings_reviewed:
        review_status = "Ready to submit"
        review_owner = "Data Analyst"
        review_note = "Data settings are ready to be sent for manager review."
    else:
        review_status = "Pending settings review"
        review_owner = "Data Analyst"
        review_note = "Finish reviewing scan findings and field settings first."

    if st.session_state.synthetic_df is None:
        output_status = "Not generated"
        output_owner = "System"
        output_note = "Synthetic output will appear after the request is approved."
    elif has_stale_generation(metadata, controls):
        output_status = "Out of date"
        output_owner = "Data Analyst"
        output_note = "Settings changed after generation. Run generation again."
    elif st.session_state.results_shared_at:
        output_status = "Shared"
        output_owner = "Data Analyst"
        output_note = f"Marked shared by {st.session_state.results_shared_by} at {st.session_state.results_shared_at}."
    else:
        output_status = "Generated"
        output_owner = "Data Analyst"
        output_note = "Synthetic output is ready for final review and download."

    rows = [
        {
            "Work item": "Current request",
            "Status": request_status,
            "Current owner": request_owner,
            "Latest update": request_note,
        },
        {
            "Work item": "Review queue",
            "Status": review_status,
            "Current owner": review_owner,
            "Latest update": review_note,
        },
        {
            "Work item": "Final output",
            "Status": output_status,
            "Current owner": output_owner,
            "Latest update": output_note,
        },
    ]
    return pd.DataFrame(rows)


def build_role_access_frame() -> pd.DataFrame:
    rows = []
    for role, config in ROLE_CONFIGS.items():
        permissions = config["permissions"]
        rows.append(
            {
                "Stakeholder group": ROLE_TO_GROUP[role],
                "Role": role,
                "Visible workflow": ", ".join(STEP_CONFIG[index]["title"] for index in visible_steps_for_role(role)),
                "Raw data access": "Yes" if "view_raw" in permissions else "No",
                "Metadata edits": "Yes" if "edit_metadata" in permissions else "No",
                "Metadata approval": "Yes" if "approve_metadata" in permissions else "No",
                "Synthetic generation": "Yes" if "generate" in permissions else "No",
                "Controlled release review": "Yes" if "approve_release_policy" in permissions else "No",
                "Final export approval": "Yes" if "approve_export" in permissions else "No",
            }
        )
    return pd.DataFrame(rows)


def field_action_options(item: dict[str, Any]) -> list[str]:
    lowered = item["column"].lower()
    role = item["data_type"]
    if role == "identifier":
        return ["Tokenize", "Exclude"]
    if role == "date":
        return ["Date shift", "Month only", "Exclude"]
    if "postal" in lowered or "address" in lowered:
        return ["Coarse geography", "Exclude", "Preserve"]
    if "complaint" in lowered or "note" in lowered or "text" in lowered:
        return ["Group text", "Exclude", "Preserve"]
    if role == "numeric":
        return ["Preserve", "Clip extremes", "Exclude"]
    return ["Preserve", "Group rare categories", "Exclude"]


ALL_CONTROL_ACTIONS = [
    "Preserve",
    "Tokenize",
    "Date shift",
    "Month only",
    "Coarse geography",
    "Group text",
    "Group rare categories",
    "Clip extremes",
    "Exclude",
]


def sanitize_control_action(item: dict[str, Any], chosen_action: str) -> str:
    allowed = field_action_options(item)
    if chosen_action in allowed:
        return chosen_action
    return allowed[0]


def normalize_metadata_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    action = sanitize_control_action(normalized, normalized.get("control_action", "Preserve") or "Preserve")
    normalized["control_action"] = action

    if action == "Exclude":
        normalized["include"] = False
        normalized["notes"] = "Excluded from the synthetic package during metadata review."
        return normalized

    normalized["include"] = True
    if action == "Tokenize":
        normalized["data_type"] = "identifier"
        normalized["strategy"] = "new_token"
    elif action in {"Date shift", "Month only"}:
        normalized["data_type"] = "date"
        normalized["strategy"] = "sample_plus_jitter"
    elif action in {"Coarse geography", "Group text", "Group rare categories"}:
        normalized["strategy"] = "sample_category"
    elif action == "Clip extremes":
        normalized["strategy"] = "sample_plus_noise"

    return normalized


def normalize_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = [normalize_metadata_item(item) for item in editor_frame_to_metadata(frame)]
    normalized = metadata_to_editor_frame(metadata)
    normalized.index = frame.index
    return normalized


def apply_bulk_metadata_profile(mode: str) -> None:
    frame = st.session_state.metadata_editor_df.copy()
    for index, row in frame.iterrows():
        column_name = str(row["column"]).lower()
        role = str(row["data_type"])
        if mode == "tighten_phi":
            if role == "identifier":
                frame.at[index, "control_action"] = "Exclude"
            elif role == "date":
                frame.at[index, "control_action"] = "Month only"
            elif "postal" in column_name or "address" in column_name:
                frame.at[index, "control_action"] = "Coarse geography"
            elif "complaint" in column_name or "note" in column_name or "text" in column_name:
                frame.at[index, "control_action"] = "Group text"
        elif mode == "preserve_analytics":
            if role == "identifier":
                frame.at[index, "control_action"] = "Tokenize"
            elif role == "date":
                frame.at[index, "control_action"] = "Date shift"
            elif role == "numeric":
                frame.at[index, "control_action"] = "Preserve"
            elif "postal" in column_name or "address" in column_name:
                frame.at[index, "control_action"] = "Coarse geography"
            elif "complaint" in column_name or "note" in column_name or "text" in column_name:
                frame.at[index, "control_action"] = "Group text"
            else:
                frame.at[index, "control_action"] = "Preserve"
        elif mode == "reset_defaults":
            metadata_defaults = build_metadata(st.session_state.source_df, st.session_state.profile)
            st.session_state.metadata_editor_df = metadata_to_editor_frame(metadata_defaults)
            persist_shared_workspace_state()
            return

    st.session_state.metadata_editor_df = normalize_metadata_frame(frame)
    persist_shared_workspace_state()


def apply_generation_preset(controls: dict[str, Any], preset: str) -> dict[str, Any]:
    updated = dict(controls)
    config = GENERATION_PRESETS.get(preset)
    if not config:
        updated["generation_preset"] = "Custom"
        return updated

    updated.update(config)
    updated["generation_preset"] = preset
    return updated


def sync_generation_preset_label(controls: dict[str, Any]) -> dict[str, Any]:
    updated = dict(controls)
    preset_name = updated.get("generation_preset", "Balanced")
    if preset_name in GENERATION_PRESETS:
        preset = GENERATION_PRESETS[preset_name]
        matches = all(updated.get(key) == value for key, value in preset.items())
        if not matches:
            updated["generation_preset"] = "Custom"
    return updated


def build_quick_controls_frame(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        rows.append(
            {
                "Field": item["column"],
                "Sensitivity": metadata_sensitivity(item),
                "Field type": item["data_type"].replace("_", " ").title(),
                "Field action": item.get("control_action", "Preserve"),
                "Included": bool(item["include"]),
                "What will happen": metadata_handling(item),
            }
        )
    quick_frame = pd.DataFrame(rows)
    sensitivity_order = {"Restricted": 0, "Sensitive": 1, "Operational": 2}
    quick_frame["_order"] = quick_frame["Sensitivity"].map(sensitivity_order).fillna(3)
    quick_frame = quick_frame.sort_values(by=["_order", "Field"]).drop(columns="_order")
    return quick_frame


STEP_PERMISSION_LABELS: dict[int, list[tuple[str, str]]] = {
    0: [
        ("upload", "Upload or replace the source dataset"),
        ("view_raw", "Inspect row-level raw source data"),
    ],
    1: [
        ("remediate", "Apply safe source-data fixes"),
    ],
    2: [
        ("edit_metadata", "Edit metadata rules and field handling"),
        ("submit_metadata", "Submit the metadata package for review"),
        ("approve_metadata", "Approve the metadata package"),
    ],
    3: [
        ("generate", "Run synthetic generation"),
        ("rollback", "Discard a run and go back upstream"),
    ],
    4: [
        ("validate", "Run a new validation pass"),
        ("request_export", "Request controlled release"),
        ("view_audit", "Review audit and validation history"),
    ],
    5: [
        ("request_export", "Request governed release"),
        ("approve_release_policy", "Review release controls"),
        ("approve_export", "Authorize final export"),
        ("view_audit", "Review audit and approval history"),
    ],
}


ROLE_PRIORITY_NOTES: dict[str, str] = {
    "Clinician / Clinical Lead": "Focus on whether the synthetic dataset still reflects clinically plausible patterns without exposing raw patient detail.",
    "Data Analyst": "Focus on data quality, metadata choices, column behavior, and whether the synthetic output remains useful for analysis.",
    "Data Steward": "Focus on request scope, metadata completeness, field ownership, and whether the package remains useful for hospital analytics.",
    "Privacy Officer": "Focus on sensitive-field handling, privacy risk, and whether governance evidence is sufficient before use.",
    "IT / Security Admin": "Focus on export controls, audit traceability, and whether release stays scoped to approved channels.",
    "Executive Approver": "Focus on whether the package is ready for final governed release and whether the control record is complete.",
}


STEP_EXPLANATION_NOTES: dict[int, str] = {
    0: "This step captures the request, places the dataset in the workspace, and makes clear who can inspect source rows.",
    1: "This step shows the automated scan for PHI, data quality issues, missingness, and fields that need closer handling.",
    2: "This step turns profiled schema into a governance-reviewed metadata package with handling rules and approval status.",
    3: "This step uses the approved package to generate synthetic output with visible distribution, structure, and noise controls.",
    4: "This step shows whether the synthetic package is still useful for analysis while keeping privacy risk understandable.",
    5: "This step keeps export permission-based, auditable, and controlled by release owners rather than the generation team.",
}


def current_owner_checkpoint(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> tuple[str, str]:
    if not has_active_dataset():
        return "Next step", "Upload a source dataset to open a new request."
    if not st.session_state.intake_confirmed:
        return "Next step", "Submit the uploaded dataset into the workspace."
    if not st.session_state.hygiene_reviewed:
        return "Next step", "Review the scan findings and confirm any safe fixes."
    if not st.session_state.settings_reviewed:
        return "Next step", "Finalize the data settings before sending the request for review."
    if st.session_state.metadata_status == "Changes Requested":
        note = st.session_state.metadata_review_note or "The reviewer sent this request back for revision."
        return "Next step", f"Revise the data settings and resubmit. {note}"
    if st.session_state.metadata_status == "Rejected":
        note = st.session_state.metadata_review_note or "The reviewer rejected this request."
        return "Next step", f"Update the request and submit a new version. {note}"
    if has_unsubmitted_metadata_changes(metadata):
        return "Next step", "A revised settings draft exists. Submit it again for review."
    if st.session_state.metadata_status == "Draft":
        return "Next step", "Submit the reviewed settings to the Manager / Reviewer."
    if st.session_state.metadata_status == "In Review":
        review_package = current_review_package_record()
        if review_package is not None:
            return "Next step", f"Manager / Reviewer should approve or reject package {review_package['package_id']}."
        return "Next step", "Manager / Reviewer should review the submitted request."
    if st.session_state.synthetic_df is None or has_stale_generation(metadata, controls):
        return "Next step", "Generate the synthetic dataset from the approved request."
    if not st.session_state.results_shared_at:
        return "Next step", "Download and share the synthetic results."
    return "Current state", "The current request has been generated and marked as shared."


def build_role_status_lists(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, list[str] | str]:
    available: list[str] = []
    waiting: list[str] = []
    completed: list[str] = []
    visible_labels = [STEP_CONFIG[index]["title"] for index in visible_steps_for_role()]

    if not has_active_dataset():
        if has_permission("upload"):
            if not st.session_state.get("project_purpose", "").strip():
                available.append("Add a short project purpose for the request.")
            available.append("Upload a source dataset to start the current request.")
        else:
            waiting.append("Waiting on the Data Analyst to upload a source dataset.")
        label, next_action = current_owner_checkpoint(metadata, controls)
        return {
            "label": label,
            "next_action": next_action,
            "visible_steps": visible_labels,
            "available": available or ["No direct action is required from this role at the moment."],
            "waiting": waiting or ["No external blockers are active right now."],
            "completed": ["No request has been started yet in this workspace."],
        }

    if st.session_state.intake_confirmed:
        completed.append("Request submitted.")
    else:
        if has_permission("upload"):
            if not st.session_state.get("project_purpose", "").strip():
                available.append("Add a short project purpose for this request.")
            available.append("Upload or replace the source dataset if needed.")
        available.append("Submit the uploaded dataset into the workflow.")

    if st.session_state.hygiene_reviewed:
        completed.append("System scan reviewed.")
    else:
        if has_permission("remediate"):
            available.append("Review scan findings and apply safe fixes.")
        else:
            waiting.append("Waiting on the Data Analyst to complete the scan review.")

    if st.session_state.settings_reviewed:
        completed.append("Data settings reviewed.")
    elif has_permission("edit_metadata"):
        available.append("Review field actions and finalize the data settings.")
    else:
        waiting.append("Waiting on the Data Analyst to finalize data settings.")

    if st.session_state.metadata_status == "Approved":
        completed.append(
            f"Metadata package approved{f' by {st.session_state.metadata_approved_by}' if st.session_state.metadata_approved_by else ''}."
        )
    elif st.session_state.metadata_status in {"Changes Requested", "Rejected"}:
        note = st.session_state.metadata_review_note or "The reviewer requested a revision."
        if has_permission("edit_metadata") or has_permission("submit_metadata"):
            available.append(f"Review the feedback and revise the request. {note}")
        else:
            waiting.append("Waiting on the Data Analyst to revise and resubmit the request.")
    elif st.session_state.metadata_status == "In Review":
        if has_permission("approve_metadata"):
            record = current_review_package_record() or active_metadata_package_record(metadata)
            available.append(
                f"Approve or reject request {record['package_id']}." if record else "Review the submitted request."
            )
        else:
            record = current_review_package_record()
            waiting.append(
                f"Waiting on the Manager / Reviewer to review package {record['package_id']}."
                if record
                else "Waiting on the Manager / Reviewer to review the submitted request."
            )
    elif st.session_state.settings_reviewed:
        if has_permission("edit_metadata"):
            available.append("Submit the reviewed settings for manager approval.")
        if has_permission("submit_metadata"):
            available.append("Submit the request for review.")

    if has_unsubmitted_metadata_changes(metadata):
        if has_permission("edit_metadata") or has_permission("submit_metadata"):
            available.append("Review the revised settings draft and resubmit the request.")
        waiting.append("The current draft no longer matches the last submitted or approved version.")

    if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls):
        completed.append("Synthetic dataset generated.")
    elif st.session_state.metadata_status == "Approved" and not has_unsubmitted_metadata_changes(metadata):
        if has_permission("generate"):
            available.append("Generate a new synthetic dataset from the approved package.")
        else:
            waiting.append("Waiting on the Data Analyst to generate synthetic data.")

    if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls):
        if st.session_state.results_shared_at:
            completed.append(f"Results marked shared by {st.session_state.results_shared_by}.")
        elif has_permission("share_results"):
            available.append("Download the synthetic dataset and mark results shared.")
        else:
            waiting.append("Waiting on the Data Analyst to share the final results.")

    if not available:
        available = ["No direct action is required from this role at the moment."]
    if not waiting:
        waiting = ["No external blockers are active right now."]

    label, next_action = current_owner_checkpoint(metadata, controls)
    return {
        "label": label,
        "next_action": next_action,
        "visible_steps": visible_labels,
        "available": available,
        "waiting": waiting,
        "completed": completed or ["No workflow checkpoints have been completed yet in this session."],
    }


def build_role_guidance(role: str, step_index: int) -> dict[str, Any]:
    permissions = ROLE_CONFIGS[role]["permissions"]
    allowed = [label for permission, label in STEP_PERMISSION_LABELS.get(step_index, []) if permission in permissions]
    blocked = [label for permission, label in STEP_PERMISSION_LABELS.get(step_index, []) if permission not in permissions]

    if not allowed:
        allowed = ["Review the current state, summaries, and controls without changing workflow state."]

    if not blocked:
        blocked = ["No major restrictions at this step."]

    return {
        "role": role,
        "summary": ROLE_CONFIGS[role]["summary"],
        "priority": ROLE_PRIORITY_NOTES[role],
        "step_note": STEP_EXPLANATION_NOTES[step_index],
        "allowed": allowed,
        "blocked": blocked,
    }


def build_work_in_progress_cards(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    active_package = active_metadata_package_record(metadata)
    review_package = current_review_package_record()
    package_status = "No package submitted"
    package_detail = "Metadata is still in draft."
    package_owner = "Users"

    if not has_active_dataset():
        package_status = "Awaiting dataset upload"
        package_owner = "Users"
        package_detail = "Upload a CSV before a metadata package can enter the workflow."
    elif review_package is not None:
        package_status = f"{review_package['package_id']} · In Review"
        package_owner = "Governance"
        package_detail = (
            f"Submitted by {review_package['submitted_by']} at {review_package['submitted_at']}. "
            "This is the package currently waiting for approval."
        )
    elif active_package is not None and active_package["status"] in {"Changes Requested", "Rejected"}:
        package_status = f"{active_package['package_id']} · {active_package['status']}"
        package_owner = "Users"
        package_detail = active_package.get("review_note") or "Governance returned the package for revision."
    elif active_package is not None:
        package_status = f"{active_package['package_id']} · {active_package['status']}"
        package_owner = "Users"
        package_detail = (
            f"Approved by {active_package['approved_by']} at {active_package['approved_at']}. "
            "This is the latest approved package in the workflow."
        )

    if has_unsubmitted_metadata_changes(metadata):
        package_detail += " A revised draft exists and still needs to be resubmitted."

    release_owner = "Users"
    release_detail = effective_release_status(metadata, controls)
    if st.session_state.export_requested_by and st.session_state.export_policy_approved_by is None:
        release_owner = "Control"
    elif st.session_state.export_policy_approved_by and st.session_state.export_approved_by is None:
        release_owner = "Executive Approver"
    elif st.session_state.export_approved_by:
        release_owner = "Ready for approved recipients"

    return [
        {
            "title": "Metadata package in progress",
            "value": package_status,
            "detail": package_detail,
            "status": "Good" if active_package and active_package["status"] == "Approved" and not has_unsubmitted_metadata_changes(metadata) else "Warn",
        },
        {
            "title": "Current owner",
            "value": package_owner,
            "detail": "Who is expected to move the metadata package forward next.",
            "status": "Warn",
        },
        {
            "title": "Release gate",
            "value": release_detail,
            "detail": f"Next release owner: {release_owner}.",
            "status": "Good" if release_detail == "Approved" else "Warn",
        },
    ]


def current_workflow_stage(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> int:
    if not has_active_dataset() or not st.session_state.intake_confirmed:
        return 0
    if not st.session_state.hygiene_reviewed:
        return 1
    if not st.session_state.settings_reviewed or st.session_state.metadata_status in {"Changes Requested", "Rejected"}:
        return 2
    if st.session_state.metadata_status != "Approved" or has_unsubmitted_metadata_changes(metadata):
        return 3
    if st.session_state.synthetic_df is None or has_stale_generation(metadata, controls):
        return 4
    return 5


def build_progress_tracker_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    inferred_stage = current_workflow_stage(metadata, controls)
    # Use the user's actual current page as the highlight anchor so the nav
    # stays in sync with what's rendered below (critical for step 6 and step 7).
    active_step = int(st.session_state.get("current_step", inferred_stage))
    all_complete = bool(st.session_state.results_shared_at)
    rows: list[dict[str, str]] = []
    status_labels = step_status_labels(metadata, controls)
    for index, step in enumerate(STEP_CONFIG):
        if index == active_step:
            state = "current"
        elif index < max(inferred_stage, active_step):
            state = "complete"
        elif all_complete:
            state = "complete"
        elif index == max(inferred_stage, active_step) + 1:
            state = "next"
        else:
            state = "future"
        marker = "✓" if state == "complete" else str(index + 1)
        meta = (
            "Complete"
            if state == "complete"
            else "Current step"
            if state == "current"
            else "Next"
            if state == "next"
            else "Locked"
        )
        rows.append(
            {
                "title": step["title"],
                "owner": step["owner"],
                "marker": marker,
                "meta": status_labels[index] if state in {"complete", "current"} else meta,
                "class": state,
            }
        )
    return rows


def build_primary_action(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, Any]:
    action_step = default_step_for_role(metadata, controls)
    step_zero_label = (
        "Upload Dataset"
        if not has_active_dataset()
        else "Submit Request"
        if submission_ready() and not st.session_state.intake_confirmed
        else "Complete Request Details"
    )
    label_map = {
        0: step_zero_label,
        1: "Review Scan Results",
        2: "Review Data Settings",
        3: "Review Submitted Request" if st.session_state.current_role == "Manager / Reviewer" else "Submit for Review",
        4: "Generate Synthetic Data",
        5: "Review Final Output" if st.session_state.current_role == "Manager / Reviewer" else "Download & Share Results",
    }
    note_map = {
        0: "Add request details, upload a CSV, and submit when the request is ready.",
        1: "Confirm the scan findings and resolve data quality issues before moving on.",
        2: "Adjust field handling and data controls before the request is sent for review.",
        3: "Move the reviewed request into approval or respond to the reviewer decision.",
        4: "Run the approved request to create the synthetic dataset.",
        5: "Review the final output, validation summary, and release-ready files.",
    }
    return {
        "step": action_step,
        "label": label_map[action_step],
        "headline": STEP_CONFIG[action_step]["title"],
        "note": note_map[action_step],
    }


def render_action_center(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    status_lists = build_role_status_lists(metadata, controls)
    status_html = "".join(
        f"<div class='status-line'><div class='status-line-label'>{label}</div><div class='status-line-value'>{value}</div></div>"
        for label, value in build_current_request_status_rows(metadata, controls)
    )
    waiting_items = [item for item in status_lists["waiting"] if "No external blockers" not in item]
    blocker_html = f"<div class='action-subtext' style='margin: 0.85rem 0 0 0;'>Blocked by: {waiting_items[0]}</div>" if waiting_items else ""
    st.markdown(
        f"""
        <div class="action-shell">
            <h4>Request status</h4>
            <div class="status-lines">{status_html}</div>
            {blocker_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_role_guidance_panel(step_index: int, compare_only: bool = False) -> None:
    default_index = list(ROLE_CONFIGS.keys()).index(st.session_state.current_role)
    selected_role = st.selectbox(
        "Compare another role",
        options=list(ROLE_CONFIGS.keys()),
        index=default_index,
        key=f"role_guidance_{step_index}_{'compare' if compare_only else 'inline'}",
    )
    guidance = build_role_guidance(selected_role, step_index)
    left_col, right_col = st.columns([1.05, 0.95], gap="large")
    with left_col:
        st.markdown(
            f"""
            <div class="note-card">
                <div class="mini-label">{guidance['role']}</div>
                <strong>What this role focuses on</strong><br/>
                {guidance['priority']} {guidance['step_note']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            f"""
            <div class="note-card">
                <strong>Can do in this step</strong>
                <ul class="bullet-list">
                    {''.join(f"<li>{item}</li>" for item in guidance['allowed'])}
                </ul>
                <strong>Cannot do in this step</strong>
                <ul class="bullet-list">
                    {''.join(f"<li>{item}</li>" for item in guidance['blocked'])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_metadata_approval_rows() -> list[dict[str, str]]:
    submitted = st.session_state.metadata_submitted_by
    approved = st.session_state.metadata_approved_by
    submitted_at = st.session_state.metadata_submitted_at
    approved_at = st.session_state.metadata_approved_at
    reviewed_by = st.session_state.metadata_reviewed_by
    reviewed_at = st.session_state.metadata_reviewed_at
    status = st.session_state.metadata_status
    if status == "Changes Requested":
        final_status = "Changes requested"
        final_kind = "Warn"
        final_detail = f"Returned by: {reviewed_by or 'Governance'}{f' at {reviewed_at}' if reviewed_at else ''}"
    elif status == "Rejected":
        final_status = "Rejected"
        final_kind = "Bad"
        final_detail = f"Rejected by: {reviewed_by or 'Governance'}{f' at {reviewed_at}' if reviewed_at else ''}"
    else:
        final_status = "Approved" if approved else "Pending"
        final_kind = "Good" if approved else "Warn"
        final_detail = f"Approved by: {approved or 'Waiting for approval'}{f' at {approved_at}' if approved_at else ''}"
    return [
        {
            "level": "Level 1",
            "stage": "Package prepared",
            "owner": "Users",
            "status": "Complete" if st.session_state.metadata_status in {"Draft", "In Review", "Approved", "Changes Requested", "Rejected"} else "Pending",
            "kind": "Good" if st.session_state.metadata_status in {"Draft", "In Review", "Approved", "Changes Requested", "Rejected"} else "Warn",
            "detail": "Profiled source fields become editable metadata with owners, handling rules, and inclusion flags.",
        },
        {
            "level": "Level 2",
            "stage": "Review submitted",
            "owner": "Users",
            "status": "Submitted" if submitted else "Pending",
            "kind": "Good" if submitted else "Warn",
            "detail": f"Submitted by: {submitted or 'Waiting for submission'}{f' at {submitted_at}' if submitted_at else ''}",
        },
        {
            "level": "Level 3",
            "stage": "Metadata approved",
            "owner": "Governance",
            "status": final_status,
            "kind": final_kind,
            "detail": final_detail,
        },
    ]


def build_release_approval_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    validation_ready = st.session_state.validation is not None and not has_stale_generation(metadata, controls)
    requested_by = st.session_state.export_requested_by
    policy_by = st.session_state.export_policy_approved_by
    final_by = st.session_state.export_approved_by
    return [
        {
            "level": "Level 1",
            "stage": "Validation complete",
            "owner": "Users + Governance",
            "status": "Verified" if validation_ready else "Waiting",
            "kind": "Good" if validation_ready else "Warn",
            "detail": "A current validation run is required before the release chain can start.",
        },
        {
            "level": "Level 2",
            "stage": "Release requested",
            "owner": "Users",
            "status": "Requested" if requested_by else "Pending",
            "kind": "Good" if requested_by else "Warn",
            "detail": f"Requested by: {requested_by or 'Waiting for request'}",
        },
        {
            "level": "Level 3",
            "stage": "Control review",
            "owner": "IT / Security Admin",
            "status": "Approved" if policy_by else "Pending",
            "kind": "Good" if policy_by else "Warn",
            "detail": f"Approved by: {policy_by or 'Waiting for control review'}",
        },
        {
            "level": "Level 4",
            "stage": "Final export authorization",
            "owner": "Executive Approver",
            "status": "Approved" if final_by else "Locked",
            "kind": "Good" if final_by else "Warn",
            "detail": f"Authorized by: {final_by or 'Waiting for Executive Approver'}",
        },
    ]


def render_approval_hierarchy(rows: list[dict[str, str]], key_prefix: str) -> None:
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        chip_class = render_status_chip(row["kind"])
        col.markdown(
            f"""
            <div class="approval-card" id="{key_prefix}_{row['level']}">
                <div class="approval-level">{row['level']}</div>
                <h4>{row['stage']}</h4>
                <div class="approval-owner">{row['owner']}</div>
                <div class="{chip_class}">{row['status']}</div>
                <div class="approval-detail">{row['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _format_percent(series: pd.Series) -> pd.Series:
    return (series * 100).round(1)


def build_distribution_comparison(metadata: list[dict[str, Any]], column_name: str) -> dict[str, Any]:
    metadata_lookup = {item["column"]: item for item in metadata}
    column_meta = metadata_lookup[column_name]
    role = column_meta["data_type"]
    original = st.session_state.source_df[column_name]
    synthetic = st.session_state.synthetic_df[column_name]

    if role == "numeric":
        original_numeric = pd.to_numeric(original, errors="coerce").dropna()
        synthetic_numeric = pd.to_numeric(synthetic, errors="coerce").dropna()
        combined = pd.concat([original_numeric, synthetic_numeric], ignore_index=True)
        if combined.empty:
            return {"kind": "empty", "frame": pd.DataFrame(), "note": "No numeric values available for this column."}

        if float(combined.min()) == float(combined.max()):
            edges = np.array([float(combined.min()) - 0.5, float(combined.max()) + 0.5])
        else:
            edges = np.linspace(float(combined.min()), float(combined.max()), 13)

        source_hist, edges = np.histogram(original_numeric, bins=edges)
        synthetic_hist, _ = np.histogram(synthetic_numeric, bins=edges)
        labels = [f"{edges[index]:.1f} to {edges[index + 1]:.1f}" for index in range(len(edges) - 1)]
        frame = pd.DataFrame(
            {
                "Bucket": labels,
                "Source": _format_percent(pd.Series(source_hist) / max(source_hist.sum(), 1)),
                "Synthetic": _format_percent(pd.Series(synthetic_hist) / max(synthetic_hist.sum(), 1)),
            }
        ).set_index("Bucket")
        note = (
            f"Source median {original_numeric.median():.1f}, synthetic median {synthetic_numeric.median():.1f}. "
            f"Source mean {original_numeric.mean():.1f}, synthetic mean {synthetic_numeric.mean():.1f}."
        )
        return {"kind": "line", "frame": frame, "note": note}

    if role == "date":
        original_dates = pd.to_datetime(original, errors="coerce", format="mixed").dropna()
        synthetic_dates = pd.to_datetime(synthetic, errors="coerce", format="mixed").dropna()
        combined = pd.concat([original_dates, synthetic_dates], ignore_index=True)
        if combined.empty:
            return {"kind": "empty", "frame": pd.DataFrame(), "note": "No valid dates available for this column."}
        freq = "W" if (combined.max() - combined.min()).days > 45 else "D"
        source_counts = original_dates.dt.to_period(freq).value_counts(normalize=True).sort_index()
        synthetic_counts = synthetic_dates.dt.to_period(freq).value_counts(normalize=True).sort_index()
        periods = sorted(set(source_counts.index).union(set(synthetic_counts.index)))
        frame = pd.DataFrame(
            {
                "Period": [str(period) for period in periods],
                "Source": _format_percent(source_counts.reindex(periods, fill_value=0.0)),
                "Synthetic": _format_percent(synthetic_counts.reindex(periods, fill_value=0.0)),
            }
        ).set_index("Period")
        note = f"Timeline compared at {'weekly' if freq == 'W' else 'daily'} resolution."
        return {"kind": "line", "frame": frame, "note": note}

    source_distribution = original.fillna("Missing").astype(str).value_counts(normalize=True)
    synthetic_distribution = synthetic.fillna("Missing").astype(str).value_counts(normalize=True)
    ranked_categories = (
        _format_percent(source_distribution).add(_format_percent(synthetic_distribution), fill_value=0.0).sort_values(ascending=False).head(8).index
    )
    frame = pd.DataFrame(
        {
            "Category": ranked_categories,
            "Source": _format_percent(source_distribution.reindex(ranked_categories, fill_value=0.0)),
            "Synthetic": _format_percent(synthetic_distribution.reindex(ranked_categories, fill_value=0.0)),
        }
    ).set_index("Category")
    note = "Top categories shown by combined frequency across source and synthetic output."
    return {"kind": "bar", "frame": frame, "note": note}


def build_use_case_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    indicators = {item["label"]: item["value"] for item in build_validation_dashboard(metadata, controls)}
    overall = st.session_state.validation["overall_score"]
    privacy = st.session_state.validation["privacy_score"]
    utility = indicators.get("Sandbox suitability", 0.0)
    fidelity = st.session_state.validation["fidelity_score"]
    wait_fields_present = any("wait" in item["column"].lower() and item["include"] for item in metadata)

    def status(value: str) -> str:
        return value

    return [
        {
            "Use case": "Operational workflow modeling",
            "Fit": status("Ready" if overall >= 75 else "Review"),
            "Why": "Key operational fields remain aligned enough to model service-line workflows without moving raw encounter rows.",
            "Guardrail": "Use synthetic output only. Do not back-infer individual patient events.",
        },
        {
            "Use case": "Patient-flow scenario analysis",
            "Fit": status("Ready" if wait_fields_present and fidelity >= 70 else "Limited"),
            "Why": "Arrival, acuity, wait, and disposition behavior can support queue and throughput experiments under controlled assumptions.",
            "Guardrail": "Treat scenario outputs as planning aids, not as exact forecasts.",
        },
        {
            "Use case": "Analytics pipeline development",
            "Fit": status("Ready" if utility >= 75 else "Review"),
            "Why": "Analysts can test joins, cohort logic, feature engineering, and notebooks before requesting sensitive data access.",
            "Guardrail": "Re-run verification after metadata changes before sharing derived work.",
        },
        {
            "Use case": "Vendor sandbox or integration testing",
            "Fit": status("Ready" if privacy >= 85 else "Review"),
            "Why": "Synthetic records can exercise file layouts, API contracts, and workflows without releasing direct identifiers.",
            "Guardrail": "Share only the generated synthetic output and keep the release readiness report attached.",
        },
        {
            "Use case": "Training and workflow rehearsal",
            "Fit": status("Ready" if overall >= 65 else "Limited"),
            "Why": "Teams can rehearse handoffs, reporting, and review processes using realistic but de-identified examples.",
            "Guardrail": "Not for clinical decision support or patient-specific intervention.",
        },
    ]


def build_validation_dashboard(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, Any]]:
    validation = st.session_state.validation
    synthetic_df = st.session_state.synthetic_df
    if validation is None or synthetic_df is None:
        return []

    active_columns = [item for item in metadata if item["include"]]
    active_names = [item["column"] for item in active_columns]
    schema_match = round(
        len([column for column in active_names if column in synthetic_df.columns]) / max(len(active_names), 1) * 100,
        1,
    )
    statistical_fidelity = round(validation["fidelity_score"] * 0.75 + schema_match * 0.25, 1)
    unresolved_hygiene_pressure = max(
        0.0,
        100.0
        - st.session_state.hygiene["severity_counts"]["High"] * 12
        - st.session_state.hygiene["severity_counts"]["Medium"] * 6,
    )
    downstream_utility = round(
        min(
            100.0,
            validation["overall_score"] * 0.62
            + schema_match * 0.18
            + unresolved_hygiene_pressure * 0.20,
        ),
        1,
    )
    return [
        {
            "label": "Schema preservation",
            "value": schema_match,
            "detail": "Approved source fields retained in the synthetic output.",
        },
        {
            "label": "Distribution alignment",
            "value": validation["fidelity_score"],
            "detail": "Operational similarity between source and synthetic columns.",
        },
        {
            "label": "Privacy boundary",
            "value": validation["privacy_score"],
            "detail": "Overlap and identifier reuse checks under the current posture.",
        },
        {
            "label": "Statistical fidelity",
            "value": statistical_fidelity,
            "detail": "Blends schema coverage with column-level fidelity.",
        },
        {
            "label": "Sandbox suitability",
            "value": downstream_utility,
            "detail": "Indicator for workflow modeling and sandbox use.",
        },
    ]


def build_comparison_table(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        if not item["include"] or item["column"] not in synthetic_df.columns:
            continue
        column = item["column"]
        role = item["data_type"]
        if role == "identifier":
            continue
        if role == "numeric":
            original_numeric = pd.to_numeric(original_df[column], errors="coerce").dropna()
            synthetic_numeric = pd.to_numeric(synthetic_df[column], errors="coerce").dropna()
            if original_numeric.empty or synthetic_numeric.empty:
                continue
            rows.append(
                {
                    "Field": column,
                    "Source summary": f"mean {original_numeric.mean():.1f} | median {original_numeric.median():.1f}",
                    "Synthetic summary": f"mean {synthetic_numeric.mean():.1f} | median {synthetic_numeric.median():.1f}",
                }
            )
        else:
            source_mode = original_df[column].fillna("Missing").astype(str).value_counts(normalize=True).head(1)
            synthetic_mode = synthetic_df[column].fillna("Missing").astype(str).value_counts(normalize=True).head(1)
            if source_mode.empty or synthetic_mode.empty:
                continue
            rows.append(
                {
                    "Field": column,
                    "Source summary": f"{source_mode.index[0]} ({source_mode.iloc[0] * 100:.1f}%)",
                    "Synthetic summary": f"{synthetic_mode.index[0]} ({synthetic_mode.iloc[0] * 100:.1f}%)",
                }
            )
        if len(rows) >= 8:
            break
    return pd.DataFrame(rows)


def build_generation_control_rows(controls: dict[str, Any]) -> list[dict[str, str]]:
    locked_columns = controls.get("locked_columns", [])
    return [
        {
            "Control": "Generation preset",
            "Setting": str(controls.get("generation_preset", "Balanced")),
            "Effect": "Provides a quick starting point for privacy, structure, and noise settings.",
        },
        {
            "Control": "Synthetic row count",
            "Setting": str(controls.get("synthetic_rows", 0)),
            "Effect": "Sets how many synthetic records will be created in the current run.",
        },
        {
            "Control": "Privacy versus fidelity",
            "Setting": f"{controls.get('fidelity_priority', 0)} / 100",
            "Effect": "Balances closeness to source behavior against privacy smoothing.",
        },
        {
            "Control": "Lock key distributions",
            "Setting": ", ".join(locked_columns[:4]) + (" ..." if len(locked_columns) > 4 else "") if locked_columns else "None selected",
            "Effect": "Keeps selected columns closer to source frequency or value patterns.",
        },
        {
            "Control": "Correlation preservation",
            "Setting": f"{controls.get('correlation_preservation', 0)} / 100",
            "Effect": "Retains more row-to-row structure across fields when increased.",
        },
        {
            "Control": "Rare case retention",
            "Setting": f"{controls.get('rare_case_retention', 0)} / 100",
            "Effect": "Gives additional weight to rare categories and atypical rows.",
        },
        {
            "Control": "Noise level",
            "Setting": f"{controls.get('noise_level', 0)} / 100",
            "Effect": "Controls how much random variation is added during generation.",
        },
        {
            "Control": "Missingness pattern",
            "Setting": str(controls.get("missingness_pattern", "Preserve source pattern")),
            "Effect": "Determines whether gaps are preserved, reduced, or filled.",
        },
        {
            "Control": "Outlier strategy",
            "Setting": str(controls.get("outlier_strategy", "Preserve tails")),
            "Effect": "Defines whether extreme values are preserved, clipped, or smoothed.",
        },
    ]


def build_validation_report(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> str:
    validation = st.session_state.validation
    if validation is None:
        return "Verification is not available yet."
    lines = [
        "Synthetic Package Release Readiness Report",
        "",
        f"Dataset: {st.session_state.source_label}",
        f"Metadata status: {st.session_state.metadata_status}",
        f"Release recommendation: {effective_release_status(metadata, controls)}",
        "",
        "Supporting verification indicators (secondary):",
        f"- Fidelity indicator: {validation['fidelity_score']}",
        f"- Privacy indicator: {validation['privacy_score']}",
        f"- Overall indicator: {validation['overall_score']}",
        "",
        "Verification breakdown:",
    ]
    for metric in build_validation_dashboard(metadata, controls):
        lines.append(f"- {metric['label']}: {metric['value']} ({metric['detail']})")
    lines.extend(["", "Active generation controls:"])
    for row in build_generation_control_rows(controls):
        lines.append(f"- {row['Control']}: {row['Setting']} ({row['Effect']})")
    lines.extend(["", "Governance and privacy checks:"])
    for _, row in validation["privacy_checks"].iterrows():
        lines.append(f"- {row['check']}: {row['result']} ({row['interpretation']})")
    lines.extend(
        [
            "",
            "Audit status:",
            f"- Metadata approved by: {st.session_state.metadata_approved_by or 'Not approved'}",
            f"- Submitted by: {st.session_state.metadata_submitted_by or 'Not submitted'}",
            f"- Shared by: {st.session_state.results_shared_by or 'Not shared'}",
            f"- Shared at: {st.session_state.results_shared_at or 'Not shared'}",
        ]
    )
    lines.extend(["", "Recommended use cases:"])
    for row in build_use_case_rows(metadata, controls):
        lines.append(f"- {row['Use case']}: {row['Fit']} ({row['Guardrail']})")
    return "\n".join(lines)


def sync_metadata_workflow_state(metadata: list[dict[str, Any]]) -> None:
    st.session_state.metadata_has_unsubmitted_changes = has_unsubmitted_metadata_changes(metadata)
    reviewed_signature = st.session_state.get("settings_review_signature")
    if reviewed_signature and build_metadata_signature(metadata) != reviewed_signature:
        st.session_state.settings_reviewed = False


def has_stale_generation(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> bool:
    if st.session_state.synthetic_df is None:
        return False
    return st.session_state.last_generation_signature != build_generation_signature(metadata, controls)


def intake_visible_to_raw_rows() -> str:
    return "Full preview" if has_permission("view_raw") else "Summary only"


def build_operating_state_cards(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    sensitive_fields = build_phi_detection_frame(st.session_state.profile, metadata) if has_active_dataset() else pd.DataFrame()
    review_fields = sum(1 for item in metadata if metadata_owner(item) in {"Data Steward", "Privacy Officer"} and item["include"])
    return [
        {
            "title": "Signed-in role",
            "value": st.session_state.current_role,
            "detail": current_role_group(),
            "status": "Good",
        },
        {
            "title": "Raw data visibility",
            "value": intake_visible_to_raw_rows(),
            "detail": "Row-level source access",
            "status": "Good" if has_permission("view_raw") else "Warn",
        },
        {
            "title": "PHI / sensitive fields",
            "value": str(len(sensitive_fields)),
            "detail": "Fields under active controls",
            "status": "Warn" if len(sensitive_fields) else "Good",
        },
        {
            "title": "Metadata status",
            "value": metadata_display_status(metadata),
            "detail": f"{review_fields} governance-owned fields",
            "status": "Good" if st.session_state.metadata_status == "Approved" and not has_unsubmitted_metadata_changes(metadata) else "Warn",
        },
        {
            "title": "Release gate",
            "value": effective_release_status(metadata, controls),
            "detail": "Current export state",
            "status": "Good" if effective_release_status(metadata, controls) == "Approved" else "Warn",
        },
    ]


def step_status_labels(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[str]:
    if not has_active_dataset():
        return ["Upload needed", "Locked", "Locked", "Locked", "Locked", "Locked", "Locked"]
    return [
        "Uploaded" if st.session_state.intake_confirmed else "Action needed",
        "Scanned" if st.session_state.hygiene_reviewed else "Ready",
        "Reviewed" if st.session_state.settings_reviewed else "Action needed",
        (
            "Pending approval"
            if st.session_state.metadata_status == "In Review"
            else "Approved"
            if st.session_state.metadata_status == "Approved"
            else "Changes requested"
            if st.session_state.metadata_status in {"Changes Requested", "Rejected"}
            else "Ready to submit"
        ),
        "Generated" if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls) else "Waiting",
        "Shared" if st.session_state.results_shared_at else ("Ready" if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls) else "Waiting"),
        "Available" if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls) else "Locked",
    ]


def max_unlocked_step(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> int:
    if not st.session_state.intake_confirmed:
        return 0
    if not st.session_state.hygiene_reviewed:
        return 1
    if not st.session_state.settings_reviewed:
        return 2
    if st.session_state.metadata_status != "Approved" or has_unsubmitted_metadata_changes(metadata):
        return 3
    if st.session_state.synthetic_df is None or has_stale_generation(metadata, controls):
        return 4
    return 6


def default_step_for_role(metadata: list[dict[str, Any]], controls: dict[str, Any], role: str | None = None) -> int:
    active_role = role or st.session_state.get("current_role")
    visible_steps = visible_steps_for_role(active_role)

    if not visible_steps:
        return 0

    if active_role == "Manager / Reviewer":
        if st.session_state.metadata_status == "In Review":
            return 3
        if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls):
            return 5
        return visible_steps[0]

    if active_role == "Data Analyst":
        if not st.session_state.intake_confirmed:
            return 0
        if not st.session_state.hygiene_reviewed:
            return 1
        if not st.session_state.settings_reviewed or st.session_state.metadata_status in {"Changes Requested", "Rejected"}:
            return 2
        if st.session_state.metadata_status != "Approved" or has_unsubmitted_metadata_changes(metadata):
            return 3
        if st.session_state.synthetic_df is None or has_stale_generation(metadata, controls):
            return 4
        return 5

    unlocked = max_unlocked_step(metadata, controls)
    for step_index in visible_steps:
        if step_index >= unlocked:
            return step_index
    return visible_steps[-1]


def ensure_role_step_visibility(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    visible_steps = visible_steps_for_role()
    if st.session_state.current_step not in visible_steps:
        st.session_state.current_step = default_step_for_role(metadata, controls)


def quick_sign_in(role: str) -> None:
    email_stub = "analyst" if role == "Data Analyst" else "reviewer"
    st.session_state.authenticated = True
    st.session_state.current_role = role
    st.session_state.current_user_email = f"{email_stub}@southlake.ca"
    record_audit_event("Quick access sign in", f"{st.session_state.current_user_email} entered as {role}.", status="Logged")
    ensure_dataset_loaded()
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    controls = st.session_state.controls
    st.session_state.current_step = default_step_for_role(metadata, controls, role)
    st.rerun()


def render_status_chip(kind: str) -> str:
    if kind == "Good":
        return "status-good"
    if kind == "Bad":
        return "status-bad"
    return "status-warn"


def build_stakeholder_group_overview_html() -> str:
    role_cards = {
        "Data Analyst": "Upload data, review scan findings, adjust settings, submit requests, and download final results.",
        "Manager / Reviewer": "Approve or reject submitted requests and review the final synthetic output.",
    }
    cards = []
    for role_name, summary in role_cards.items():
        cards.append(
            f"""
            <div class="group-chip">
                <div class="group-chip-label">{role_name}</div>
                <div class="group-chip-value">{ROLE_TO_GROUP.get(role_name, '')}</div>
                <div class="group-chip-meta">{summary}</div>
            </div>
            """
        )
    return f"<div class='group-strip' style='grid-template-columns: repeat(2, minmax(0, 1fr));'>{''.join(cards)}</div>"


def render_stakeholder_group_overview() -> None:
    st.markdown(build_stakeholder_group_overview_html(), unsafe_allow_html=True)


def render_login_screen() -> None:
    logo_uri = load_logo_data_uri()

    # Page-level CSS
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem !important; }

            /* Left panel — dark brand */
            .login-left-panel {
                background: linear-gradient(140deg, #0B3A6B 0%, #08467D 40%, #0B5EA8 100%);
                border-radius: 20px;
                padding: 2.5rem 2.5rem;
                min-height: 620px;
                color: #FFFFFF;
                position: relative;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .login-left-panel::before {
                content: "";
                position: absolute;
                top: -150px;
                right: -150px;
                width: 400px;
                height: 400px;
                background: radial-gradient(circle, rgba(25,203,197,0.12) 0%, transparent 70%);
                border-radius: 50%;
                z-index: 0;
            }
            .login-left-inner { position: relative; z-index: 1; }
            .login-brand-logo {
                width: 280px;
                max-width: 80%;
                height: auto;
                margin-bottom: 1.8rem;
                background: rgba(255,255,255,0.98);
                padding: 0.7rem 1rem;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            .login-brand-kicker {
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: rgba(255,255,255,0.65);
                margin-bottom: 0.9rem;
            }
            .login-brand-headline {
                font-size: 2.2rem;
                font-weight: 600;
                line-height: 1.15;
                letter-spacing: -0.015em;
                margin: 0 0 1.1rem 0;
                color: #FFFFFF;
            }
            .login-brand-sub {
                font-size: 0.98rem;
                line-height: 1.6;
                color: rgba(255,255,255,0.78);
                margin: 0 0 2.2rem 0;
            }
            .login-feature {
                display: flex;
                gap: 0.85rem;
                align-items: flex-start;
                margin-bottom: 0.9rem;
            }
            .login-feature-icon {
                flex: 0 0 32px;
                height: 32px;
                border-radius: 8px;
                background: #19CBC5;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #FFFFFF;
                font-weight: 900;
                font-size: 1rem;
                box-shadow: 0 2px 6px rgba(25,203,197,0.35);
            }
            .login-feature-title {
                font-size: 0.92rem;
                font-weight: 600;
                color: #FFFFFF;
            }
            .login-feature-desc {
                font-size: 0.82rem;
                color: rgba(255,255,255,0.72);
                margin-top: 0.1rem;
                line-height: 1.5;
            }

            /* Right panel — form */
            .login-right-panel {
                padding: 2.5rem 1rem 2.5rem 1.5rem;
            }
            .login-form-title {
                font-size: 1.75rem;
                font-weight: 600;
                color: #0F172A;
                letter-spacing: -0.015em;
                margin: 0 0 0.35rem 0;
                line-height: 1.2;
            }
            .login-form-sub {
                font-size: 0.92rem;
                color: #64748B;
                line-height: 1.5;
                margin-bottom: 1.8rem;
            }

            /* Input styling — matches demo button shape for visual consistency */
            .login-right-panel [data-baseweb="input"] {
                border-radius: 12px !important;
                border: 1.5px solid #CBD5E1 !important;
                min-height: 52px !important;
                box-sizing: border-box !important;
                background: #FFFFFF !important;
            }
            .login-right-panel [data-baseweb="input"]:hover {
                border-color: #94A3B8 !important;
            }
            .login-right-panel [data-baseweb="input"]:focus-within {
                border-color: #0B5EA8 !important;
                box-shadow: 0 0 0 3px rgba(11,94,168,0.12) !important;
            }
            .login-right-panel [data-baseweb="input"] input,
            .login-right-panel [data-baseweb="base-input"] input {
                padding: 0.85rem 1rem !important;
                font-size: 0.95rem !important;
                line-height: 1.5 !important;
                color: #0F172A !important;
                background: transparent !important;
            }
            .login-right-panel .stTextInput label p,
            .login-right-panel .stTextInput label {
                font-size: 0.87rem !important;
                font-weight: 500 !important;
                color: #334155 !important;
                margin-bottom: 0.3rem !important;
            }
            .login-right-panel .stTextInput {
                margin-bottom: 0.85rem !important;
            }
            .login-divider {
                display: flex;
                align-items: center;
                gap: 0.8rem;
                margin: 1.5rem 0 1rem 0;
            }
            .login-divider-line { flex: 1; height: 1px; background: #E2E8F0; }
            .login-divider-text { font-size: 0.76rem; color: #94A3B8; font-weight: 500; }
            .login-footer {
                margin-top: 1.6rem;
                padding-top: 1.1rem;
                border-top: 1px solid #F1F5F9;
                font-size: 0.8rem;
                color: #94A3B8;
                line-height: 1.55;
                text-align: left;
            }
            .login-footer a {
                color: #0B5EA8;
                text-decoration: none;
                font-weight: 500;
                margin-right: 1.2rem;
            }
            .login-footer code {
                background: #F1F5F9;
                padding: 0.1rem 0.4rem;
                border-radius: 4px;
                font-size: 0.76rem;
                color: #475569;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1], gap="medium")

    # Global placeholder styling for login form inputs — targets stTextInput directly
    # because Streamlit widgets are NOT children of custom markdown wrappers in the DOM.
    st.markdown(
        """
        <style>
            .stTextInput input {
                color: #0F172A !important;
            }
            .stTextInput input::placeholder {
                color: #94A3B8 !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #94A3B8 !important;
            }
            .stTextInput input::-webkit-input-placeholder {
                color: #94A3B8 !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #94A3B8 !important;
            }
            .stTextInput input::-moz-placeholder {
                color: #94A3B8 !important;
                opacity: 1 !important;
            }
            .stTextInput input:-ms-input-placeholder {
                color: #94A3B8 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── LEFT: brand panel (single self-contained HTML block) ──
    logo_html = (
        f'<img src="{logo_uri}" alt="Southlake Health" class="login-brand-logo" />'
        if logo_uri else ''
    )
    with left_col:
        st.markdown(
            f"""
            <div class="login-left-panel">
                <div class="login-left-inner">
                    {logo_html}
                    <div class="login-brand-kicker">Agentic Synthetic Data Workspace</div>
                    <h1 class="login-brand-headline">Governed synthetic data for hospital teams.</h1>
                    <p class="login-brand-sub">
                        Create synthetic healthcare datasets with metadata transparency,
                        reviewer sign-off, and controlled release &mdash; without moving source data.
                    </p>
                </div>
                <div class="login-left-inner">
                    <div class="login-feature">
                        <div class="login-feature-icon">&#10003;</div>
                        <div>
                            <div class="login-feature-title">Role-based access</div>
                            <div class="login-feature-desc">Data Analyst uploads. Manager signs off. Auditable throughout.</div>
                        </div>
                    </div>
                    <div class="login-feature">
                        <div class="login-feature-icon">&#10003;</div>
                        <div>
                            <div class="login-feature-title">Metadata-only transformation</div>
                            <div class="login-feature-desc">Source records never leave the governed boundary.</div>
                        </div>
                    </div>
                    <div class="login-feature">
                        <div class="login-feature-icon">&#10003;</div>
                        <div>
                            <div class="login-feature-title">Controlled release</div>
                            <div class="login-feature-desc">Verified synthetic packages released for internal modeling only.</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── RIGHT: form panel — use pure Streamlit widgets inside a styled wrapper ──
    with right_col:
        # We use a class on the column area via a marker div — all widgets below inherit the CSS
        st.markdown('<div class="login-right-panel">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="login-form-title">Sign in</div>
            <div class="login-form-sub">Use your hospital workspace credentials to continue.</div>
            """,
            unsafe_allow_html=True,
        )

        work_email = st.text_input(
            "Work Email",
            placeholder="name@southlake.ca",
            key="login_email_input",
        )
        password = st.text_input(
            "Password",
            type="password",
            key="login_password_input",
        )

        submitted = st.button(
            "Sign In",
            type="primary",
            use_container_width=True,
            key="login_submit_btn",
        )

        if submitted:
            email_value = work_email.strip().lower()
            email_valid = "@" in email_value and "." in email_value.split("@")[-1] if "@" in email_value else False
            role = "Data Analyst"
            if not email_value:
                st.error("Enter your work email to continue.")
            elif not email_valid:
                st.error("Enter a valid work email address.")
            elif password != ROLE_CONFIGS[role]["password"]:
                st.error("Password did not match your access profile.")
            else:
                st.session_state.authenticated = True
                st.session_state.current_role = role
                st.session_state.current_user_email = email_value
                record_audit_event("User signed in", f"{email_value} signed in as {role}.", status="Logged")
                ensure_dataset_loaded()
                metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
                controls = st.session_state.controls
                st.session_state.current_step = default_step_for_role(metadata, controls, role)
                st.rerun()

        st.markdown(
            """
            <div style="text-align:center;margin:0.9rem 0 0.3rem 0;">
                <a href="mailto:itsupport@southlake.ca?subject=Password%20Reset%20Request"
                    style="color:#0B5EA8;text-decoration:none;font-weight:500;font-size:0.85rem;margin:0 0.8rem;">Forgot password?</a>
                <a href="mailto:accessrequests@southlake.ca?subject=Workspace%20Access"
                    style="color:#0B5EA8;text-decoration:none;font-weight:500;font-size:0.85rem;margin:0 0.8rem;">Request access</a>
            </div>
            <div class="login-divider">
                <div class="login-divider-line"></div>
                <div class="login-divider-text">or continue as demo</div>
                <div class="login-divider-line"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Data Analyst", use_container_width=True, key="login_demo_analyst"):
            quick_sign_in("Data Analyst")
        if st.button("Manager / Reviewer", use_container_width=True, key="login_demo_manager"):
            quick_sign_in("Manager / Reviewer")

        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full-width bottom footer — demo credential + security notice only
    st.markdown(
        """
        <div style="margin-top:2rem;padding:1.5rem 1rem 1rem 1rem;border-top:1px solid #E2E8F0;text-align:center;">
            <div style="font-size:0.82rem;color:#64748B;margin-bottom:0.4rem;">
                Demo credential for all access profiles:
                <code style="background:#F1F5F9;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.78rem;color:#475569;margin-left:0.3rem;">test</code>
            </div>
            <div style="font-size:0.76rem;color:#94A3B8;">
                Authorized users only. Activity is logged for compliance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    st.sidebar.markdown("## Session")
    st.sidebar.markdown(f"**Role**  \n{st.session_state.current_role}")
    st.sidebar.caption(current_role_summary())
    st.sidebar.metric("Current step", f"{st.session_state.current_step + 1}/{len(STEP_CONFIG)}")
    st.sidebar.metric("Release gate", effective_release_status(metadata, controls))
    st.sidebar.metric("Audit events", len(st.session_state.audit_events))

    st.sidebar.markdown("## Access in this role")
    for permission in sorted(ROLE_CONFIGS[st.session_state.current_role]["permissions"]):
        st.sidebar.markdown(f"- {permission.replace('_', ' ').title()}")

    if st.sidebar.button("Switch role", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_role = None
        st.session_state.current_user_email = None
        st.session_state.current_step = 0
        st.rerun()


def render_header(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    logo_uri = load_logo_data_uri()
    dataset_status, dataset_meta = dataset_status_summary()
    workflow_step = build_primary_action(metadata, controls)["step"]
    request_status = request_status_from_snapshot(capture_workflow_snapshot())
    st.markdown(
        f"""
            <div class="topbar-shell">
                <div class="brand-lockup">
                    {'<img src="' + logo_uri + '" class="brand-logo" alt="Southlake Health logo" />' if logo_uri else ''}
                    <div class="brand-copy">
                        <div class="brand-title">Southlake Health — Agentic Synthetic Data Workspace</div>
                        <div class="brand-subtitle">A governed, metadata-driven workflow for synthetic healthcare data creation, review, and release readiness</div>
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="summary-grid compact">
            <div class="summary-tile slim">
                <div class="summary-label">Current Role</div>
                <div class="summary-value">{st.session_state.current_role}</div>
                <div class="summary-meta">{ROLE_CONFIGS[st.session_state.current_role]['summary']}</div>
            </div>
            <div class="summary-tile slim">
                <div class="summary-label">Request Status</div>
                <div class="summary-value">{request_status}</div>
                <div class="summary-meta">{st.session_state.active_request_id or "No active request selected"}</div>
            </div>
            <div class="summary-tile slim">
                <div class="summary-label">Dataset Status</div>
                <div class="summary-value">{dataset_status}</div>
                <div class="summary-meta">{dataset_meta}</div>
            </div>
            <div class="summary-tile slim">
                <div class="summary-label">Current Step</div>
                <div class="summary-value">Step {workflow_step + 1}</div>
                <div class="summary-meta">{STEP_CONFIG[workflow_step]["title"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    queue_cols = st.columns([1.05, 0.45, 0.5], gap="large")
    with queue_cols[0]:
        request_ids = [record["request_id"] for record in st.session_state.get("request_registry", [])]
        active_id = st.session_state.get("active_request_id")
        if active_id is None and request_ids:
            active_id = request_ids[0]
        pending_selector = st.session_state.get("pending_active_request_selector")
        if pending_selector in request_ids:
            st.session_state.active_request_selector = pending_selector
            st.session_state.pending_active_request_selector = None
            active_id = pending_selector
        selector_value = st.session_state.get("active_request_selector")
        if selector_value not in request_ids and active_id in request_ids:
            st.session_state.active_request_selector = active_id
        selected_request = st.selectbox(
            "Request queue",
            options=request_ids,
            index=request_ids.index(active_id) if active_id in request_ids else 0,
            format_func=request_display_label,
            key="active_request_selector",
        )
        if selected_request != st.session_state.get("active_request_id"):
            sync_active_request_snapshot()
            restore_request_workspace(selected_request)
            persist_shared_workspace_state()
            st.rerun()
    with queue_cols[1]:
        # Invisible spacer label so button aligns vertically with selectbox
        st.markdown("<div style='font-size:0.875rem;margin-bottom:0.25rem;visibility:hidden;'>&nbsp;</div>", unsafe_allow_html=True)
        if has_permission("upload") and st.button("New request", use_container_width=True):
            sync_active_request_snapshot()
            create_blank_request()
            st.session_state.current_step = 0
            st.rerun()
    with queue_cols[2]:
        st.markdown("<div style='font-size:0.875rem;margin-bottom:0.25rem;visibility:hidden;'>&nbsp;</div>", unsafe_allow_html=True)
        clear_col, switch_col = st.columns(2, gap="small")
        if clear_col.button("Clear queue", use_container_width=True):
            schedule_request_queue_clear()
        if switch_col.button("Switch role", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_role = None
            st.session_state.current_user_email = None
            st.session_state.current_step = 0
            st.rerun()


def render_step_navigation(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    st.markdown("**Workflow**")
    progress_html = "".join(
        (
            f'<div class="workflow-progress-step {row["class"]}">'
            f'<div class="workflow-progress-marker">{row["marker"]}</div>'
            f'<div class="workflow-progress-owner">{row["owner"]}</div>'
            f'<div class="workflow-progress-title">{row["title"]}</div>'
            f'<div class="workflow-progress-state">{row["meta"]}</div>'
            "</div>"
        )
        for row in build_progress_tracker_rows(metadata, controls)
    )
    st.markdown(f"<div class='workflow-progress'>{progress_html}</div>", unsafe_allow_html=True)


def render_section_header(step_index: int, checkpoint: str) -> None:
    step = STEP_CONFIG[step_index]
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">Step {step_index + 1} · Owner {step['owner']}</div>
            <h3>{step['title']}</h3>
            <div class="state-text">{checkpoint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_previous_step_control(step_index: int) -> None:
    visible_steps = visible_steps_for_role()
    if step_index not in visible_steps:
        return
    position = visible_steps.index(step_index)
    if position == 0:
        return
    previous_step = visible_steps[position - 1]
    nav_cols = st.columns([0.34, 0.66], gap="large")
    if nav_cols[0].button(
        f"Back to {STEP_CONFIG[previous_step]['title']}",
        key=f"previous_step_{step_index}",
        use_container_width=True,
    ):
        st.session_state.current_step = previous_step
        st.rerun()


def render_role_restriction(message: str) -> None:
    st.info(message)


def render_step_one(metadata: list[dict[str, Any]]) -> None:
    render_section_header(0, "Register the source dataset into the governed workflow.")

    snapshot = capture_workflow_snapshot()
    status_value = request_status_from_snapshot(snapshot)
    has_data = has_active_dataset()
    active_request = st.session_state.active_request_id or "Not yet created"

    # Status chip color based on state
    if has_data:
        status_chip_color = "#136B48"; status_chip_bg = "#EDF9F3"
    else:
        status_chip_color = "#9C6A17"; status_chip_bg = "#FFF6E3"

    # ─────────────────────────────────────────────────────────────
    # A. STATUS STRIP — matches section-shell styling (white bg, --line border, --text color, --brand kickers)
    # ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1.2rem;padding:0.8rem 1.15rem;background:#ffffff;border:1px solid #d6e2ec;border-radius:20px;box-shadow:0 10px 24px rgba(8,70,125,0.08);margin-bottom:1rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Request</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;font-family:ui-monospace,monospace;">{active_request}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Role</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">{st.session_state.current_role}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Step</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">1 of {len(STEP_CONFIG)}</span>
            </div>
            <div style="margin-left:auto;display:inline-flex;align-items:center;gap:0.4rem;padding:0.28rem 0.7rem;background:{status_chip_bg};border-radius:999px;">
                <span style="width:6px;height:6px;border-radius:50%;background:{status_chip_color};"></span>
                <span style="font-size:0.8rem;color:{status_chip_color};font-weight:700;">{status_value}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────
    # B. REQUEST DETAILS
    # ─────────────────────────────────────────────────────────────
    with st.container(border=True, key="request_details_panel"):
        st.markdown(
            """
            <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Request details</div>
            <div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Project purpose</div>
            """,
            unsafe_allow_html=True,
        )
        if has_permission("upload"):
            st.text_input(
                "Project purpose",
                key="project_purpose",
                placeholder="Example: ED operational workflow modeling",
                label_visibility="collapsed",
            )
            st.caption("Describe how the synthetic data will be used. Required for submission.")
        else:
            render_role_restriction("This role can view the request summary but cannot edit request details.")

    # ─────────────────────────────────────────────────────────────
    # C. TWO-COLUMN TASK AREA — both columns in parallel bordered containers
    # ─────────────────────────────────────────────────────────────
    task_cols = st.columns([1.35, 1], gap="large")

    # ─── LEFT: Upload zone ───
    with task_cols[0]:
        with st.container(border=True, key="upload_task_panel"):
            if not has_data:
                st.markdown(
                    """
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Primary action</div>
                    <div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.35rem;line-height:1.3;">
                        Start by uploading your source dataset
                    </div>
                    <p style="font-size:0.9rem;color:#668097;line-height:1.55;margin:0 0 1rem 0;">
                        Drop a CSV file from Southlake\'s operational system, or click to browse.
                        The workspace will extract metadata and flag sensitive fields automatically.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Source dataset</div>
                    <div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Dataset loaded</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div style="background:#EDF9F3;border:1px solid #B8E3CC;border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:0.9rem;display:flex;align-items:center;gap:0.9rem;">
                        <div style="flex:0 0 34px;height:34px;border-radius:9px;background:#136B48;color:#FFFFFF;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem;">&#10003;</div>
                        <div style="flex:1;min-width:0;">
                            <div style="font-size:0.9rem;color:#17324d;font-weight:600;">{current_dataset_label()}</div>
                            <div style="font-size:0.78rem;color:#668097;margin-top:0.15rem;">
                                {format_file_size(st.session_state.get("source_file_size")) or "size unknown"}
                                &middot; {st.session_state.profile['summary']['rows']:,} rows
                                &middot; {st.session_state.profile['summary']['columns']} columns
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if has_permission("upload"):
                uploaded_file = st.file_uploader(
                    "Upload source dataset" if not has_data else "Replace source dataset",
                    type=["csv"],
                    help="CSV only. Headers required in first row.",
                    label_visibility="collapsed",
                )
                if uploaded_file is not None:
                    current_signature = (uploaded_file.name, uploaded_file.size)
                    if st.session_state.get("uploaded_signature") != current_signature:
                        st.session_state.source_file_size = uploaded_file.size
                        set_source_dataframe(load_csv_bytes(uploaded_file.getvalue()), f"Uploaded dataset * {uploaded_file.name}")
                        st.session_state.uploaded_signature = current_signature
                        persist_shared_workspace_state()
                        st.rerun()

                st.markdown(
                    """
                    <div style="display:flex;gap:1rem;margin-top:0.25rem;margin-bottom:0;font-size:0.78rem;color:#668097;flex-wrap:wrap;">
                        <span>&#128196; CSV format only</span>
                        <span>&#9881;&#65039; Headers in first row</span>
                        <span>&#128274; Source never leaves governance boundary</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                render_role_restriction("This role cannot upload or replace the source dataset.")

    # ─── RIGHT: Submission readiness ───
    with task_cols[1]:
        with st.container(border=True, key="submission_readiness_panel"):
            checklist = build_submission_checklist()
            total = len(checklist)
            completed_count = sum(1 for _, done in checklist if done)
            progress_pct = int((completed_count / total) * 100) if total else 0
            is_ready = submission_ready()

            if completed_count == total:
                progress_color = "#136B48"
            elif completed_count > 0:
                progress_color = "#0b5ea8"
            else:
                progress_color = "#94A3B8"

            checklist_items_html = ""
            for label, done in checklist:
                if done:
                    icon_html = '<span style="flex:0 0 18px;width:18px;height:18px;border-radius:50%;background:#136B48;color:#FFFFFF;display:inline-flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">&#10003;</span>'
                    label_style = "color:#17324d;font-weight:500;"
                else:
                    icon_html = '<span style="flex:0 0 18px;width:18px;height:18px;border-radius:50%;background:#FFFFFF;border:1.5px solid #d6e2ec;display:inline-block;"></span>'
                    label_style = "color:#17324d;font-weight:500;"
                checklist_items_html += (
                    f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.55rem;">'
                    f'{icon_html}'
                    f'<span style="font-size:0.9rem;{label_style}">{label}</span>'
                    f'</div>'
                )

            missing = submission_missing_items()
            if is_ready:
                helper_msg = "All requirements met. You can submit this request."
                helper_color = "#136B48"
            elif missing:
                helper_msg = f"Complete {missing[0].lower()} to continue."
                helper_color = "#668097"
            else:
                helper_msg = "Upload required to continue."
                helper_color = "#9C6A17"

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem;">
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;">Submission readiness</div>
                    <div style="font-size:0.82rem;font-weight:600;color:{progress_color};">{completed_count} of {total}</div>
                </div>
                <div style="width:100%;height:6px;background:#f1f5f9;border-radius:999px;margin-bottom:1rem;overflow:hidden;">
                    <div style="width:{progress_pct}%;height:100%;background:{progress_color};border-radius:999px;transition:width 0.3s ease;"></div>
                </div>
                {checklist_items_html}
                <div style="font-size:0.82rem;color:{helper_color};margin-top:0.3rem;margin-bottom:0.8rem;line-height:1.5;">{helper_msg}</div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.intake_confirmed:
                st.success("Request entered the governed flow.")
                if st.button("Continue to Scan Source Data", type="primary", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()
            else:
                submit_clicked = st.button(
                    "Submit request" if is_ready else "Submit request (blocked)",
                    type="primary",
                    use_container_width=True,
                    disabled=not is_ready,
                    help=None if is_ready else f"Blocked until: {', '.join(submission_missing_items())}",
                )
                if submit_clicked:
                    st.session_state.intake_confirmed = True
                    record_audit_event("Request submitted", "Source workspace request was acknowledged and entered into the governed flow.", status="Completed")
                    st.session_state.current_step = 1
                    st.rerun()

    # ─────────────────────────────────────────────────────────────
    # D. INTAKE SUMMARY (only after upload) — wrapped in matching container
    # ─────────────────────────────────────────────────────────────
    if has_data:
        sensitive_count = len(build_phi_detection_frame(st.session_state.profile, metadata))
        if sensitive_count == 0:
            sens_accent = "#136B48"; sens_bg = "#EDF9F3"
        else:
            sens_accent = "#9C6A17"; sens_bg = "#FFF6E3"

        def _stat_capsule(kicker: str, value: str, detail: str, accent: str = "#0b5ea8", bg: str = "#f5f9fc") -> str:
            return (
                f'<div style="flex:1;min-width:140px;padding:0.85rem 1rem;background:{bg};'
                f'border:1px solid #d6e2ec;border-radius:12px;margin-bottom:0.25rem;">'
                f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{accent};margin-bottom:0.25rem;">{kicker}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:#17324d;line-height:1.15;">{value}</div>'
                f'<div style="font-size:0.76rem;color:#668097;margin-top:0.2rem;line-height:1.35;">{detail}</div>'
                f'</div>'
            )

        with st.container(border=True, key="intake_summary_panel"):
            rows_val = f"{st.session_state.profile['summary']['rows']:,}"
            cols_val = str(st.session_state.profile['summary']['columns'])
            intake_html = (
                '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Dataset intake summary</div>'
                '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.9rem;line-height:1.3;">Source package overview</div>'
                '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;">'
                + _stat_capsule("Rows", rows_val, "Loaded from CSV")
                + _stat_capsule("Columns", cols_val, "Fields detected")
                + _stat_capsule(
                    "Sensitive fields",
                    str(sensitive_count),
                    "Flagged for review" if sensitive_count > 0 else "No PHI detected",
                    accent=sens_accent,
                    bg=sens_bg,
                )
                + _stat_capsule(
                    "File size",
                    format_file_size(st.session_state.get("source_file_size")) or "—",
                    "Source dataset",
                )
                + '</div>'
            )
            st.markdown(intake_html, unsafe_allow_html=True)


def render_step_two() -> None:
    render_section_header(1, "Review agent-classified hygiene findings and field sensitivity.")
    render_previous_step_control(1)

    if not has_active_dataset():
        st.info("Upload and submit a source dataset before running the scan.")
        return

    # ─────────────────────────────────────────────────────────────
    # A. STATUS STRIP
    # ─────────────────────────────────────────────────────────────
    snapshot = capture_workflow_snapshot()
    active_request = st.session_state.active_request_id or "Not yet created"
    is_reviewed = st.session_state.get("hygiene_reviewed", False)

    if is_reviewed:
        status_chip_color = "#136B48"; status_chip_bg = "#EDF9F3"; status_label = "Scan reviewed"
    else:
        status_chip_color = "#9C6A17"; status_chip_bg = "#FFF6E3"; status_label = "Scan in review"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1.2rem;padding:0.8rem 1.15rem;background:#ffffff;border:1px solid #d6e2ec;border-radius:20px;box-shadow:0 10px 24px rgba(8,70,125,0.08);margin-bottom:1rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Request</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;font-family:ui-monospace,monospace;">{active_request}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Role</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">{st.session_state.current_role}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Step</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">2 of {len(STEP_CONFIG)}</span>
            </div>
            <div style="margin-left:auto;display:inline-flex;align-items:center;gap:0.4rem;padding:0.28rem 0.7rem;background:{status_chip_bg};border-radius:999px;">
                <span style="width:6px;height:6px;border-radius:50%;background:{status_chip_color};"></span>
                <span style="font-size:0.8rem;color:{status_chip_color};font-weight:700;">{status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hygiene = st.session_state.hygiene
    missingness_frame = build_missingness_strategy_frame(st.session_state.profile)
    missingness_chart = missingness_frame[missingness_frame["Field"] != "No incomplete field"][["Field", "Missing %"]]
    default_options = build_hygiene_option_defaults(hygiene)

    # ─────────────────────────────────────────────────────────────
    # B. HYGIENE SUMMARY — capsule cards
    # ─────────────────────────────────────────────────────────────
    def _stat_capsule(kicker: str, value: str, detail: str, accent: str = "#0b5ea8", bg: str = "#f5f9fc") -> str:
        return (
            f'<div style="flex:1;min-width:140px;padding:0.85rem 1rem;background:{bg};'
            f'border:1px solid #d6e2ec;border-radius:12px;margin-bottom:0.25rem;">'
            f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{accent};margin-bottom:0.25rem;">{kicker}</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:#17324d;line-height:1.15;">{value}</div>'
            f'<div style="font-size:0.76rem;color:#668097;margin-top:0.2rem;line-height:1.35;">{detail}</div>'
            f'</div>'
        )

    quality_score = hygiene["quality_score"]
    high_sev = hygiene["severity_counts"]["High"]
    med_sev = hygiene["severity_counts"]["Medium"]
    dup_rows = st.session_state.profile["summary"]["duplicate_rows"]

    if high_sev > 0:
        high_accent = "#9d2b3c"; high_bg = "#fff1f3"
    else:
        high_accent = "#0b5ea8"; high_bg = "#f5f9fc"

    if med_sev > 0:
        med_accent = "#9C6A17"; med_bg = "#FFF6E3"
    else:
        med_accent = "#0b5ea8"; med_bg = "#f5f9fc"

    with st.container(border=True, key="hygiene_summary_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Hygiene assessment</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.9rem;line-height:1.3;">Source quality overview</div>'
            '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;">'
            + _stat_capsule("Quality score", f"{quality_score}", "Agent-assessed dataset quality")
            + _stat_capsule("High severity", str(high_sev), "Critical fixes required" if high_sev > 0 else "None flagged", accent=high_accent, bg=high_bg)
            + _stat_capsule("Medium severity", str(med_sev), "Cleanup recommended" if med_sev > 0 else "None flagged", accent=med_accent, bg=med_bg)
            + _stat_capsule("Duplicate rows", str(dup_rows), "Exact duplicates detected" if dup_rows > 0 else "No duplicates")
            + '</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────
    # C. FINDINGS → REMEDIATION (unified panel — cause-to-effect UI)
    # ─────────────────────────────────────────────────────────────
    # Map concern types to their remediation options
    # Each concern can trigger 1 or more action options
    CONCERN_TO_ACTIONS = {
        "Missingness": [
            ("standardize_blank_strings", "Standardize blank strings as missing", "Converts empty strings to null values"),
            ("fill_operational_gaps", "Fill common missing operational values", "Imputes missing values using median / mode"),
        ],
        "Duplicate rows": [
            ("remove_duplicates", "Remove exact duplicate rows", "Keeps only unique rows"),
        ],
        "Category normalization": [
            ("normalize_categories", "Normalize category labels", "Unifies case, whitespace, and common synonyms"),
        ],
        "Negative values": [
            ("fix_negative_values", "Convert invalid negative values to missing", "Replaces impossible negatives (e.g., age, counts) with null"),
        ],
        "Invalid dates": [
            ("repair_invalid_dates", "Convert invalid dates to missing", "Replaces unparseable date strings with null"),
        ],
        "Outliers": [
            ("cap_numeric_extremes", "Cap numeric extremes", "Clips values beyond the 1st/99th percentile"),
        ],
    }

    # Severity styling
    def _severity_badge(severity: str) -> str:
        if severity == "High":
            color = "#9d2b3c"; bg = "#fff1f3"
        elif severity == "Medium":
            color = "#9C6A17"; bg = "#FFF6E3"
        else:
            color = "#136B48"; bg = "#EDF9F3"
        return (
            f'<span style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.2rem 0.55rem;'
            f'background:{bg};border-radius:999px;font-size:0.72rem;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:0.05em;">'
            f'<span style="width:5px;height:5px;border-radius:50%;background:{color};"></span>'
            f'{severity} severity</span>'
        )

    # Group issues by concern
    concern_groups: dict[str, list[dict[str, Any]]] = {}
    for issue in hygiene.get("issues", []):
        concern_groups.setdefault(issue["concern"], []).append(issue)

    # Track which option keys have been rendered (to show remaining discretionary ones)
    rendered_action_keys: set[str] = set()
    selected_options: dict[str, bool] = {}

    with st.container(border=True, key="findings_remediation_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Findings &amp; remediation plan</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.4rem;line-height:1.3;">Each finding maps to a recommended cleanup action</div>'
            '<div style="font-size:0.88rem;color:#668097;margin-bottom:1rem;line-height:1.5;">Toggle actions below to build a remediation plan. Enabled fixes are applied when you submit.</div>',
            unsafe_allow_html=True,
        )

        if not concern_groups:
            st.markdown(
                '<div style="padding:1.1rem 1.15rem;background:#EDF9F3;border:1px solid #B8E3CC;border-radius:12px;margin-bottom:0.5rem;">'
                '<div style="display:flex;align-items:center;gap:0.7rem;">'
                '<span style="width:32px;height:32px;border-radius:9px;background:#136B48;color:#FFFFFF;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem;">&#10003;</span>'
                '<div>'
                '<div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#136B48;margin-bottom:0.1rem;">Clean</div>'
                '<div style="font-size:0.96rem;color:#17324d;font-weight:600;">No hygiene issues detected</div>'
                '<div style="font-size:0.82rem;color:#475569;margin-top:0.1rem;">The source dataset passed all agent hygiene checks.</div>'
                '</div></div></div>',
                unsafe_allow_html=True,
            )

        # Render each concern group as a Finding → Action card
        for concern, issues_list in concern_groups.items():
            # Determine highest severity in this concern group
            severities = [i.get("severity", "Low") for i in issues_list]
            if "High" in severities:
                top_sev = "High"
            elif "Medium" in severities:
                top_sev = "Medium"
            else:
                top_sev = "Low"

            # Collect affected fields and finding texts
            affected_fields = sorted({i["column"] for i in issues_list if i.get("column")})
            finding_texts = [i.get("finding", "") for i in issues_list]

            # Build the finding header HTML (severity badge + concern + affected fields + descriptions)
            if affected_fields:
                chip_style = "background:#F1F5F9;padding:0.1rem 0.4rem;border-radius:4px;font-size:0.78rem;color:#17324d;"
                field_chips = ", ".join(f'<code style="{chip_style}">{f}</code>' for f in affected_fields)
                fields_html = (
                    f'<div style="font-size:0.8rem;color:#668097;margin-top:0.35rem;">'
                    f'<span style="font-weight:600;color:#475569;">Affects:</span> '
                    f'{field_chips}'
                    f'</div>'
                )
            else:
                fields_html = ''
            findings_list_html = ''.join(
                f'<li style="font-size:0.85rem;color:#475569;line-height:1.5;margin-bottom:0.15rem;">{text}</li>'
                for text in finding_texts if text
            )

            st.markdown(
                f'<div style="padding:1rem 1.15rem 0.3rem 1.15rem;background:#F8FAFC;border:1px solid #E5EDF5;border-radius:14px;margin-bottom:0.8rem;">'
                f'<div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.35rem;flex-wrap:wrap;">'
                f'{_severity_badge(top_sev)}'
                f'<div style="font-size:1rem;font-weight:600;color:#17324d;">{concern}</div>'
                f'</div>'
                f'{fields_html}'
                f'<ul style="margin:0.5rem 0 0.5rem 1.1rem;padding:0;">{findings_list_html}</ul>'
                f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-top:0.8rem;margin-bottom:0.35rem;">Recommended action{"s" if len(CONCERN_TO_ACTIONS.get(concern, [])) > 1 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Render action checkboxes for this concern
            actions = CONCERN_TO_ACTIONS.get(concern, [])
            for action_key, action_label, action_helper in actions:
                # Put the checkbox inside a marker div so we can position it visually inside the card above
                val = st.checkbox(
                    action_label,
                    value=default_options.get(action_key, False),
                    disabled=not has_permission("remediate"),
                    key=f"hygiene_opt_{action_key}",
                    help=action_helper,
                )
                selected_options[action_key] = val
                rendered_action_keys.add(action_key)
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#668097;margin-top:-0.4rem;margin-bottom:0.8rem;margin-left:1.8rem;line-height:1.45;">{action_helper}</div>',
                    unsafe_allow_html=True,
                )

        # ─ Discretionary / additional options not tied to findings ─
        all_action_keys = {key for actions in CONCERN_TO_ACTIONS.values() for (key, _, _) in actions} | {"group_rare_categories"}
        extra_keys = all_action_keys - rendered_action_keys

        if extra_keys:
            st.markdown(
                '<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#668097;margin-top:1rem;margin-bottom:0.45rem;">Optional additional cleanup</div>'
                '<div style="font-size:0.82rem;color:#668097;margin-bottom:0.6rem;line-height:1.5;">Not triggered by scan findings. Enable only if needed for downstream modeling.</div>',
                unsafe_allow_html=True,
            )
            extra_labels = {
                "standardize_blank_strings": ("Standardize blank strings as missing", "Converts empty strings to null values"),
                "fill_operational_gaps": ("Fill common missing operational values", "Imputes missing values using median / mode"),
                "remove_duplicates": ("Remove exact duplicate rows", "Keeps only unique rows"),
                "normalize_categories": ("Normalize category labels", "Unifies case, whitespace, synonyms"),
                "fix_negative_values": ("Convert invalid negative values to missing", "Replaces impossible negatives with null"),
                "repair_invalid_dates": ("Convert invalid dates to missing", "Replaces unparseable dates with null"),
                "cap_numeric_extremes": ("Cap numeric extremes", "Clips values beyond the 1st/99th percentile"),
                "group_rare_categories": ("Group rare category labels into Other", "Consolidates low-frequency categories"),
            }
            for key in extra_keys:
                lbl, helper = extra_labels.get(key, (key, ""))
                val = st.checkbox(
                    lbl,
                    value=default_options.get(key, False),
                    disabled=not has_permission("remediate"),
                    key=f"hygiene_opt_{key}",
                    help=helper,
                )
                selected_options[key] = val
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#668097;margin-top:-0.4rem;margin-bottom:0.6rem;margin-left:1.8rem;line-height:1.45;">{helper}</div>',
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────────────────────────
    # D. REMEDIATION PREVIEW + APPLY (primary card)
    # ─────────────────────────────────────────────────────────────
    # Ensure all option keys are in selected_options
    for key in [k for ks in CONCERN_TO_ACTIONS.values() for (k, _, _) in ks] + ["group_rare_categories"]:
        selected_options.setdefault(key, default_options.get(key, False))

    options = selected_options
    preview_df, preview_actions = apply_hygiene_fixes(st.session_state.source_df, options)
    preview_summary = summarize_dataframe_change(st.session_state.source_df, preview_df)

    any_action_selected = any(options.values())

    with st.container(border=True, key="remediation_preview_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Remediation preview</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.7rem;line-height:1.3;">Impact of your selected fixes</div>',
            unsafe_allow_html=True,
        )

        if not any_action_selected:
            st.markdown(
                '<div style="font-size:0.88rem;color:#668097;padding:0.7rem 0;line-height:1.55;">No fixes selected yet. Toggle a recommended action above to see the impact preview.</div>',
                unsafe_allow_html=True,
            )
        else:
            cells_changed = preview_summary["changed_cells"]
            cols_changed = preview_summary["changed_columns"]
            rows_before = preview_summary["rows_before"]
            rows_after = preview_summary["rows_after"]
            missing_delta = f"{preview_summary['missing_before']} \u2192 {preview_summary['missing_after']}"

            st.markdown(
                '<div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.8rem;">'
                + _stat_capsule("Rows", f"{rows_before} \u2192 {rows_after}", "Before vs. after")
                + _stat_capsule("Cells changed", str(cells_changed), f"{cols_changed} column(s)")
                + _stat_capsule("Missing cells", missing_delta, "Before vs. after")
                + '</div>',
                unsafe_allow_html=True,
            )
            if preview_actions:
                st.dataframe(pd.DataFrame(preview_actions), use_container_width=True, hide_index=True)

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        apply_cols = st.columns([1, 1], gap="small")
        if has_permission("remediate"):
            if apply_cols[0].button(
                "Apply selected fixes",
                type="primary",
                use_container_width=True,
                disabled=not any_action_selected,
            ):
                cleaned_df, actions = apply_hygiene_fixes(st.session_state.source_df, options)
                set_source_dataframe(cleaned_df, f"{st.session_state.source_label} * remediated")
                st.session_state.last_cleaning_actions = actions
                st.session_state.intake_confirmed = True
                st.session_state.hygiene_reviewed = True
                record_audit_event("Hygiene fixes applied", "; ".join(item["effect"] for item in actions), status="Completed")
                st.session_state.current_step = 2
                st.rerun()
        else:
            apply_cols[0].empty()
            render_role_restriction("This role cannot modify the source dataset. Review the scan results and wait for Data Analyst action.")

        if apply_cols[1].button(
            "Mark scan review complete",
            use_container_width=True,
            disabled=st.session_state.hygiene_reviewed,
        ):
            st.session_state.hygiene_reviewed = True
            record_audit_event("Scan review completed", "Source quality and risk findings were acknowledged.", status="Completed")
            st.session_state.current_step = 2
            st.rerun()

    # ─────────────────────────────────────────────────────────────
    # E. SUPPORTING CONTEXT (tabs — optional deeper view)
    # ─────────────────────────────────────────────────────────────
    with st.container(border=True, key="scan_context_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Supporting context</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.7rem;line-height:1.3;">Deeper views for investigation</div>',
            unsafe_allow_html=True,
        )
        scan_tabs = st.tabs(["Source preview", "Missingness strategy", "Field profile"])
        with scan_tabs[0]:
            preview_cols = st.columns(2)
            preview_cols[0].metric("Rows in source data", len(st.session_state.source_df))
            preview_cols[1].metric("Columns in source data", len(st.session_state.source_df.columns))
            st.dataframe(st.session_state.source_df.head(15), use_container_width=True, hide_index=True)
        with scan_tabs[1]:
            if not missingness_chart.empty:
                chart_frame = missingness_chart.set_index("Field")
                st.bar_chart(chart_frame, use_container_width=True)
            st.dataframe(missingness_frame, use_container_width=True, hide_index=True)
        with scan_tabs[2]:
            profile_rows = []
            for column, details in st.session_state.profile["columns"].items():
                profile_rows.append(
                    {
                        "Field": column,
                        "Role": details["semantic_role"],
                        "Missing %": details["missing_pct"],
                        "Unique values": details["unique_count"],
                        "Example values": ", ".join(details.get("examples", [])),
                    }
                )
            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────
    # F. LAST REMEDIATION APPLIED (only shown post-apply)
    # ─────────────────────────────────────────────────────────────
    if st.session_state.last_cleaning_actions:
        with st.container(border=True, key="last_remediation_panel"):
            st.markdown(
                '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#136B48;margin-bottom:0.3rem;">Remediation applied</div>'
                '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.65rem;line-height:1.3;">Latest cleanup actions</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(st.session_state.last_cleaning_actions), use_container_width=True, hide_index=True)


def render_step_three() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    render_section_header(2, "Confirm approved metadata assumptions for each field.")
    render_previous_step_control(2)

    if not has_active_dataset():
        st.info("Upload a dataset to review metadata assumptions.")
        return [], st.session_state.controls.copy()

    controls = st.session_state.controls.copy()
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    sync_metadata_workflow_state(metadata)
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    metadata_frame = build_metadata_review_frame(metadata)
    active_package = active_metadata_package_record(metadata)
    metadata_edit_locked = (
        has_permission("edit_metadata")
        and st.session_state.metadata_status in {"In Review", "Approved"}
        and not has_unsubmitted_metadata_changes(metadata)
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Settings status", "Reviewed" if st.session_state.settings_reviewed else "Draft")
    summary_cols[1].metric("Included fields", int(st.session_state.metadata_editor_df["include"].sum()))
    summary_cols[2].metric("Sensitive fields", int(metadata_frame["Sensitivity"].isin(["Restricted", "Sensitive"]).sum()))
    summary_cols[3].metric("Targeted actions", int(st.session_state.metadata_editor_df["control_action"].ne("Preserve").sum()))

    if st.session_state.metadata_status == "In Review" and active_package is not None:
        st.info(
            f"Package {active_package['package_id']} is currently with the Manager / Reviewer. Start a revised draft only if you need to change the submitted version."
        )
    elif st.session_state.metadata_status == "Approved" and active_package is not None:
        st.success(
            f"Package {active_package['package_id']} was approved by {active_package['approved_by']} at {active_package['approved_at']}."
        )
    elif st.session_state.metadata_status == "Changes Requested" and active_package is not None:
        st.warning(active_package.get("review_note") or "The Manager / Reviewer requested changes to this package.")
    elif st.session_state.metadata_status == "Rejected" and active_package is not None:
        st.error(active_package.get("review_note") or "The Manager / Reviewer rejected this request package.")

    if metadata_edit_locked and active_package is not None:
        lock_cols = st.columns([1.55, 0.45], gap="large")
        lock_cols[0].info(
            f"Package {active_package['package_id']} is frozen so both roles see the same submitted version."
        )
        if st.session_state.metadata_status == "Approved":
            if lock_cols[1].button("Revise draft", use_container_width=True):
                st.session_state.metadata_status = "Draft"
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None
                record_audit_event(
                    "Settings draft reopened",
                    f"A revised working draft was opened from package {active_package['package_id']}.",
                    status="Updated",
                )
                rerun_with_persist()

    # Agent: Metadata review artifact (unified primary artifact for this page)
    classified_for_artifact = classify_hygiene_issues(st.session_state.hygiene) if has_active_dataset() else []
    meta_artifact = build_metadata_review_artifact(
        metadata,
        st.session_state.profile,
        classified_for_artifact,
        sensitivity_fn=metadata_sensitivity,
        generation_rule_fn=metadata_handling,
    )
    render_metadata_review_artifact(meta_artifact)

    quick_tab, edit_tab, record_tab = st.tabs(
        ["Targeted field actions", "Advanced editor", "Submitted version"]
    )
    with quick_tab:
        quick_summary_cols = st.columns(4)
        quick_summary_cols[0].metric("Restricted fields", int(metadata_frame["Sensitivity"].eq("Restricted").sum()))
        quick_summary_cols[1].metric("Sensitive fields", int(metadata_frame["Sensitivity"].eq("Sensitive").sum()))
        quick_summary_cols[2].metric("Excluded fields", int(st.session_state.metadata_editor_df["include"].eq(False).sum()))
        quick_summary_cols[3].metric("Targeted actions", int(st.session_state.metadata_editor_df["control_action"].ne("Preserve").sum()))

        if has_permission("edit_metadata") and not metadata_edit_locked:
            bulk_cols = st.columns(3)
            if bulk_cols[0].button("Tighten PHI controls", use_container_width=True):
                apply_bulk_metadata_profile("tighten_phi")
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None
                rerun_with_persist()
            if bulk_cols[1].button("Preserve analytics detail", use_container_width=True):
                apply_bulk_metadata_profile("preserve_analytics")
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None
                rerun_with_persist()
            if bulk_cols[2].button("Reset metadata defaults", use_container_width=True):
                apply_bulk_metadata_profile("reset_defaults")
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None
                rerun_with_persist()

            quick_editor_source = st.session_state.metadata_editor_df.copy()
            quick_editor_source["Sensitivity"] = [metadata_sensitivity(item) for item in editor_frame_to_metadata(quick_editor_source)]
            quick_editor_source["Current handling"] = [metadata_handling(item) for item in editor_frame_to_metadata(quick_editor_source)]
            quick_editor_source = quick_editor_source[
                ["column", "Sensitivity", "data_type", "control_action", "include", "Current handling", "notes"]
            ]
            quick_editor_source = quick_editor_source.sort_values(
                by=["Sensitivity", "column"],
                key=lambda series: series.map({"Restricted": 0, "Sensitive": 1, "Operational": 2}).fillna(series),
            )
            quick_editor = st.data_editor(
                quick_editor_source,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key="metadata_quick_editor_widget",
                column_config={
                    "column": st.column_config.TextColumn("Field", disabled=True),
                    "Sensitivity": st.column_config.TextColumn("Sensitivity", disabled=True),
                    "data_type": st.column_config.TextColumn("Field type", disabled=True),
                    "control_action": st.column_config.SelectboxColumn("Field action", options=ALL_CONTROL_ACTIONS),
                    "include": st.column_config.CheckboxColumn("Include"),
                    "Current handling": st.column_config.TextColumn("Current handling", disabled=True, width="large"),
                    "notes": st.column_config.TextColumn("Notes", width="large"),
                },
            )

            updated_frame = st.session_state.metadata_editor_df.copy()
            for _, row in quick_editor.iterrows():
                mask = updated_frame["column"] == row["column"]
                if not mask.any():
                    continue
                base_item = editor_frame_to_metadata(updated_frame.loc[mask].head(1))[0]
                chosen_action = sanitize_control_action(base_item, str(row["control_action"]))
                updated_frame.loc[mask, "control_action"] = chosen_action
                updated_frame.loc[mask, "include"] = bool(row["include"]) and chosen_action != "Exclude"
                updated_frame.loc[mask, "notes"] = str(row["notes"])
            updated_frame = normalize_metadata_frame(updated_frame)
            if not updated_frame.equals(st.session_state.metadata_editor_df):
                st.session_state.metadata_editor_df = updated_frame
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None

            latest_metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
            action_counts = pd.Series([item["control_action"] for item in latest_metadata]).value_counts()
            st.dataframe(
                pd.DataFrame({"Field action": action_counts.index, "Fields": action_counts.values}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(build_quick_controls_frame(metadata), use_container_width=True, hide_index=True)
    with edit_tab:
        if has_permission("edit_metadata") and not metadata_edit_locked:
            editor = st.data_editor(
                st.session_state.metadata_editor_df,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key="metadata_editor_widget",
                column_config={
                    "column": st.column_config.TextColumn("Field", disabled=True),
                    "include": st.column_config.CheckboxColumn("Include"),
                    "data_type": st.column_config.SelectboxColumn(
                        "Field type",
                        options=["identifier", "numeric", "categorical", "binary", "date"],
                    ),
                    "strategy": st.column_config.SelectboxColumn(
                        "Generation rule",
                        options=["new_token", "sample_plus_noise", "sample_category", "sample_plus_jitter"],
                    ),
                    "control_action": st.column_config.SelectboxColumn("Field action", options=ALL_CONTROL_ACTIONS),
                    "nullable": st.column_config.CheckboxColumn("Nullable"),
                    "notes": st.column_config.TextColumn("Notes", width="large"),
                },
            )
            normalized_editor = normalize_metadata_frame(editor)
            if not normalized_editor.equals(st.session_state.metadata_editor_df):
                st.session_state.metadata_editor_df = normalized_editor
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None
        else:
            st.dataframe(st.session_state.metadata_editor_df, use_container_width=True, hide_index=True)
    with record_tab:
        if active_package is None and not st.session_state.metadata_package_log:
            st.info("No submitted package yet.")
        else:
            if active_package is not None:
                st.dataframe(build_metadata_review_frame(active_package["snapshot"]), use_container_width=True, hide_index=True)
                if active_package.get("review_note"):
                    st.info(
                        f"Manager note from {active_package.get('reviewed_by') or 'reviewer'}"
                        f"{f' at {active_package.get('reviewed_at')}' if active_package.get('reviewed_at') else ''}: "
                        f"{active_package.get('review_note')}"
                    )
            if st.session_state.metadata_package_log:
                st.markdown("**Package history**")
                st.dataframe(build_metadata_package_log_frame(), use_container_width=True, hide_index=True)

    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    sync_metadata_workflow_state(metadata)
    current_signature = build_metadata_signature(metadata)

    action_cols = st.columns([0.9, 1.1], gap="large")
    mark_disabled = metadata_edit_locked and st.session_state.metadata_status in {"In Review", "Approved"}
    if action_cols[0].button("Mark data settings reviewed", type="primary", use_container_width=True, disabled=mark_disabled):
        st.session_state.settings_reviewed = True
        st.session_state.settings_review_signature = current_signature
        record_audit_event("Data settings reviewed", "Field-level settings were finalized for submission.", status="Completed")
        rerun_with_persist()

    can_continue = st.session_state.settings_reviewed and st.session_state.settings_review_signature == current_signature
    if action_cols[1].button("Continue to Reviewer Sign-off", use_container_width=True, disabled=not can_continue):
        st.session_state.current_step = 3
        st.rerun()

    return metadata, controls


def render_step_four(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    render_section_header(3, "Transfer the analyst-confirmed package to the reviewer queue.")
    render_previous_step_control(3)

    if not has_active_dataset():
        st.info("Confirm metadata assumptions before initiating reviewer handoff.")
        return metadata, controls

    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    sync_metadata_workflow_state(metadata)
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    active_package = active_metadata_package_record(metadata)
    review_package = current_review_package_record()
    package_summary = summarize_metadata_package(metadata)

    # Review package summary (replaces KPI metric tiles with governance handoff card)
    _step4_classified = classify_hygiene_issues(st.session_state.hygiene) if st.session_state.get("hygiene") else []
    _step4_blockers = sum(1 for c in _step4_classified if c.get("classification") == "blocker")
    _step4_warnings = sum(1 for c in _step4_classified if c.get("classification") == "warning")

    render_review_package_summary(
        request_id=str(st.session_state.active_request_id or ""),
        settings_reviewed=bool(st.session_state.settings_reviewed),
        metadata_status=st.session_state.metadata_status,
        package_id=(review_package["package_id"] if review_package else (active_package["package_id"] if active_package else "")),
        dataset_label=current_dataset_label(),
        included_fields=int(package_summary["included_fields"]),
        sensitive_fields=int(package_summary["restricted_fields"] + package_summary["sensitive_fields"]),
        targeted_actions=int(package_summary["targeted_actions"]),
        current_owner="Data Analyst" if st.session_state.metadata_status in ("Draft", "Changes Requested") else "Manager / Reviewer",
        next_owner="Manager / Reviewer" if st.session_state.metadata_status in ("Draft", "Changes Requested", "In Review") else "Data Analyst",
        blockers_remaining=_step4_blockers,
        warnings_remaining=_step4_warnings,
        submitted_by=st.session_state.metadata_submitted_by if review_package else None,
        submitted_at=st.session_state.metadata_submitted_at if review_package else None,
    )

    if st.session_state.metadata_status == "Approved" and active_package is not None:
        st.success(
            f"Reviewer sign-off complete. Approved by {active_package['approved_by']} at {active_package['approved_at']}. Generation unlocked."
        )
    elif st.session_state.metadata_status == "In Review" and review_package is not None:
        st.info(
            f"Package {review_package['package_id']} awaiting reviewer sign-off. Submitted by {review_package['submitted_by']} at {review_package['submitted_at']}."
        )
    elif st.session_state.metadata_status == "Changes Requested" and active_package is not None:
        st.warning(active_package.get("review_note") or "Reviewer returned the package — changes requested before approval.")
    elif st.session_state.metadata_status == "Rejected" and active_package is not None:
        st.error(active_package.get("review_note") or "Reviewer rejected the submitted package.")

    left_col, right_col = st.columns([1.15, 0.85], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown("**Request summary**")
            request_summary = pd.DataFrame(
                [
                    {"Item": "Request id", "Value": active_package["package_id"] if active_package else "Draft"},
                    {"Item": "Dataset", "Value": current_dataset_label()},
                    {"Item": "Included fields", "Value": package_summary["included_fields"]},
                    {"Item": "Sensitive fields", "Value": package_summary["restricted_fields"] + package_summary["sensitive_fields"]},
                    {"Item": "Targeted actions", "Value": package_summary["targeted_actions"]},
                ]
            )
            st.dataframe(request_summary, use_container_width=True, hide_index=True)
            st.dataframe(build_metadata_review_frame(metadata), use_container_width=True, hide_index=True)

    with right_col:
        with st.container(border=True):
            st.markdown("**Decision panel**")
            if has_permission("submit_metadata"):
                if st.session_state.metadata_status == "Approved":
                    st.success("This request is already approved.")
                elif st.session_state.metadata_status == "In Review":
                    st.info("This request is already with the Manager / Reviewer.")
                else:
                    submit_disabled = not st.session_state.settings_reviewed
                    if st.button("Submit for reviewer sign-off", type="primary", use_container_width=True, disabled=submit_disabled):
                        st.session_state.metadata_status = "In Review"
                        st.session_state.metadata_submitted_by = st.session_state.current_role
                        st.session_state.metadata_submitted_at = format_timestamp()
                        st.session_state.metadata_approved_by = None
                        st.session_state.metadata_approved_at = None
                        st.session_state.metadata_review_note = None
                        st.session_state.metadata_reviewed_by = None
                        st.session_state.metadata_reviewed_at = None
                        st.session_state.last_reviewed_metadata_signature = build_metadata_signature(metadata)
                        submitted_record = register_metadata_submission(metadata)
                        clear_generation_outputs()
                        record_audit_event(
                            "Request submitted for review",
                            f"{submitted_record['package_id']} submitted by {st.session_state.current_role}.",
                            status="Submitted",
                        )
                        rerun_with_persist()
                    if submit_disabled:
                        st.caption("Review the data settings first before submitting this request.")
            else:
                review_note = st.text_area(
                    "Review note",
                    placeholder="Add approval context or revision guidance for the Data Analyst.",
                    key="manager_review_note",
                )
                if review_package is None:
                    st.info("No request is waiting for your review right now.")
                else:
                    action_cols = st.columns(3)
                    if action_cols[0].button("Approve request", type="primary", use_container_width=True):
                        st.session_state.metadata_status = "Approved"
                        st.session_state.metadata_approved_by = st.session_state.current_role
                        st.session_state.metadata_approved_at = format_timestamp()
                        st.session_state.metadata_review_note = review_note.strip() or None
                        st.session_state.metadata_reviewed_by = st.session_state.current_role
                        st.session_state.metadata_reviewed_at = format_timestamp()
                        st.session_state.last_reviewed_metadata_signature = build_metadata_signature(metadata)
                        approved_record = register_metadata_approval(metadata)
                        if review_note.strip():
                            approved_record["review_note"] = review_note.strip()
                        record_audit_event(
                            "Request approved",
                            f"{approved_record['package_id']} approved by {st.session_state.current_role}.",
                            status="Approved",
                        )
                        rerun_with_persist()
                    change_disabled = not review_note.strip()
                    if action_cols[1].button("Request changes", use_container_width=True, disabled=change_disabled):
                        feedback_record = register_metadata_feedback(metadata, "Changes Requested", review_note)
                        st.session_state.metadata_status = "Changes Requested"
                        st.session_state.metadata_review_note = review_note.strip()
                        st.session_state.metadata_reviewed_by = st.session_state.current_role
                        st.session_state.metadata_reviewed_at = format_timestamp()
                        st.session_state.metadata_approved_by = None
                        st.session_state.metadata_approved_at = None
                        record_audit_event(
                            "Request changes requested",
                            f"{feedback_record['package_id'] if feedback_record else 'Current package'} returned with manager feedback.",
                            status="Returned",
                        )
                        rerun_with_persist()
                    if action_cols[2].button("Reject request", use_container_width=True, disabled=change_disabled):
                        feedback_record = register_metadata_feedback(metadata, "Rejected", review_note)
                        st.session_state.metadata_status = "Rejected"
                        st.session_state.metadata_review_note = review_note.strip()
                        st.session_state.metadata_reviewed_by = st.session_state.current_role
                        st.session_state.metadata_reviewed_at = format_timestamp()
                        st.session_state.metadata_approved_by = None
                        st.session_state.metadata_approved_at = None
                        record_audit_event(
                            "Request rejected",
                            f"{feedback_record['package_id'] if feedback_record else 'Current package'} rejected by {st.session_state.current_role}.",
                            status="Rejected",
                        )
                        rerun_with_persist()

            history_cols = st.columns(2)
            history_cols[0].metric("Submitted by", st.session_state.metadata_submitted_by or "Not submitted")
            history_cols[1].metric("Submitted at", st.session_state.metadata_submitted_at or "Pending")
            history_cols[0].metric("Reviewed by", st.session_state.metadata_reviewed_by or "Not reviewed")
            history_cols[1].metric("Approved at", st.session_state.metadata_approved_at or "Pending")

    if st.session_state.metadata_package_log:
        st.markdown("**Request history**")
        st.dataframe(build_metadata_package_log_frame(), use_container_width=True, hide_index=True)

    footer_cols = st.columns([0.9, 1.1], gap="large")
    if footer_cols[0].button("Return to Metadata Assumptions", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()
    if footer_cols[1].button(
        "Continue to Generate Synthetic Package",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.metadata_status != "Approved",
    ):
        st.session_state.current_step = 4
        st.rerun()

    return metadata, controls


def render_step_five(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    render_section_header(4, "Synthesize the approved metadata package. No source records copied.")
    render_previous_step_control(4)

    if st.session_state.metadata_status != "Approved":
        st.warning("Generation gated — awaiting reviewer sign-off on the metadata package.")
        return

    eligible_distribution_columns = [
        item["column"] for item in metadata if item["include"] and item["data_type"] != "identifier"
    ]
    controls = sync_generation_preset_label(controls)
    controls["locked_columns"] = [column for column in controls.get("locked_columns", []) if column in eligible_distribution_columns]
    posture_label = "Balanced" if controls["fidelity_priority"] < 70 else "Higher fidelity"
    locked_preview = ", ".join(controls.get("locked_columns", [])[:3]) or "None selected"
    generate_clicked = False

    if has_permission("generate"):
        top_controls = st.columns(2, gap="large")
        with top_controls[0]:
            with st.container(border=True):
                st.markdown("**Core generation**")
                preset_options = list(GENERATION_PRESETS.keys()) + ["Custom"]
                current_preset = str(controls.get("generation_preset", "Balanced"))
                if current_preset not in preset_options:
                    current_preset = "Custom"
                selected_preset = st.selectbox("Generation preset", options=preset_options, index=preset_options.index(current_preset))
                if selected_preset != current_preset:
                    controls = apply_generation_preset(controls, selected_preset) if selected_preset != "Custom" else {**controls, "generation_preset": "Custom"}
                controls["fidelity_priority"] = st.slider("Privacy versus fidelity", 0, 100, int(controls["fidelity_priority"]))
                controls["synthetic_rows"] = int(st.number_input("Synthetic row count", value=int(controls["synthetic_rows"]), min_value=1, step=10))
        with top_controls[1]:
            with st.container(border=True):
                st.markdown("**Distribution & structure controls**")
                controls["locked_columns"] = st.multiselect("Lock key distributions", options=eligible_distribution_columns, default=controls["locked_columns"])
                controls["correlation_preservation"] = st.slider("Correlation preservation", 0, 100, int(controls["correlation_preservation"]))
                controls["rare_case_retention"] = st.slider("Rare case retention", 0, 100, int(controls["rare_case_retention"]))
                st.caption(f"Locked columns: {locked_preview}")

        bottom_controls = st.columns([1.05, 0.95], gap="large")
        with bottom_controls[0]:
            with st.container(border=True):
                st.markdown("**Data quality & noise controls**")
                controls["noise_level"] = st.slider("Noise level", 0, 100, int(controls["noise_level"]))
                missingness_options = ["Preserve source pattern", "Reduce missingness", "Fill gaps"]
                controls["missingness_pattern"] = st.selectbox("Missingness pattern", options=missingness_options, index=missingness_options.index(controls["missingness_pattern"]))
                outlier_options = ["Preserve tails", "Clip extremes", "Smooth tails"]
                controls["outlier_strategy"] = st.selectbox("Outlier strategy", options=outlier_options, index=outlier_options.index(controls["outlier_strategy"]))
        with bottom_controls[1]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="generate-action-card">
                        <div class="control-card-kicker">Generation action</div>
                        <div class="generate-action-title">Run the approved request</div>
                        <div class="generate-action-text">
                            {controls['synthetic_rows']} rows, {len(controls.get('locked_columns', []))} locked distributions,
                            correlation {controls.get('correlation_preservation', 0)}/100, and noise {controls.get('noise_level', 0)}/100.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                info_cols = st.columns(2)
                info_cols[0].metric("Posture", posture_label)
                info_cols[1].metric("Outlier strategy", controls.get("outlier_strategy", "Preserve tails"))
                generate_clicked = st.button("Generate synthetic dataset", type="primary", use_container_width=True)

        controls = sync_generation_preset_label(controls)
        st.session_state.controls = controls
    else:
        render_role_restriction("This role can review the approved generation settings and the current run status.")
        read_cols = st.columns(3, gap="large")
        read_rows = [
            (str(controls.get("generation_preset", "Balanced")), posture_label, "Current generation preset."),
            ("Locked distributions", str(len(controls.get("locked_columns", []))), locked_preview),
            ("Noise profile", str(controls.get("outlier_strategy", "Preserve tails")), f"Missingness: {controls.get('missingness_pattern', 'Preserve source pattern')}"),
        ]
        for col, (title, value, detail) in zip(read_cols, read_rows):
            col.markdown(
                f"""
                <div class="state-card">
                    <h4>{title}</h4>
                    <div class="state-value">{value}</div>
                    <div class="state-text">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if has_permission("generate") and generate_clicked:
        progress = st.progress(0, text="Preparing approved request...")
        time.sleep(0.12)
        progress.progress(30, text="Applying data settings and generation controls...")
        time.sleep(0.12)
        synthetic_df, generation_summary = generate_synthetic_data(st.session_state.source_df, metadata, controls)
        progress.progress(72, text="Running fidelity and privacy checks...")
        validation = validate_synthetic_data(st.session_state.source_df, synthetic_df, metadata, controls)
        time.sleep(0.12)
        progress.progress(100, text="Synthetic generation completed.")
        st.session_state.synthetic_df = synthetic_df
        st.session_state.generation_summary = generation_summary
        st.session_state.validation = validation
        st.session_state.last_generation_signature = build_generation_signature(metadata, controls)
        st.session_state.release_status = "Generated"
        record_audit_event(
            "Synthetic dataset generated",
            f"{generation_summary['rows_generated']} rows generated for approved package {st.session_state.current_metadata_package_id or 'current request'}.",
            status="Completed",
        )
        rerun_with_persist()

    if st.session_state.synthetic_df is None:
        st.info("No synthetic package has been generated yet.")
        return

    if has_stale_generation(metadata, controls):
        st.warning("Metadata assumptions changed after last run. Regenerate to refresh the synthetic package.")

    summary = st.session_state.generation_summary
    render_synthetic_verification_summary(
        rows_generated=int(summary["rows_generated"]),
        fields_included=int(summary["columns_generated"]),
        noise_posture=str(summary["noise_mode"]),
        verification_complete=st.session_state.validation is not None,
        metadata_package_id=st.session_state.current_metadata_package_id,
        approved_by=st.session_state.metadata_approved_by,
    )

    preview_tab, note_tab = st.tabs(["Synthetic preview", "Run settings"])
    with preview_tab:
        st.dataframe(st.session_state.synthetic_df.head(12), use_container_width=True, hide_index=True)
    with note_tab:
        st.dataframe(pd.DataFrame(build_generation_control_rows(controls)), use_container_width=True, hide_index=True)

    footer_cols = st.columns([0.9, 1.1], gap="large")
    if has_permission("rollback") and footer_cols[0].button("Return to Reviewer Sign-off", use_container_width=True):
        st.session_state.current_step = 3
        st.rerun()
    if footer_cols[1].button("Continue to Release Synthetic Package", type="primary", use_container_width=True):
        st.session_state.current_step = 5
        st.rerun()


def render_step_six(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    render_section_header(5, "Release the verified synthetic package for controlled distribution.")
    render_previous_step_control(5)

    if st.session_state.synthetic_df is None or st.session_state.validation is None:
        st.warning("Synthetic package not yet available for release.")
        return

    if has_stale_generation(metadata, controls):
        st.warning("Release held — metadata assumptions changed after last generation. Regenerate to refresh.")
        return

    validation_report = build_validation_report(metadata, controls)

    # Compact status strip (no prominent scores)
    release_state = effective_release_status(metadata, controls)
    gen_by = st.session_state.current_role if has_permission("generate") else (st.session_state.metadata_submitted_by or "System")
    shared_at = st.session_state.results_shared_at or "Not yet shared"
    st.markdown(
        f"""
        <div style="display:flex;gap:1.5rem;padding:0.75rem 1rem;background:var(--surface);border:1px solid var(--line);
            border-radius:14px;margin-bottom:0.85rem;font-size:0.86rem;box-shadow:var(--shadow);">
            <div><span style="color:var(--muted);font-weight:600;">Release state:</span> <strong style="color:var(--text);">{release_state}</strong></div>
            <div><span style="color:var(--muted);font-weight:600;">Generated by:</span> <strong style="color:var(--text);">{gen_by}</strong></div>
            <div><span style="color:var(--muted);font-weight:600;">Shared:</span> <strong style="color:var(--text);">{shared_at}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Agent: Release Readiness Verdicts ──
    verdicts = build_release_readiness_verdicts(
        validation=st.session_state.validation,
        metadata=metadata,
        hygiene=st.session_state.hygiene,
        synthetic_df_columns=list(st.session_state.synthetic_df.columns),
    )
    render_release_readiness_verdicts(verdicts)

    # Agent: Stakeholder interpretation
    step6_readiness = compute_agent_readiness(
        profile=st.session_state.get("profile"),
        hygiene=st.session_state.get("hygiene"),
        metadata=metadata, controls=controls,
        validation=st.session_state.get("validation"),
        intake_confirmed=True, hygiene_reviewed=True, settings_reviewed=True,
        metadata_status=st.session_state.get("metadata_status", "Approved"),
        synthetic_ready=True,
        results_shared=bool(st.session_state.get("results_shared_at")),
    )
    stakeholder_blocks = build_stakeholder_interpretations(
        validation=st.session_state.validation,
        hygiene=st.session_state.hygiene,
        metadata=metadata,
        readiness=step6_readiness,
    )
    render_stakeholder_interpretations(stakeholder_blocks)

    left_col, right_col = st.columns([1.08, 0.92], gap="large")
    with left_col:
        result_tabs = st.tabs(["Synthetic output", "Validation summary", "Audit log"])
        with result_tabs[0]:
            st.dataframe(st.session_state.synthetic_df.head(20), use_container_width=True, hide_index=True)
        with result_tabs[1]:
            validation = st.session_state.validation
            st.markdown(
                f"""
                <div style="padding:0.6rem 0.85rem;background:var(--surface-soft);border:1px solid var(--line);
                    border-radius:12px;margin-bottom:0.7rem;font-size:0.82rem;color:var(--muted);">
                    <span style="color:var(--text);font-weight:600;">Supporting metrics:</span>
                    fidelity {validation['fidelity_score']} · privacy {validation['privacy_score']} · overall {validation['overall_score']}.
                    Release decisions are driven by the verdict panel above, not raw scores.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(build_comparison_table(st.session_state.source_df, st.session_state.synthetic_df, metadata), use_container_width=True, hide_index=True)
            available_columns = [
                item["column"]
                for item in metadata
                if item["include"] and item["column"] in st.session_state.synthetic_df.columns and item["data_type"] != "identifier"
            ]
            if available_columns:
                selected_column = st.selectbox("Distribution comparison", options=available_columns, key="distribution_column")
                comparison = build_distribution_comparison(metadata, selected_column)
                if comparison["kind"] == "line":
                    st.line_chart(comparison["frame"], use_container_width=True)
                elif comparison["kind"] == "bar":
                    st.bar_chart(comparison["frame"], use_container_width=True)
                if not comparison["frame"].empty:
                    st.dataframe(comparison["frame"].reset_index(), use_container_width=True, hide_index=True)
                st.caption(comparison["note"])
        with result_tabs[2]:
            st.dataframe(pd.DataFrame(st.session_state.audit_events), use_container_width=True, hide_index=True)

    with right_col:
        with st.container(border=True):
            st.markdown("**Release actions**")
            if has_permission("download_results"):
                st.download_button(
                    "Download synthetic package",
                    data=st.session_state.synthetic_df.to_csv(index=False).encode("utf-8"),
                    file_name="synthetic_package.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.download_button(
                    "Download verification report",
                    data=validation_report.encode("utf-8"),
                    file_name="verification_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                if st.button("Mark package released", type="primary", use_container_width=True, disabled=bool(st.session_state.results_shared_at)):
                    st.session_state.results_shared_by = st.session_state.current_role
                    st.session_state.results_shared_at = format_timestamp()
                    record_audit_event("Package released", f"Synthetic package released by {st.session_state.current_role}.", status="Released")
                    rerun_with_persist()
            else:
                st.info("The Manager / Reviewer can review the final output and validation summary here.")
                st.download_button(
                    "Download validation report",
                    data=validation_report.encode("utf-8"),
                    file_name="validation_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            if st.session_state.results_shared_at:
                st.success(f"Shared by {st.session_state.results_shared_by} at {st.session_state.results_shared_at}.")

            if active_metadata_package_record(metadata) is not None:
                package = active_metadata_package_record(metadata)
                st.metric("Approved package", package["package_id"])
                st.metric("Approved by", package.get("approved_by") or "Pending")
                st.metric("Review note", package.get("review_note") or "None")

    # End-of-step transition to Step 7
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    footer_cols = st.columns([0.35, 0.65], gap="large")
    if has_permission("rollback") and footer_cols[0].button("Return to Generate Package", use_container_width=True):
        st.session_state.current_step = 4
        st.rerun()
    if footer_cols[1].button(
        "Continue to Analyze Synthetic Data  →",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.synthetic_df is None,
    ):
        st.session_state.current_step = 6
        st.rerun()


def render_step_seven(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    render_section_header(6, "Analyze the released synthetic package. Source data remains inside governance.")
    render_previous_step_control(6)

    if st.session_state.synthetic_df is None or st.session_state.validation is None:
        st.warning("Generate and verify synthetic data before analysis.")
        return

    if has_stale_generation(metadata, controls):
        st.warning("Synthetic output is out of date. Regenerate before analyzing.")
        return

    # ── Prominent privacy boundary (core to this step, not a footnote) ──
    st.markdown(
        """
        <div style="padding:0.85rem 1rem;background:#EDF9F3;border:1px solid rgba(19,107,72,0.2);
            border-left:4px solid #136B48;border-radius:12px;margin-bottom:1.1rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#136B48;margin-bottom:0.25rem;">
                Privacy boundary
            </div>
            <div style="font-size:0.9rem;color:#2D3E50;line-height:1.5;">
                Only the <strong>synthetic package</strong> is available for analysis below.
                No source records, no raw metadata, and no real patient data are exposed —
                even when optional external chat analysis is used
                <span style="font-family:monospace;color:#668097;font-weight:500;font-size:0.78rem;">(GOV-01)</span>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    synthetic_df = st.session_state.synthetic_df

    # ── Mode selection as prominent side-by-side cards ──
    if "step7_analysis_mode" not in st.session_state:
        st.session_state.step7_analysis_mode = "Local analysis"
    current_mode = st.session_state.step7_analysis_mode

    st.markdown(
        """
        <div style="margin:0.25rem 0 0.6rem 0;">
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#2D3E50;">Choose analysis mode</div>
            <div style="font-size:0.82rem;color:#668097;margin-top:0.15rem;">Only synthetic output is eligible for this step.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode_cols = st.columns(2, gap="medium")

    # Local analysis card
    local_selected = current_mode == "Local analysis"
    if local_selected:
        local_border = "#004B8B"; local_bg = "#EBF1F7"; local_shadow = "0 0 0 2px rgba(0,75,139,0.18)"
        local_badge = '<span style="display:inline-block;padding:0.12rem 0.55rem;border-radius:999px;font-size:0.7rem;font-weight:700;background:#004B8B;color:#FFFFFF;letter-spacing:0.04em;">&#10003; SELECTED</span>'
    else:
        local_border = "#DDE5ED"; local_bg = "#FFFFFF"; local_shadow = "none"
        local_badge = '<span style="font-size:0.7rem;color:transparent;">&nbsp;</span>'

    with mode_cols[0]:
        st.markdown(
            f"""
            <div style="padding:1rem 1.1rem;background:{local_bg};border:1.5px solid {local_border};
                border-radius:12px;min-height:120px;box-shadow:{local_shadow};transition:all 0.15s ease;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                    <div style="font-size:1rem;font-weight:700;color:#2D3E50;">Local analysis</div>
                    {local_badge}
                </div>
                <div style="font-size:0.84rem;color:#668097;line-height:1.5;">
                    Review the synthetic dataset inside the governed workspace.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Use local analysis", key="step7_select_local", use_container_width=True, disabled=local_selected):
            st.session_state.step7_analysis_mode = "Local analysis"
            st.rerun()

    # Chat analysis card
    chat_selected = current_mode == "Chat analysis (synthetic data only)"
    if chat_selected:
        chat_border = "#004B8B"; chat_bg = "#EBF1F7"; chat_shadow = "0 0 0 2px rgba(0,75,139,0.18)"
        chat_badge = '<span style="display:inline-block;padding:0.12rem 0.55rem;border-radius:999px;font-size:0.7rem;font-weight:700;background:#004B8B;color:#FFFFFF;letter-spacing:0.04em;">&#10003; SELECTED</span>'
    else:
        chat_border = "#DDE5ED"; chat_bg = "#FFFFFF"; chat_shadow = "none"
        chat_badge = '<span style="font-size:0.7rem;color:transparent;">&nbsp;</span>'

    with mode_cols[1]:
        st.markdown(
            f"""
            <div style="padding:1rem 1.1rem;background:{chat_bg};border:1.5px solid {chat_border};
                border-radius:12px;min-height:120px;box-shadow:{chat_shadow};transition:all 0.15s ease;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                    <div style="font-size:1rem;font-weight:700;color:#2D3E50;">Chat analysis
                        <span style="font-size:0.7rem;font-weight:600;color:#9C6A17;margin-left:0.4rem;">synthetic output only</span>
                    </div>
                    {chat_badge}
                </div>
                <div style="font-size:0.84rem;color:#668097;line-height:1.5;">
                    Use optional external chat analysis on synthetic output only. No source records or raw metadata are sent externally.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Use chat analysis", key="step7_select_chat", use_container_width=True, disabled=chat_selected):
            st.session_state.step7_analysis_mode = "Chat analysis (synthetic data only)"
            st.rerun()

    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

    # ── Render selected mode ──
    analysis_mode = st.session_state.step7_analysis_mode

    if analysis_mode == "Local analysis":
        numeric_cols = synthetic_df.select_dtypes(include="number").columns.tolist()
        missing_cells = int(synthetic_df.isna().sum().sum())

        # Styled capsule cards (matching governed workflow panels)
        def _stat_capsule(kicker: str, value: str, detail: str, accent: str = "#08467D", bg: str = "#EBF1F7") -> str:
            return (
                f'<div style="flex:1;min-width:0;padding:0.75rem 0.95rem;background:{bg};'
                f'border:1px solid {accent}22;border-radius:10px;">'
                f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:{accent};margin-bottom:0.2rem;">{kicker}</div>'
                f'<div style="font-size:1.25rem;font-weight:700;color:#2D3E50;line-height:1.15;">{value}</div>'
                f'<div style="font-size:0.74rem;color:#668097;margin-top:0.15rem;line-height:1.35;">{detail}</div>'
                f'</div>'
            )

        stats_html = (
            '<div class="action-shell" style="margin-bottom:0.85rem;">'
            '<h4>Synthetic package — local analysis</h4>'
            '<div style="display:flex;gap:0.65rem;flex-wrap:wrap;">'
            + _stat_capsule("Records", f"{len(synthetic_df):,}", "Synthetic rows in package")
            + _stat_capsule("Fields", str(len(synthetic_df.columns)), "Columns included")
            + _stat_capsule("Numeric fields", str(len(numeric_cols)), "Eligible for distribution view")
            + _stat_capsule(
                "Missing cells",
                f"{missing_cells:,}",
                "Gaps retained from metadata",
                accent=("#136B48" if missing_cells == 0 else "#9C6A17"),
                bg=("#EDF9F3" if missing_cells == 0 else "#FFF6E3"),
            )
            + '</div></div>'
        )
        st.markdown(stats_html, unsafe_allow_html=True)

        preview_tab, stats_tab, dist_tab = st.tabs(["Data preview", "Descriptive statistics", "Distribution"])
        with preview_tab:
            st.dataframe(synthetic_df.head(20), use_container_width=True, hide_index=True)
        with stats_tab:
            if numeric_cols:
                st.dataframe(synthetic_df[numeric_cols].describe().round(2), use_container_width=True)
            else:
                st.info("No numeric columns in the synthetic output.")
        with dist_tab:
            if numeric_cols:
                selected_col = st.selectbox("Select field", numeric_cols, key="step7_dist_col")
                st.bar_chart(synthetic_df[selected_col].dropna().value_counts().head(20).sort_index())
            cat_cols = [c for c in synthetic_df.columns if c not in numeric_cols and synthetic_df[c].nunique() <= 20]
            if cat_cols:
                selected_cat = st.selectbox("Categorical field", cat_cols, key="step7_cat_col")
                st.bar_chart(synthetic_df[selected_cat].fillna("Missing").value_counts().head(15))

    else:
        st.markdown("**Synthetic package — chat analysis**")
        st.markdown(
            '<div style="padding:0.6rem 0.85rem;background:#FFF6E3;border:1px solid rgba(138,97,22,0.18);'
            'border-radius:10px;margin-bottom:0.8rem;font-size:0.83rem;color:#9C6A17;">'
            'Optional external call. Only aggregate synthetic statistics are transmitted — '
            'no source records, no raw metadata, no real patient data.</div>',
            unsafe_allow_html=True,
        )

        from src.chat_assistant import build_chat_context, generate_chat_reply, generate_demo_chat_reply
        if "step7_chat" not in st.session_state:
            st.session_state.step7_chat = []

        synth_context = (
            f"Synthetic dataset: {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns.\n"
            f"Columns: {', '.join(synthetic_df.columns[:15])}\n"
        )
        for col in synthetic_df.select_dtypes(include="number").columns[:6]:
            synth_context += f"{col}: mean={synthetic_df[col].mean():.1f}, std={synthetic_df[col].std():.1f}, min={synthetic_df[col].min():.1f}, max={synthetic_df[col].max():.1f}\n"
        for col in synthetic_df.select_dtypes(exclude="number").columns[:4]:
            top = synthetic_df[col].value_counts().head(3)
            synth_context += f"{col}: {', '.join(f'{k} ({v})' for k, v in top.items())}\n"

        for msg in st.session_state.step7_chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ask about the synthetic package..."):
            st.session_state.step7_chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                reply = generate_chat_reply(
                    api_key=api_key,
                    user_message=prompt,
                    chat_history=st.session_state.step7_chat,
                    context=f"SYNTHETIC DATA ANALYSIS MODE. Only synthetic data is being analyzed.\n{synth_context}",
                )
            else:
                reply = generate_demo_chat_reply(
                    prompt,
                    st.session_state.get("profile", {}),
                    st.session_state.get("hygiene", {}),
                    st.session_state.get("controls", {}),
                    st.session_state.get("validation"),
                )

            st.session_state.step7_chat.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="H", layout="wide", initial_sidebar_state="collapsed")
    inject_styles()
    initialize_app_state()
    process_pending_workspace_actions()

    if not st.session_state.authenticated:
        render_login_screen()
        return

    ensure_dataset_loaded()
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    controls = st.session_state.controls
    sync_metadata_workflow_state(metadata)
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    controls = st.session_state.controls
    ensure_role_step_visibility(metadata, controls)

    render_header(metadata, controls)
    render_step_navigation(metadata, controls)
    render_action_center(metadata, controls)

    # ── Agent Decision Log (step-aware) ──
    current_step = st.session_state.current_step

    if current_step == 0:
        # Step 0: Agent Decision Log is rendered INSIDE render_step_one at the proper position
        # (between the upload+readiness area and the intake summary)
        pass
    else:
        # Steps 1+: full consolidated decision log with readiness engine
        readiness = compute_agent_readiness(
            profile=st.session_state.get("profile"),
            hygiene=st.session_state.get("hygiene"),
            metadata=metadata, controls=controls,
            validation=st.session_state.get("validation"),
            intake_confirmed=st.session_state.get("intake_confirmed", False),
            hygiene_reviewed=st.session_state.get("hygiene_reviewed", False),
            settings_reviewed=st.session_state.get("settings_reviewed", False),
            metadata_status=st.session_state.get("metadata_status", "Draft"),
            synthetic_ready=st.session_state.get("synthetic_df") is not None,
            results_shared=bool(st.session_state.get("results_shared_at")),
        )
        # On Scan Data step, inline hygiene findings inside the decision log
        inline_hygiene = None
        if current_step == 1 and st.session_state.get("hygiene") is not None:
            inline_hygiene = classify_hygiene_issues(st.session_state.get("hygiene"))

        render_consolidated_decision_log(
            readiness=readiness,
            profile=st.session_state.get("profile"),
            hygiene=st.session_state.get("hygiene"),
            metadata=metadata, controls=controls,
            generation_summary=st.session_state.get("generation_summary"),
            validation=st.session_state.get("validation"),
            intake_confirmed=st.session_state.get("intake_confirmed", False),
            hygiene_reviewed=st.session_state.get("hygiene_reviewed", False),
            settings_reviewed=st.session_state.get("settings_reviewed", False),
            metadata_status=st.session_state.get("metadata_status", "Draft"),
            synthetic_ready=st.session_state.get("synthetic_df") is not None,
            results_shared=bool(st.session_state.get("results_shared_at")),
            classified_hygiene=inline_hygiene,
        )

    if current_step == 0:
        render_step_one(metadata)
    elif current_step == 1:
        render_step_two()
    elif current_step == 2:
        metadata, controls = render_step_three()
    elif current_step == 3:
        metadata, controls = render_step_four(metadata, controls)
    elif current_step == 4:
        render_step_five(metadata, controls)
    elif current_step == 5:
        render_step_six(metadata, controls)
    elif current_step == 6:
        render_step_seven(metadata, controls)

    persist_shared_workspace_state()


if __name__ == "__main__":
    main()
