from __future__ import annotations

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

APP_TITLE = "Healthcare Synthetic Data Workspace"
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data.csv"

STEP_CONFIG = [
    {
        "title": "Secure Intake",
        "description": "Bring a dataset into the workspace, detect sensitive fields, and set raw-data visibility.",
    },
    {
        "title": "Hygiene Review",
        "description": "Scan missingness, duplicates, coding drift, and invalid values before metadata is locked.",
    },
    {
        "title": "Metadata Review",
        "description": "Review field rules, assign ownership, and approve the de-identification plan.",
    },
    {
        "title": "Synthetic Generation",
        "description": "Generate synthetic output from the approved metadata package with visible controls.",
    },
    {
        "title": "Validation & Release",
        "description": "Validate source versus synthetic behavior and control governed export.",
    },
]

ROLE_VISIBLE_STEPS: dict[str, list[int]] = {
    "Clinician": [4],
    "Data Analyst": [0, 1, 2, 3, 4],
    "Compliance Officer": [2, 4],
    "IT / Security": [0, 4],
    "Admin Office": [4],
    "Master": [0, 1, 2, 3, 4],
}

ROLE_CONFIGS: dict[str, dict[str, Any]] = {
    "Clinician": {
        "password": "test",
        "summary": "Reviews privacy-safe summaries and validation outputs without editing source rules.",
        "permissions": {"view_summary_only"},
    },
    "Data Analyst": {
        "password": "test",
        "summary": "Uploads data, fixes hygiene issues, edits metadata, generates synthetic output, and requests release.",
        "permissions": {
            "upload",
            "view_raw",
            "remediate",
            "edit_metadata",
            "submit_metadata",
            "generate",
            "validate",
            "request_export",
            "rollback",
        },
    },
    "Compliance Officer": {
        "password": "test",
        "summary": "Reviews sensitive fields, approves metadata handling, and performs policy approval before release.",
        "permissions": {"approve_metadata", "approve_release_policy", "view_audit", "view_security"},
    },
    "IT / Security": {
        "password": "test",
        "summary": "Monitors access boundaries, PHI exposure, anomaly conditions, and audit history.",
        "permissions": {"view_audit", "view_security"},
    },
    "Admin Office": {
        "password": "test",
        "summary": "Handles final release authorization and sees only the approval-stage workflow that needs office action.",
        "permissions": {
            "approve_export",
            "view_audit",
            "view_security",
        },
    },
    "Master": {
        "password": "test",
        "summary": "Internal master role with full cross-workflow visibility and override capability for every step.",
        "permissions": {
            "upload",
            "view_raw",
            "remediate",
            "edit_metadata",
            "submit_metadata",
            "approve_metadata",
            "generate",
            "validate",
            "request_export",
            "approve_release_policy",
            "approve_export",
            "view_audit",
            "view_security",
            "rollback",
        },
    },
}

SHARED_WORKFLOW_KEYS = [
    "source_df",
    "source_label",
    "profile",
    "hygiene",
    "metadata_editor_df",
    "controls",
    "intake_confirmed",
    "hygiene_reviewed",
    "metadata_status",
    "metadata_submitted_by",
    "metadata_submitted_at",
    "metadata_approved_by",
    "metadata_approved_at",
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
    "audit_events",
]


@st.cache_resource
def get_shared_workspace_store() -> dict[str, Any]:
    return {"state": {}}


def load_shared_workspace_state() -> None:
    shared_state = get_shared_workspace_store()["state"]
    if not shared_state:
        return
    for key, value in shared_state.items():
        st.session_state[key] = deepcopy(value)


def persist_shared_workspace_state() -> None:
    shared_state = {}
    for key in SHARED_WORKFLOW_KEYS:
        if key in st.session_state:
            shared_state[key] = deepcopy(st.session_state[key])
    get_shared_workspace_store()["state"] = shared_state


def rerun_with_persist() -> None:
    persist_shared_workspace_state()
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

            :root {
                --brand: #0f4f95;
                --brand-deep: #0a3f79;
                --accent: #1cd8d3;
                --bg: #f4f8fc;
                --surface: rgba(255, 255, 255, 0.97);
                --surface-soft: #f7fafc;
                --line: #d5e1eb;
                --text: #18344f;
                --muted: #5d7891;
                --warn: #8a6116;
                --warn-bg: #fff7e6;
                --danger: #8f2d35;
                --danger-bg: #fff1f2;
                --good: #16704a;
                --good-bg: #eefbf4;
                --shadow: 0 12px 30px rgba(15, 79, 149, 0.08);
            }

            @media (prefers-color-scheme: dark) {
                :root {
                    --bg: #0b1725;
                    --surface: rgba(17, 31, 45, 0.98);
                    --surface-soft: #122233;
                    --line: #24415b;
                    --text: #ebf3f8;
                    --muted: #a8bfd1;
                    --warn: #f4cc7b;
                    --warn-bg: rgba(122, 86, 14, 0.26);
                    --danger: #f2a6ae;
                    --danger-bg: rgba(130, 36, 49, 0.25);
                    --good: #7fe1b6;
                    --good-bg: rgba(22, 112, 74, 0.25);
                    --shadow: 0 18px 38px rgba(0, 0, 0, 0.28);
                }
            }

            html {
                color-scheme: light dark;
            }

            html, body, [class*="css"] {
                font-family: "IBM Plex Sans", sans-serif;
                color: var(--text);
                background: var(--bg);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 0% 0%, rgba(15, 79, 149, 0.08), transparent 24%),
                    radial-gradient(circle at 100% 8%, rgba(28, 216, 211, 0.08), transparent 18%),
                    linear-gradient(180deg, var(--bg), var(--bg));
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
                max-width: 1440px;
                padding-top: 1rem;
                padding-bottom: 2.8rem;
            }

            .banner {
                background: linear-gradient(135deg, rgba(15, 79, 149, 0.98), rgba(15, 79, 149, 0.93) 65%, rgba(28, 216, 211, 0.78));
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 26px;
                padding: 1.45rem 1.6rem;
                color: #ffffff;
                box-shadow: 0 26px 54px rgba(15, 79, 149, 0.16);
                margin-bottom: 1rem;
            }

            .banner-kicker {
                display: inline-block;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 999px;
                padding: 0.34rem 0.72rem;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                margin-bottom: 0.8rem;
            }

            .banner h1 {
                margin: 0;
                font-size: 2.2rem;
                line-height: 1.08;
                color: #ffffff;
            }

            .banner p {
                margin: 0.7rem 0 0 0;
                max-width: 860px;
                line-height: 1.6;
                color: rgba(255,255,255,0.92);
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
                border-radius: 24px;
                padding: 1.05rem 1.15rem;
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
                border-radius: 20px;
                padding: 1rem;
                box-shadow: var(--shadow);
                min-height: 152px;
            }

            .state-card h4 {
                margin: 0;
                font-size: 0.88rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .state-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--brand);
                margin-top: 0.55rem;
                margin-bottom: 0.25rem;
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
                min-height: 250px;
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
                padding: 0.95rem;
                box-shadow: var(--shadow);
                min-height: 160px;
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

            .action-hero-card,
            .action-panel-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 22px;
                box-shadow: var(--shadow);
            }

            .action-hero-card {
                padding: 1.05rem 1.1rem;
                min-height: 168px;
            }

            .action-next-card {
                background:
                    linear-gradient(135deg, rgba(15, 79, 149, 0.08), rgba(28, 216, 211, 0.08)),
                    var(--surface);
                border: 1px solid rgba(15, 79, 149, 0.18);
                box-shadow: 0 18px 34px rgba(15, 79, 149, 0.1);
            }

            .action-panel-card {
                padding: 0.95rem 1rem;
                min-height: 170px;
            }

            .action-panel-card.primary {
                border-color: rgba(15, 79, 149, 0.16);
                background: linear-gradient(180deg, rgba(15, 79, 149, 0.04), rgba(28, 216, 211, 0.03)), var(--surface);
            }

            .action-role-name {
                font-size: 1.55rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.45rem;
            }

            .action-summary {
                color: var(--muted);
                line-height: 1.55;
                margin-bottom: 0.85rem;
            }

            .action-next-title {
                font-size: 1.55rem;
                line-height: 1.28;
                color: var(--text);
                font-weight: 700;
                max-width: 780px;
            }

            .action-subtle {
                color: var(--muted);
                line-height: 1.5;
                margin-top: 0.65rem;
                font-size: 0.92rem;
            }

            .steps-wrap {
                margin-top: 0.25rem;
            }

            .step-pill-compact {
                display: inline-block;
                margin-right: 0.32rem;
                margin-bottom: 0.38rem;
                padding: 0.26rem 0.58rem;
                border-radius: 999px;
                border: 1px solid rgba(15, 79, 149, 0.12);
                background: rgba(15, 79, 149, 0.06);
                color: var(--brand-deep);
                font-size: 0.78rem;
                font-weight: 600;
            }

            .task-card-title {
                font-size: 0.84rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.65rem;
            }

            .task-row {
                display: flex;
                align-items: flex-start;
                gap: 0.6rem;
                padding: 0.48rem 0;
                border-top: 1px solid rgba(15, 79, 149, 0.08);
            }

            .task-row:first-of-type {
                border-top: none;
                padding-top: 0;
            }

            .task-dot {
                width: 0.55rem;
                height: 0.55rem;
                border-radius: 999px;
                background: var(--accent);
                flex: 0 0 auto;
                margin-top: 0.38rem;
            }

            .task-dot.waiting {
                background: #f0b55a;
            }

            .task-dot.completed {
                background: #2aa56f;
            }

            .task-text {
                color: var(--text);
                line-height: 1.45;
                font-size: 0.95rem;
            }

            .task-more {
                color: var(--muted);
                font-size: 0.84rem;
                margin-top: 0.55rem;
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.65rem 0.8rem;
                box-shadow: var(--shadow);
            }

            div[data-testid="stMetric"] * {
                color: var(--text) !important;
            }

            div.stButton > button,
            div.stDownloadButton > button {
                min-height: 2.8rem;
                border-radius: 14px;
                border: 1px solid rgba(15, 79, 149, 0.18);
                background: var(--surface) !important;
                color: var(--text) !important;
                font-weight: 600;
            }

            div.stButton > button[kind="primary"] {
                background: var(--brand) !important;
                color: #ffffff !important;
                border: 1px solid rgba(10, 79, 147, 0.28) !important;
            }

            div[data-baseweb="select"] > div,
            div[data-testid="stNumberInput"] input,
            div[data-testid="stTextInput"] input,
            div[data-testid="stTextArea"] textarea {
                border-radius: 14px;
                background: var(--surface) !important;
                color: var(--text) !important;
                -webkit-text-fill-color: var(--text) !important;
                border: 1px solid var(--line) !important;
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
                background: var(--surface);
                border: 1px solid var(--line) !important;
                border-radius: 22px;
                box-shadow: var(--shadow);
                padding: 0.2rem 0.35rem 0.55rem 0.35rem;
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
            "event": event,
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


def set_source_dataframe(df: pd.DataFrame, label: str) -> None:
    profile = profile_dataframe(df)
    hygiene = review_hygiene(df, profile)
    metadata = build_metadata(df, profile)

    st.session_state.source_df = df
    st.session_state.source_label = label
    st.session_state.profile = profile
    st.session_state.hygiene = hygiene
    st.session_state.metadata_editor_df = metadata_to_editor_frame(metadata)
    st.session_state.controls = {
        "generation_preset": "Balanced",
        "fidelity_priority": 62,
        "synthetic_rows": max(len(df), 20),
        "locked_columns": [],
        "correlation_preservation": 40,
        "rare_case_retention": 35,
        "noise_level": 45,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Preserve tails",
        "seed": 42,
    }
    st.session_state.intake_confirmed = False
    st.session_state.hygiene_reviewed = False
    st.session_state.metadata_status = "Draft"
    st.session_state.metadata_submitted_by = None
    st.session_state.metadata_submitted_at = None
    st.session_state.metadata_approved_by = None
    st.session_state.metadata_approved_at = None
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
        "shared_workspace_loaded": False,
        "current_step": 0,
        "audit_events": [],
        "release_status": "Not ready",
        "export_requested_by": None,
        "export_policy_approved_by": None,
        "export_approved_by": None,
        "uploaded_signature": None,
        "metadata_submitted_at": None,
        "metadata_approved_at": None,
        "current_metadata_package_id": None,
        "metadata_package_log": [],
        "metadata_has_unsubmitted_changes": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.shared_workspace_loaded:
        load_shared_workspace_state()
        st.session_state.shared_workspace_loaded = True


def ensure_dataset_loaded() -> None:
    if "source_df" not in st.session_state:
        set_source_dataframe(load_sample_dataframe(), f"Bundled ER sample • {SAMPLE_DATA_PATH.name}")


def effective_release_status(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> str:
    if has_unsubmitted_metadata_changes(metadata):
        return "Metadata review required"
    if st.session_state.synthetic_df is None:
        return "Not generated"
    if has_stale_generation(metadata, controls):
        return "Rerun required"
    if st.session_state.validation is None:
        return "Validation required"
    return st.session_state.release_status


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
        return "Compliance Officer"
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
                    "Control point": "Dates are jittered after metadata approval and before synthetic release.",
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
                    "Control point": "Normalize or group before sampling and keep under compliance review.",
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
            record["status"] = "Approved"
            st.session_state.current_metadata_package_id = record["package_id"]
            return record

    record = register_metadata_submission(metadata)
    record["approved_by"] = st.session_state.current_role
    record["approved_at"] = format_timestamp()
    record["status"] = "Approved"
    return record


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

    if review_package is not None:
        metadata_status = f"{review_package['package_id']} awaiting approval"
        metadata_owner = "Compliance Officer or Master"
        metadata_note = f"Submitted by {review_package['submitted_by']} at {review_package['submitted_at']}."
    elif active_package is not None and active_package["status"] == "Approved":
        metadata_status = f"{active_package['package_id']} approved"
        metadata_owner = "Data Analyst or Master"
        metadata_note = f"Approved by {active_package['approved_by']} at {active_package['approved_at']}."
    else:
        metadata_status = "Draft not submitted"
        metadata_owner = "Data Analyst or Master"
        metadata_note = "No metadata package is currently in the approval queue."

    if has_unsubmitted_metadata_changes(metadata):
        metadata_note += " A revised draft exists and still needs resubmission."

    if st.session_state.synthetic_df is None:
        generation_status = "Not generated"
        generation_owner = "Data Analyst or Master"
        generation_note = "Synthetic output has not been produced for the current package."
    elif has_stale_generation(metadata, controls):
        generation_status = "Stale"
        generation_owner = "Data Analyst or Master"
        generation_note = "Upstream controls changed after generation. A new run is required."
    else:
        generation_status = "Current"
        generation_owner = "Validation team"
        generation_note = "The synthetic run matches the latest approved package and generation controls."

    if st.session_state.export_requested_by is None:
        release_status = effective_release_status(metadata, controls)
        release_owner = "Data Analyst or Master"
        release_note = "Release review has not been requested yet."
    elif st.session_state.export_policy_approved_by is None:
        release_status = "Pending policy approval"
        release_owner = "Compliance Officer or Master"
        release_note = f"Requested by {st.session_state.export_requested_by}."
    elif st.session_state.export_approved_by is None:
        release_status = "Pending final authorization"
        release_owner = "Admin Office or Master"
        release_note = f"Policy approved by {st.session_state.export_policy_approved_by}."
    else:
        release_status = "Approved"
        release_owner = "Approved recipients"
        release_note = f"Final export authorized by {st.session_state.export_approved_by}."

    rows = [
        {
            "Work item": "Metadata package",
            "Status": metadata_status,
            "Current owner": metadata_owner,
            "Latest update": metadata_note,
        },
        {
            "Work item": "Synthetic run",
            "Status": generation_status,
            "Current owner": generation_owner,
            "Latest update": generation_note,
        },
        {
            "Work item": "Controlled release",
            "Status": release_status,
            "Current owner": release_owner,
            "Latest update": release_note,
        },
    ]
    return pd.DataFrame(rows)


def build_role_access_frame() -> pd.DataFrame:
    rows = []
    for role, config in ROLE_CONFIGS.items():
        permissions = config["permissions"]
        rows.append(
            {
                "Role": role,
                "Visible workflow": ", ".join(STEP_CONFIG[index]["title"] for index in visible_steps_for_role(role)),
                "Raw data access": "Yes" if "view_raw" in permissions else "No",
                "Metadata edits": "Yes" if "edit_metadata" in permissions else "No",
                "Metadata approval": "Yes" if "approve_metadata" in permissions else "No",
                "Synthetic generation": "Yes" if "generate" in permissions else "No",
                "Policy approval": "Yes" if "approve_release_policy" in permissions else "No",
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
        ("request_export", "Request governed release"),
        ("approve_release_policy", "Approve the policy checkpoint"),
        ("approve_export", "Authorize final export"),
        ("view_audit", "Review audit and approval history"),
    ],
}


ROLE_PRIORITY_NOTES: dict[str, str] = {
    "Clinician": "Focus on whether the synthetic dataset still reflects clinically plausible patterns without exposing raw patient detail.",
    "Data Analyst": "Focus on data quality, metadata choices, column behavior, and whether the synthetic output remains useful for analysis.",
    "Compliance Officer": "Focus on sensitive-field handling, approval checkpoints, privacy evidence, and whether release conditions are met.",
    "IT / Security": "Focus on raw-data boundaries, audit traceability, policy checkpoints, and whether exports stay locked until approvals complete.",
    "Admin Office": "Focus on the release gate, final authorization, and whether the package is ready for governed export.",
    "Master": "Focus on the full operating state across intake, approvals, generation, validation, and final release.",
}


STEP_EXPLANATION_NOTES: dict[int, str] = {
    0: "This step establishes where data enters, which fields look sensitive, and who is allowed to inspect source rows.",
    1: "This step determines whether source quality issues should be fixed before metadata and synthetic generation inherit them.",
    2: "This step turns profiled schema into a reviewed metadata package with ownership, handling rules, and approval checkpoints.",
    3: "This step uses the approved metadata package to produce synthetic output under a visible privacy-versus-fidelity posture.",
    4: "This step tests whether the synthetic output stays useful enough for analysis while release remains governed and reversible.",
}


def current_owner_checkpoint(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> tuple[str, str]:
    if not st.session_state.intake_confirmed:
        return "Current checkpoint", "Confirm intake controls for the dataset now in the workspace."
    if not st.session_state.hygiene_reviewed:
        return "Current checkpoint", "Finish the hygiene review and decide whether safe fixes should be applied."
    if has_unsubmitted_metadata_changes(metadata):
        return "Next required action", "A revised metadata draft exists. Data Analyst or Master should resubmit it for review."
    if st.session_state.metadata_status == "Draft":
        return "Next required action", "Data Analyst or Master should review field actions and submit the metadata package."
    if st.session_state.metadata_status == "In Review":
        review_package = current_review_package_record()
        if review_package is not None:
            return "Next required action", f"Compliance Officer or Master should review and approve package {review_package['package_id']}."
        return "Next required action", "Compliance Officer or Master should approve the metadata package."
    if st.session_state.synthetic_df is None or has_stale_generation(metadata, controls):
        return "Next required action", "Data Analyst or Master should generate a current synthetic dataset."
    if st.session_state.validation is None:
        return "Next required action", "Data Analyst or Master should run validation on the current synthetic output."
    if st.session_state.export_requested_by is None:
        return "Next required action", "Data Analyst or Master should request release review."
    if st.session_state.export_policy_approved_by is None:
        return "Next required action", "Compliance Officer or Master should approve the policy checkpoint."
    if st.session_state.export_approved_by is None:
        return "Next required action", "Admin Office or Master should authorize final export."
    return "Current state", "The current package is fully approved for governed export."


def build_role_status_lists(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, list[str] | str]:
    available: list[str] = []
    waiting: list[str] = []
    completed: list[str] = []
    visible_labels = [STEP_CONFIG[index]["title"] for index in visible_steps_for_role()]

    if st.session_state.intake_confirmed:
        completed.append("Intake controls confirmed.")
    else:
        if has_permission("upload"):
            available.append("Upload or replace the source dataset if needed.")
        available.append("Confirm intake controls for this dataset.")

    if st.session_state.hygiene_reviewed:
        completed.append("Hygiene review completed.")
    else:
        if has_permission("remediate"):
            available.append("Review hygiene findings and apply safe fixes.")
        else:
            waiting.append("Waiting on Data Analyst or Master to complete hygiene review.")

    if st.session_state.metadata_status == "Approved":
        completed.append(
            f"Metadata package approved{f' by {st.session_state.metadata_approved_by}' if st.session_state.metadata_approved_by else ''}."
        )
    elif st.session_state.metadata_status == "In Review":
        if has_permission("approve_metadata"):
            record = current_review_package_record() or active_metadata_package_record(metadata)
            available.append(
                f"Approve metadata package {record['package_id']}." if record else "Approve the metadata package."
            )
        else:
            record = current_review_package_record()
            waiting.append(
                f"Waiting on Compliance Officer or Master to approve package {record['package_id']}."
                if record
                else "Waiting on Compliance Officer or Master to approve metadata."
            )
    else:
        if has_permission("edit_metadata"):
            available.append("Adjust field actions or advanced metadata settings.")
        if has_permission("submit_metadata"):
            available.append("Submit the metadata package for review.")
        elif not has_permission("edit_metadata"):
            waiting.append("Waiting on Data Analyst or Master to submit metadata.")

    if has_unsubmitted_metadata_changes(metadata):
        if has_permission("edit_metadata") or has_permission("submit_metadata"):
            available.append("Review the revised metadata draft and resubmit the package.")
        waiting.append("Current working draft no longer matches the last submitted or approved package.")

    if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls):
        completed.append("Synthetic dataset generated.")
    elif st.session_state.metadata_status == "Approved" and not has_unsubmitted_metadata_changes(metadata):
        if has_permission("generate"):
            available.append("Generate a new synthetic dataset from the approved package.")
        else:
            waiting.append("Waiting on Data Analyst or Master to run synthetic generation.")

    if st.session_state.validation is not None and not has_stale_generation(metadata, controls):
        completed.append("Validation completed on the current synthetic run.")
    elif st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls):
        if has_permission("validate"):
            available.append("Run validation on the current synthetic output.")
        else:
            waiting.append("Waiting on Data Analyst or Master to run validation.")

    if st.session_state.validation is not None and not has_stale_generation(metadata, controls):
        if st.session_state.export_requested_by is None:
            if has_permission("request_export"):
                available.append("Request governed release review.")
            else:
                waiting.append("Waiting on Data Analyst or Master to request release review.")
        else:
            completed.append(f"Release review requested by {st.session_state.export_requested_by}.")

        if st.session_state.export_policy_approved_by is None:
            if has_permission("approve_release_policy"):
                available.append("Approve the policy checkpoint for release.")
            elif st.session_state.export_requested_by is not None:
                waiting.append("Waiting on Compliance Officer or Master to approve the policy checkpoint.")
        else:
            completed.append(f"Policy checkpoint approved by {st.session_state.export_policy_approved_by}.")

        if st.session_state.export_approved_by is None:
            if has_permission("approve_export"):
                available.append("Authorize final export.")
            elif st.session_state.export_policy_approved_by is not None:
                waiting.append("Waiting on Admin Office or Master to authorize final export.")
        else:
            completed.append(f"Final export authorized by {st.session_state.export_approved_by}.")

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
    package_owner = "Data Analyst or Master"

    if review_package is not None:
        package_status = f"{review_package['package_id']} · In Review"
        package_owner = "Compliance Officer or Master"
        package_detail = (
            f"Submitted by {review_package['submitted_by']} at {review_package['submitted_at']}. "
            "This is the package currently waiting for approval."
        )
    elif active_package is not None:
        package_status = f"{active_package['package_id']} · {active_package['status']}"
        package_owner = "Data Analyst or Master"
        package_detail = (
            f"Approved by {active_package['approved_by']} at {active_package['approved_at']}. "
            "This is the latest approved package in the workflow."
        )

    if has_unsubmitted_metadata_changes(metadata):
        package_detail += " A revised draft exists and still needs to be resubmitted."

    release_owner = "Data Analyst or Master"
    release_detail = effective_release_status(metadata, controls)
    if st.session_state.export_requested_by and st.session_state.export_policy_approved_by is None:
        release_owner = "Compliance Officer or Master"
    elif st.session_state.export_policy_approved_by and st.session_state.export_approved_by is None:
        release_owner = "Admin Office or Master"
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


def render_action_center(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    def task_rows(items: list[str], tone: str, limit: int) -> str:
        shown = items[:limit]
        rows = "".join(
            f'<div class="task-row"><span class="task-dot {tone}"></span><div class="task-text">{item}</div></div>'
            for item in shown
        )
        extra = ""
        if len(items) > limit:
            extra = f'<div class="task-more">+{len(items) - limit} more in details</div>'
        return rows + extra

    status_lists = build_role_status_lists(metadata, controls)
    st.markdown("**Action center**")
    top_cols = st.columns([0.9, 1.6], gap="large")
    top_cols[0].markdown(
        f"""
        <div class="action-hero-card">
            <div class="mini-label">Signed in as</div>
            <div class="action-role-name">{st.session_state.current_role}</div>
            <div class="action-summary">{current_role_summary()}</div>
            <div class="mini-label">Visible steps</div>
            <div class="steps-wrap">
                {"".join(f'<span class="step-pill-compact">{step}</span>' for step in status_lists['visible_steps'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_cols[1].markdown(
        f"""
        <div class="action-hero-card action-next-card">
            <div class="mini-label">{status_lists['label']}</div>
            <div class="action-next-title">{status_lists['next_action']}</div>
            <div class="action-subtle">Only the workflow steps relevant to this role are shown in navigation. Items blocked here are owned by another function.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_step = default_step_for_role(metadata, controls)
    action_cols = st.columns([0.92, 2.58], gap="large")
    if action_cols[0].button("Open my work area", type="primary", use_container_width=True):
        st.session_state.current_step = action_step
        st.rerun()
    action_cols[1].markdown(
        f"""
        <div class="note-card">
            <strong>Current work area</strong><br/>
            {STEP_CONFIG[action_step]['title']} is the main area for this role right now.
        </div>
        """,
        unsafe_allow_html=True,
    )

    bottom_cols = st.columns(3, gap="large")
    bottom_cols[0].markdown(
        f"""
        <div class="action-panel-card primary">
            <div class="task-card-title">You can do now</div>
            {task_rows(status_lists['available'], '', 3)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    bottom_cols[1].markdown(
        f"""
        <div class="action-panel-card">
            <div class="task-card-title">Waiting on</div>
            {task_rows(status_lists['waiting'], 'waiting', 3)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    bottom_cols[2].markdown(
        f"""
        <div class="action-panel-card">
            <div class="task-card-title">Completed</div>
            {task_rows(status_lists['completed'], 'completed', 3)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Work in progress**")
    wip_cols = st.columns(3, gap="large")
    for col, card in zip(wip_cols, build_work_in_progress_cards(metadata, controls)):
        chip_class = render_status_chip(card["status"])
        col.markdown(
            f"""
            <div class="state-card">
                <h4>{card['title']}</h4>
                <div class="state-value" style="font-size:1.18rem;">{card['value']}</div>
                <div class="{chip_class}">{card['status']}</div>
                <div class="state-text">{card['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        build_work_in_progress_frame(metadata, controls),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("More workflow detail", expanded=False):
        st.markdown(
            f"""
            <div class="note-card">
                <div class="mini-label">Full task lists</div>
                <strong>You can do now</strong>
                <ul class="bullet-list">
                    {''.join(f"<li>{item}</li>" for item in status_lists['available'])}
                </ul>
                <strong>Waiting on</strong>
                <ul class="bullet-list">
                    {''.join(f"<li>{item}</li>" for item in status_lists['waiting'])}
                </ul>
                <strong>Completed in this session</strong>
                <ul class="bullet-list">
                    {''.join(f"<li>{item}</li>" for item in status_lists['completed'])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_role_guidance_panel(st.session_state.current_step, compare_only=True)


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
    return [
        {
            "level": "Level 1",
            "stage": "Package prepared",
            "owner": "Data Analyst or Master",
            "status": "Complete" if st.session_state.metadata_status in {"Draft", "In Review", "Approved"} else "Pending",
            "kind": "Good" if st.session_state.metadata_status in {"Draft", "In Review", "Approved"} else "Warn",
            "detail": "Profiled source fields become editable metadata with owners, handling rules, and inclusion flags.",
        },
        {
            "level": "Level 2",
            "stage": "Review submitted",
            "owner": "Data Analyst or Master",
            "status": "Submitted" if submitted else "Pending",
            "kind": "Good" if submitted else "Warn",
            "detail": f"Submitted by: {submitted or 'Waiting for submission'}{f' at {submitted_at}' if submitted_at else ''}",
        },
        {
            "level": "Level 3",
            "stage": "Metadata approved",
            "owner": "Compliance Officer or Master",
            "status": "Approved" if approved else "Pending",
            "kind": "Good" if approved else "Warn",
            "detail": f"Approved by: {approved or 'Waiting for approval'}{f' at {approved_at}' if approved_at else ''}",
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
            "owner": "Data Analyst or Master",
            "status": "Verified" if validation_ready else "Waiting",
            "kind": "Good" if validation_ready else "Warn",
            "detail": "A current validation run is required before the release chain can start.",
        },
        {
            "level": "Level 2",
            "stage": "Release requested",
            "owner": "Data Analyst or Master",
            "status": "Requested" if requested_by else "Pending",
            "kind": "Good" if requested_by else "Warn",
            "detail": f"Requested by: {requested_by or 'Waiting for request'}",
        },
        {
            "level": "Level 3",
            "stage": "Policy approval",
            "owner": "Compliance Officer or Master",
            "status": "Approved" if policy_by else "Pending",
            "kind": "Good" if policy_by else "Warn",
            "detail": f"Approved by: {policy_by or 'Waiting for policy approval'}",
        },
        {
            "level": "Level 4",
            "stage": "Final export authorization",
            "owner": "Admin Office or Master",
            "status": "Approved" if final_by else "Locked",
            "kind": "Good" if final_by else "Warn",
            "detail": f"Authorized by: {final_by or 'Waiting for Admin Office or Master'}",
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
    dashboard = {item["label"]: item["value"] for item in build_validation_dashboard(metadata, controls)}
    overall = st.session_state.validation["overall_score"]
    privacy = st.session_state.validation["privacy_score"]
    utility = dashboard.get("Downstream utility", 0.0)
    fidelity = st.session_state.validation["fidelity_score"]
    wait_fields_present = any("wait" in item["column"].lower() and item["include"] for item in metadata)

    def status(value: str) -> str:
        return value

    return [
        {
            "Use case": "Operations dashboard prototyping",
            "Fit": status("Ready" if overall >= 75 else "Review"),
            "Why": "Key operational fields remain aligned enough to prototype service-line dashboards without moving raw encounter rows.",
            "Guardrail": "Use synthetic output only; do not back-infer individual patient events.",
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
            "Guardrail": "Re-run validation after metadata changes before sharing derived work.",
        },
        {
            "Use case": "Vendor sandbox or integration testing",
            "Fit": status("Ready" if privacy >= 85 else "Review"),
            "Why": "Synthetic records can exercise file layouts, API contracts, and workflows without releasing direct identifiers.",
            "Guardrail": "Exports remain blocked until policy approval and final authorization are completed.",
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
            "label": "Schema match",
            "value": schema_match,
            "detail": "Approved source fields retained in the synthetic output.",
        },
        {
            "label": "Distribution alignment",
            "value": validation["fidelity_score"],
            "detail": "Operational similarity between source and synthetic columns.",
        },
        {
            "label": "Privacy score",
            "value": validation["privacy_score"],
            "detail": "Overlap and identifier reuse checks under the current posture.",
        },
        {
            "label": "Statistical fidelity",
            "value": statistical_fidelity,
            "detail": "Blends schema coverage with detailed fidelity scoring.",
        },
        {
            "label": "Downstream utility",
            "value": downstream_utility,
            "detail": "Heuristic signal for analytics, testing, and sandbox usefulness.",
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
        return "Validation is not available yet."
    lines = [
        "Synthetic Data Validation Report",
        "",
        f"Dataset: {st.session_state.source_label}",
        f"Metadata status: {st.session_state.metadata_status}",
        f"Release status: {effective_release_status(metadata, controls)}",
        f"Overall score: {validation['overall_score']}",
        f"Fidelity score: {validation['fidelity_score']}",
        f"Privacy score: {validation['privacy_score']}",
        "",
        "Operational dashboard:",
    ]
    for metric in build_validation_dashboard(metadata, controls):
        lines.append(f"- {metric['label']}: {metric['value']} ({metric['detail']})")
    lines.extend(["", "Active generation controls:"])
    for row in build_generation_control_rows(controls):
        lines.append(f"- {row['Control']}: {row['Setting']} ({row['Effect']})")
    lines.extend(["", "Privacy checks:"])
    for _, row in validation["privacy_checks"].iterrows():
        lines.append(f"- {row['check']}: {row['result']} ({row['interpretation']})")
    lines.extend(
        [
            "",
            "Audit status:",
            f"- Metadata approved by: {st.session_state.metadata_approved_by or 'Not approved'}",
            f"- Export requested by: {st.session_state.export_requested_by or 'Not requested'}",
            f"- Policy approved by: {st.session_state.export_policy_approved_by or 'Not approved'}",
            f"- Final export authorized by: {st.session_state.export_approved_by or 'Not approved'}",
        ]
    )
    lines.extend(["", "Recommended use cases:"])
    for row in build_use_case_rows(metadata, controls):
        lines.append(f"- {row['Use case']}: {row['Fit']} ({row['Guardrail']})")
    return "\n".join(lines)


def sync_metadata_workflow_state(metadata: list[dict[str, Any]]) -> None:
    st.session_state.metadata_has_unsubmitted_changes = has_unsubmitted_metadata_changes(metadata)


def has_stale_generation(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> bool:
    if st.session_state.synthetic_df is None:
        return False
    return st.session_state.last_generation_signature != build_generation_signature(metadata, controls)


def intake_visible_to_raw_rows() -> str:
    return "Full preview" if has_permission("view_raw") else "Summary only"


def build_operating_state_cards(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    sensitive_fields = build_phi_detection_frame(st.session_state.profile, metadata)
    review_fields = sum(1 for item in metadata if metadata_owner(item) == "Compliance Officer" and item["include"])
    return [
        {
            "title": "Signed-in role",
            "value": st.session_state.current_role,
            "detail": current_role_summary(),
            "status": "Good",
        },
        {
            "title": "Raw data visibility",
            "value": intake_visible_to_raw_rows(),
            "detail": "Who can inspect row-level source data in the current session.",
            "status": "Good" if has_permission("view_raw") else "Warn",
        },
        {
            "title": "PHI / sensitive fields",
            "value": str(len(sensitive_fields)),
            "detail": "Fields currently under identifier, timing, geography, or free-text controls.",
            "status": "Warn" if len(sensitive_fields) else "Good",
        },
        {
            "title": "Metadata status",
            "value": metadata_display_status(metadata),
            "detail": f"{review_fields} fields require compliance ownership in the current package.",
            "status": "Good" if st.session_state.metadata_status == "Approved" and not has_unsubmitted_metadata_changes(metadata) else "Warn",
        },
        {
            "title": "Release gate",
            "value": effective_release_status(metadata, controls),
            "detail": "Synthetic export remains blocked until the current package passes validation and approval.",
            "status": "Good" if effective_release_status(metadata, controls) == "Approved" else "Warn",
        },
    ]


def step_status_labels(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[str]:
    return [
        "Complete" if st.session_state.intake_confirmed else "Action needed",
        "Complete" if st.session_state.hygiene_reviewed else "Action needed",
        metadata_display_status(metadata),
        "Complete" if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls) else "Waiting",
        "Complete" if st.session_state.validation is not None and not has_stale_generation(metadata, controls) else "Waiting",
    ]


def max_unlocked_step(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> int:
    if not st.session_state.intake_confirmed:
        return 0
    if not st.session_state.hygiene_reviewed:
        return 1
    if st.session_state.metadata_status != "Approved" or has_unsubmitted_metadata_changes(metadata):
        return 2
    if st.session_state.synthetic_df is None or has_stale_generation(metadata, controls):
        return 3
    return 4


def default_step_for_role(metadata: list[dict[str, Any]], controls: dict[str, Any], role: str | None = None) -> int:
    active_role = role or st.session_state.get("current_role")
    visible_steps = visible_steps_for_role(active_role)

    if not visible_steps:
        return 0

    if active_role in {"Clinician", "Admin Office"}:
        return 4
    if active_role == "Compliance Officer":
        if st.session_state.metadata_status != "Approved":
            return 2
        return 4
    if active_role == "IT / Security":
        if not st.session_state.intake_confirmed:
            return 0
        return 4

    unlocked = max_unlocked_step(metadata, controls)
    for step_index in visible_steps:
        if step_index >= unlocked:
            return step_index
    return visible_steps[-1]


def ensure_role_step_visibility(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    visible_steps = visible_steps_for_role()
    if st.session_state.current_step not in visible_steps:
        st.session_state.current_step = default_step_for_role(metadata, controls)


def render_status_chip(kind: str) -> str:
    if kind == "Good":
        return "status-good"
    if kind == "Bad":
        return "status-bad"
    return "status-warn"


def render_login_screen() -> None:
    st.markdown(
        """
        <div class="banner">
                <div class="banner-kicker">Southlake Health synthetic data workflow</div>
                <h1>Role-based access to a real synthetic data workflow.</h1>
                <p>
                Sign in as a clinician, analyst, compliance lead, security lead, admin office reviewer, or master operator.
                The system changes what you can see and what you are expected to do.
                </p>
            </div>
        """,
        unsafe_allow_html=True,
    )

    intro_col, form_col = st.columns([1.15, 0.95], gap="large")
    with intro_col:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-kicker">Why this version is different</div>
                <h3>It behaves like a workflow, not a brochure.</h3>
                <p>
                    This workspace makes control points explicit: where data enters, who can see raw rows, when de-identification occurs,
                    how metadata is generated, how synthetic output is produced, how fidelity is validated, and how teams roll back or revise a run.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        role_cols = st.columns(3)
        roles = list(ROLE_CONFIGS.items())
        for index, (role, config) in enumerate(roles):
            role_cols[index % 3].markdown(
                f"""
                <div class="role-card">
                    <strong>{role}</strong>
                    <p>{config['summary']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with form_col:
        with st.form("login_form", clear_on_submit=False):
            role = st.selectbox("Select role", options=list(ROLE_CONFIGS.keys()))
            password = st.text_input("Enter demo access code", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                if password == ROLE_CONFIGS[role]["password"]:
                    st.session_state.authenticated = True
                    st.session_state.current_role = role
                    record_audit_event("User signed in", f"Session started as {role}.", status="Logged")
                    ensure_dataset_loaded()
                    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
                    controls = st.session_state.controls
                    st.session_state.current_step = default_step_for_role(metadata, controls, role)
                    st.rerun()
                else:
                    st.error("Access code did not match the selected role.")

        st.caption("Demo password for every role: `test`")


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
        st.session_state.current_step = 0
        st.rerun()


def render_header(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="banner">
            <div class="banner-kicker">Healthcare synthetic data workspace</div>
            <h1>Five-step workflow for governed synthetic data operations.</h1>
            <p>
                This workflow is designed to show how the system actually runs: intake, hygiene review, metadata approval,
                synthetic generation, and fidelity validation with release controls.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_cols = st.columns(len(build_operating_state_cards(metadata, controls)))
    for col, card in zip(top_cols, build_operating_state_cards(metadata, controls)):
        chip_class = render_status_chip(card["status"])
        col.markdown(
            f"""
            <div class="state-card">
                <h4>{card['title']}</h4>
                <div class="state-value">{card['value']}</div>
                <div class="{chip_class}">{card['status']}</div>
                <div class="state-text">{card['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_step_navigation(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    visible_steps = visible_steps_for_role()
    visible_position = visible_steps.index(st.session_state.current_step) + 1
    st.progress(
        visible_position / len(visible_steps),
        text=f"Your view {visible_position} of {len(visible_steps)} • Workflow step {st.session_state.current_step + 1} of {len(STEP_CONFIG)}",
    )
    statuses = step_status_labels(metadata, controls)
    nav_cols = st.columns(len(visible_steps))
    for col, index in zip(nav_cols, visible_steps):
        step = STEP_CONFIG[index]
        card_class = "step-card active" if index == st.session_state.current_step else "step-card"
        with col:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="step-number">{index + 1}</div>
                    <div class="step-title">{step['title']}</div>
                    <div class="step-description">{step['description']}</div>
                    <div class="pill">{statuses[index]}</div>
                    <div class="step-action-note">Open this step to continue the workflow.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Current step" if index == st.session_state.current_step else "Open step",
                key=f"step_nav_{index}",
                use_container_width=True,
                type="primary" if index == st.session_state.current_step else "secondary",
                on_click=lambda i=index: st.session_state.update(current_step=i),
            )


def render_section_header(step_index: int, checkpoint: str) -> None:
    step = STEP_CONFIG[step_index]
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">Step {step_index + 1} · {step['title']}</div>
            <h3>{step['description']}</h3>
            <p><strong>Current checkpoint:</strong> {checkpoint}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_role_restriction(message: str) -> None:
    st.info(message)


def render_step_one(metadata: list[dict[str, Any]]) -> None:
    render_section_header(0, "Confirm how data enters the workspace and who can see source rows.")

    info_cols = st.columns([1.1, 0.9], gap="large")
    with info_cols[0]:
        st.markdown(
            """
            <div class="note-card">
                <strong>What happens at intake</strong><br/>
                Source data enters a local governed workspace. PHI-like fields are identified immediately, access is limited by role,
                and de-identification rules are not applied until the metadata package is reviewed and approved.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if has_permission("upload"):
            uploaded_file = st.file_uploader(
                "Upload a healthcare CSV",
                type=["csv"],
                help="The bundled emergency department sample is already loaded for a fast walkthrough.",
            )
            if uploaded_file is not None:
                current_signature = (uploaded_file.name, uploaded_file.size)
                if st.session_state.get("uploaded_signature") != current_signature:
                    set_source_dataframe(load_csv_bytes(uploaded_file.getvalue()), f"Uploaded dataset • {uploaded_file.name}")
                    st.session_state.uploaded_signature = current_signature
                    st.rerun()
            action_cols = st.columns(2)
            if action_cols[0].button("Reload bundled sample", use_container_width=True):
                set_source_dataframe(load_sample_dataframe(), f"Bundled ER sample • {SAMPLE_DATA_PATH.name}")
                st.session_state.uploaded_signature = None
                st.rerun()
            action_cols[1].markdown(
                f"""
                <div class="note-card">
                    <strong>Current dataset</strong><br/>
                    {st.session_state.source_label}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_role_restriction("This role can review intake controls and schema summaries, but cannot upload or replace the source dataset.")

        confirm_label = "Confirm intake controls" if not st.session_state.intake_confirmed else "Intake confirmed"
        if st.button(confirm_label, type="primary", use_container_width=True, disabled=st.session_state.intake_confirmed):
            st.session_state.intake_confirmed = True
            record_audit_event("Intake confirmed", "Source workspace controls acknowledged.", status="Completed")
            st.session_state.current_step = 1
            st.rerun()

    with info_cols[1]:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Rows loaded", st.session_state.profile["summary"]["rows"])
        metric_cols[1].metric("Columns", st.session_state.profile["summary"]["columns"])
        metric_cols[0].metric("Sensitive fields", len(build_phi_detection_frame(st.session_state.profile, metadata)))
        metric_cols[1].metric("Raw data visibility", intake_visible_to_raw_rows())
        st.dataframe(build_role_access_frame(), use_container_width=True, hide_index=True)

    preview_tab, phi_tab = st.tabs(["Source workspace view", "PHI / sensitive field detection"])
    with preview_tab:
        if has_permission("view_raw"):
            st.dataframe(st.session_state.source_df.head(12), use_container_width=True, hide_index=True)
        else:
            summary_rows = [
                {
                    "Field": column,
                    "Role": details["semantic_role"],
                    "Missing %": details["missing_pct"],
                    "Unique values": details["unique_count"],
                }
                for column, details in st.session_state.profile["columns"].items()
            ]
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.caption("This role sees schema-level and control-level detail only, not row-level raw records.")
    with phi_tab:
        st.dataframe(build_phi_detection_frame(st.session_state.profile, metadata), use_container_width=True, hide_index=True)


def render_step_two() -> None:
    render_section_header(1, "Review source quality before metadata is finalized.")

    hygiene = st.session_state.hygiene
    missingness_frame = build_missingness_strategy_frame(st.session_state.profile)
    missingness_chart = missingness_frame[missingness_frame["Field"] != "No incomplete field"][["Field", "Missing %"]]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Quality score", hygiene["quality_score"])
    metric_cols[1].metric("High severity", hygiene["severity_counts"]["High"])
    metric_cols[2].metric("Medium severity", hygiene["severity_counts"]["Medium"])
    metric_cols[3].metric("Duplicate rows", st.session_state.profile["summary"]["duplicate_rows"])

    issues_frame = pd.DataFrame(
        [
            {
                "Field": issue["column"],
                "Concern": issue["concern"],
                "Issue": issue["finding"],
                "Severity": issue["severity"],
                "Recommended action": issue["recommendation"],
            }
            for issue in hygiene["issues"]
        ]
    )

    review_col, action_col = st.columns([1.25, 0.95], gap="large")
    with review_col:
        st.markdown(
            """
            <div class="note-card">
                <strong>Hospital risk questions answered here</strong><br/>
                Is the source data too dirty to trust? Are identifiers still present? Are duplicates or extreme values likely to distort synthetic output?
                The hygiene scan surfaces those risks before any metadata is approved.
            </div>
            """,
            unsafe_allow_html=True,
        )
        scan_tabs = st.tabs(["Findings", "Source preview", "Missingness strategy"])
        with scan_tabs[0]:
            if issues_frame.empty:
                st.success("No hygiene issues were detected in the current dataset.")
            else:
                st.dataframe(issues_frame, use_container_width=True, hide_index=True)
        with scan_tabs[1]:
            preview_cols = st.columns(2)
            preview_cols[0].metric("Rows in source data", len(st.session_state.source_df))
            preview_cols[1].metric("Columns in source data", len(st.session_state.source_df.columns))
            st.dataframe(st.session_state.source_df.head(15), use_container_width=True, hide_index=True)
        with scan_tabs[2]:
            if not missingness_chart.empty:
                chart_frame = missingness_chart.set_index("Field")
                st.bar_chart(chart_frame, use_container_width=True)
            st.dataframe(missingness_frame, use_container_width=True, hide_index=True)

    with action_col:
        st.markdown(
                """
                <div class="note-card">
                    <strong>Available actions</strong><br/>
                    Data Analysts and Master can remediate safe hygiene issues here. Other roles can review the findings and confirm they were assessed.
                </div>
                """,
                unsafe_allow_html=True,
            )
        remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True, disabled=not has_permission("remediate"))
        normalize_categories = st.checkbox("Normalize category labels", value=True, disabled=not has_permission("remediate"))
        fix_negative_values = st.checkbox("Convert invalid negative values to missing", value=True, disabled=not has_permission("remediate"))

        options = {
            "remove_duplicates": remove_duplicates,
            "normalize_categories": normalize_categories,
            "fix_negative_values": fix_negative_values,
        }
        preview_df, preview_actions = apply_hygiene_fixes(st.session_state.source_df, options)
        st.dataframe(pd.DataFrame(preview_actions), use_container_width=True, hide_index=True)
        preview_cols = st.columns(2)
        preview_cols[0].metric("Rows before", len(st.session_state.source_df))
        preview_cols[1].metric("Rows after", len(preview_df))
        st.markdown(
            """
            <div class="note-card">
                <strong>Missing data strategy</strong><br/>
                Keep missingness visible by default unless there is a documented business rule. The scan recommends where to preserve gaps,
                where to escalate review, and where cleanup is safe before metadata approval.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if has_permission("remediate"):
            if st.button("Apply selected fixes", type="primary", use_container_width=True):
                cleaned_df, actions = apply_hygiene_fixes(st.session_state.source_df, options)
                set_source_dataframe(cleaned_df, f"{st.session_state.source_label} • remediated")
                st.session_state.last_cleaning_actions = actions
                st.session_state.intake_confirmed = True
                st.session_state.hygiene_reviewed = True
                record_audit_event("Hygiene fixes applied", "; ".join(item["effect"] for item in actions), status="Completed")
                st.session_state.current_step = 2
                st.rerun()
        else:
            render_role_restriction("This role cannot modify the source dataset. Review the scan results and wait for Data Analyst or Master action.")

        if st.button(
            "Mark hygiene review complete",
            use_container_width=True,
            disabled=st.session_state.hygiene_reviewed,
        ):
            st.session_state.hygiene_reviewed = True
            record_audit_event("Hygiene review completed", "Source quality and risk findings were acknowledged.", status="Completed")
            st.session_state.current_step = 2
            st.rerun()

    if st.session_state.last_cleaning_actions:
        st.success("Latest remediation applied")
        st.dataframe(pd.DataFrame(st.session_state.last_cleaning_actions), use_container_width=True, hide_index=True)


def render_step_three() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    render_section_header(2, "Approve the metadata package that controls de-identification and generation.")

    controls = st.session_state.controls.copy()
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    sync_metadata_workflow_state(metadata)
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    metadata_frame = build_metadata_review_frame(metadata)
    active_package = active_metadata_package_record(metadata)
    review_package = current_review_package_record()
    metadata_status_label = metadata_display_status(metadata)
    metadata_edit_locked = (
        has_permission("edit_metadata")
        and st.session_state.metadata_status in {"In Review", "Approved"}
        and not has_unsubmitted_metadata_changes(metadata)
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Metadata status", metadata_status_label)
    summary_cols[1].metric(
        "Package in process",
        review_package["package_id"] if review_package else (active_package["package_id"] if active_package else "None"),
    )
    summary_cols[2].metric("Restricted or sensitive fields", int(metadata_frame["Sensitivity"].isin(["Restricted", "Sensitive"]).sum()))
    summary_cols[3].metric("Included fields", int(st.session_state.metadata_editor_df["include"].sum()))

    if review_package is not None and has_permission("approve_metadata"):
        st.warning(
            f"Package {review_package['package_id']} from {review_package['submitted_by']} submitted at {review_package['submitted_at']} is waiting for your approval. Review the package record below, then approve the exact package listed there."
        )
    elif st.session_state.metadata_status == "In Review" and active_package is not None:
        st.warning(
            f"Package {active_package['package_id']} is in review. Submitted by {active_package['submitted_by']} at {active_package['submitted_at']}. Waiting for Compliance Officer or Master approval."
        )
    elif st.session_state.metadata_status == "Approved" and active_package is not None:
        st.success(
            f"Package {active_package['package_id']} was approved by {active_package['approved_by']} at {active_package['approved_at']}. Synthetic generation is now unlocked."
        )
    elif active_package is not None and st.session_state.metadata_status == "Draft":
        st.info(
            f"The current metadata has changed since package {active_package['package_id']} was submitted. Review the package record below to see what was previously submitted and approved."
        )

    if metadata_edit_locked and active_package is not None:
        lock_cols = st.columns([1.7, 0.9], gap="large")
        if st.session_state.metadata_status == "Approved":
            lock_cols[0].info(
                f"Package {active_package['package_id']} is the approved package in use. Editing is locked so every role sees the same submitted package. Start a revised draft only when you want to change what will be resubmitted."
            )
            if lock_cols[1].button("Start revised draft", use_container_width=True):
                st.session_state.metadata_status = "Draft"
                record_audit_event(
                    "Metadata draft reopened",
                    f"A revised working draft was opened from package {active_package['package_id']}.",
                    status="Updated",
                )
                rerun_with_persist()
        else:
            lock_cols[0].info(
                f"Package {active_package['package_id']} is currently in review. Editing is paused so the submitted package stays stable for Compliance Officer review."
            )

    st.markdown(
        """
        <div class="note-card">
            <strong>How metadata is generated</strong><br/>
            The system profiles each source column, assigns a semantic type, flags sensitive handling needs, and proposes a generation rule.
            Human reviewers then decide what stays included, what gets generalized, and when the package is approved for synthetic generation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Approval hierarchy for metadata**")
    render_approval_hierarchy(build_metadata_approval_rows(), "metadata_approval")

    record_cols = st.columns(4)
    if active_package is not None:
        record_cols[0].metric("Active package", active_package["package_id"])
        record_cols[1].metric("Submitted by", active_package["submitted_by"])
        record_cols[2].metric("Approved by", active_package["approved_by"] or "Not approved")
        record_cols[3].metric("Targeted actions", active_package["summary"]["targeted_actions"])
    else:
        record_cols[0].metric("Active package", "None")
        record_cols[1].metric("Submitted by", "Not submitted")
        record_cols[2].metric("Approved by", "Not approved")
        record_cols[3].metric("Targeted actions", int(st.session_state.metadata_editor_df["control_action"].ne("Preserve").sum()))

    review_tab, record_tab, quick_tab, edit_tab = st.tabs(["Metadata review", "Package record", "Targeted field actions", "Advanced editor"])
    with review_tab:
        st.dataframe(metadata_frame, use_container_width=True, hide_index=True)
    with record_tab:
        if active_package is None and not st.session_state.metadata_package_log:
            st.info("No metadata package has been submitted yet.")
        else:
            if active_package is not None:
                package_summary = active_package["summary"]
                package_summary_cols = st.columns(5)
                package_summary_cols[0].metric("Package", active_package["package_id"])
                package_summary_cols[1].metric("Included fields", package_summary["included_fields"])
                package_summary_cols[2].metric("Restricted fields", package_summary["restricted_fields"])
                package_summary_cols[3].metric("Sensitive fields", package_summary["sensitive_fields"])
                package_summary_cols[4].metric("Targeted actions", package_summary["targeted_actions"])

                st.markdown(
                    f"""
                    <div class="note-card">
                        <strong>Submission record</strong><br/>
                        Submitted by {active_package['submitted_by']} at {active_package['submitted_at']}.
                        {'Approved by ' + active_package['approved_by'] + ' at ' + active_package['approved_at'] + '.' if active_package['approved_by'] else 'This package is still waiting for approval.'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    build_metadata_review_frame(active_package["snapshot"]),
                    use_container_width=True,
                    hide_index=True,
                )

            if st.session_state.metadata_package_log:
                st.markdown("**Recent package history**")
                st.dataframe(build_metadata_package_log_frame(), use_container_width=True, hide_index=True)
    with quick_tab:
        st.markdown(
            """
            <div class="note-card">
                <strong>Targeted field actions</strong><br/>
                Use this layer for quick metadata manipulation without rebuilding the whole package. It is especially useful for PHI and quasi-identifiers,
                but it also supports numeric clipping and category grouping for operational fields.
            </div>
            """,
            unsafe_allow_html=True,
        )
        quick_summary_cols = st.columns(4)
        quick_summary_cols[0].metric("Restricted fields", int(metadata_frame["Sensitivity"].eq("Restricted").sum()))
        quick_summary_cols[1].metric("Sensitive fields", int(metadata_frame["Sensitivity"].eq("Sensitive").sum()))
        quick_summary_cols[2].metric("Excluded fields", int(st.session_state.metadata_editor_df["include"].eq(False).sum()))
        quick_summary_cols[3].metric(
            "Fields with targeted actions",
            int(st.session_state.metadata_editor_df["control_action"].ne("Preserve").sum()),
        )

        if has_permission("edit_metadata") and not metadata_edit_locked:
            bulk_cols = st.columns(3)
            if bulk_cols[0].button("Tighten PHI controls", use_container_width=True):
                apply_bulk_metadata_profile("tighten_phi")
                rerun_with_persist()
            if bulk_cols[1].button("Preserve analytics detail", use_container_width=True):
                apply_bulk_metadata_profile("preserve_analytics")
                rerun_with_persist()
            if bulk_cols[2].button("Reset metadata defaults", use_container_width=True):
                apply_bulk_metadata_profile("reset_defaults")
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

            latest_metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
            action_counts = pd.Series([item["control_action"] for item in latest_metadata]).value_counts()
            st.dataframe(
                pd.DataFrame({"Field action": action_counts.index, "Fields": action_counts.values}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            if metadata_edit_locked:
                st.info("The working draft is locked while the current package is in review or already approved. Review the submitted package record or start a revised draft first.")
            else:
                render_role_restriction("This role can review targeted field actions, but cannot change them.")
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
                    "control_action": st.column_config.SelectboxColumn(
                        "Field action",
                        options=ALL_CONTROL_ACTIONS,
                    ),
                    "nullable": st.column_config.CheckboxColumn("Nullable"),
                    "notes": st.column_config.TextColumn("Notes", width="large"),
                },
            )
            st.session_state.metadata_editor_df = normalize_metadata_frame(editor)
        else:
            if metadata_edit_locked:
                st.info("The submitted package is frozen for review. Open a revised draft when you want to change field rules.")
            else:
                render_role_restriction("This role can review metadata controls and ownership, but cannot edit field rules.")
            st.dataframe(st.session_state.metadata_editor_df, use_container_width=True, hide_index=True)

    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    sync_metadata_workflow_state(metadata)

    action_cols = st.columns(3)
    if has_permission("submit_metadata"):
        submit_label = "Submit revised metadata for review" if active_package is not None else "Submit metadata for review"
        if action_cols[0].button(submit_label, type="primary", use_container_width=True, disabled=metadata_edit_locked):
            st.session_state.metadata_status = "In Review"
            st.session_state.metadata_submitted_by = st.session_state.current_role
            st.session_state.metadata_submitted_at = format_timestamp()
            st.session_state.metadata_approved_by = None
            st.session_state.metadata_approved_at = None
            st.session_state.last_reviewed_metadata_signature = build_metadata_signature(metadata)
            submitted_record = register_metadata_submission(metadata)
            st.session_state.release_status = "Not ready"
            record_audit_event(
                "Metadata submitted",
                f"{submitted_record['package_id']} submitted by {st.session_state.current_role} with {submitted_record['summary']['included_fields']} included fields.",
                status="Submitted",
            )
            rerun_with_persist()
    else:
        action_cols[0].button("Submit metadata for review", use_container_width=True, disabled=True)

    if has_permission("approve_metadata"):
        approve_label = f"Approve package {review_package['package_id']}" if review_package else "Approve metadata package"
        if action_cols[1].button(
            approve_label,
            type="primary",
            use_container_width=True,
            disabled=review_package is None,
        ):
            st.session_state.metadata_status = "Approved"
            st.session_state.metadata_approved_by = st.session_state.current_role
            st.session_state.metadata_approved_at = format_timestamp()
            st.session_state.last_reviewed_metadata_signature = build_metadata_signature(metadata)
            approved_record = register_metadata_approval(metadata)
            st.session_state.release_status = "Not ready"
            record_audit_event(
                "Metadata approved",
                f"{approved_record['package_id']} approved by {st.session_state.current_role}.",
                status="Approved",
            )
            st.session_state.current_step = 3
            rerun_with_persist()
    else:
        action_cols[1].button("Approve metadata package", use_container_width=True, disabled=True)

    action_cols[2].markdown(
        f"""
        <div class="note-card">
            <strong>Current owner flow</strong><br/>
            Submitted by: {st.session_state.metadata_submitted_by or "Not submitted"} {f"at {st.session_state.metadata_submitted_at}" if st.session_state.metadata_submitted_at else ""}<br/>
            Approved by: {st.session_state.metadata_approved_by or "Not approved"} {f"at {st.session_state.metadata_approved_at}" if st.session_state.metadata_approved_at else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    return metadata, controls


def render_step_four(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    render_section_header(3, "Generate synthetic output from the approved metadata package.")

    if st.session_state.metadata_status != "Approved":
        st.warning("Synthetic generation is locked until the metadata package is approved.")
        return metadata, controls

    eligible_distribution_columns = [
        item["column"] for item in metadata if item["include"] and item["data_type"] != "identifier"
    ]
    controls = sync_generation_preset_label(controls)
    controls["locked_columns"] = [column for column in controls.get("locked_columns", []) if column in eligible_distribution_columns]
    posture_label = "Balanced" if controls["fidelity_priority"] < 70 else "Higher fidelity"
    locked_preview = ", ".join(controls.get("locked_columns", [])[:3]) or "None selected"

    overview_cols = st.columns([1.15, 0.85, 0.85], gap="large")
    with overview_cols[0]:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-kicker">Generation workspace</div>
                <h3>Configure how synthetic data will be produced.</h3>
                <p>
                    This step uses the approved metadata package only. If metadata or generation settings change later,
                    the current synthetic output is marked stale and must be rerun.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_cols[1]:
        st.markdown(
            f"""
            <div class="state-card">
                <h4>Current output plan</h4>
                <div class="state-value">{controls['synthetic_rows']}</div>
                <div class="state-text">Target synthetic rows with a {posture_label.lower()} privacy posture.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_cols[2]:
        st.markdown(
            f"""
            <div class="state-card">
                <h4>Policy snapshot</h4>
                <div class="state-value">{len(controls.get('locked_columns', []))}</div>
                <div class="state-text">
                    Locked distributions, missingness set to {controls.get('missingness_pattern', 'Preserve source pattern').lower()},
                    and outlier handling set to {controls.get('outlier_strategy', 'Preserve tails').lower()}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    generate_clicked = False

    if has_permission("generate"):
        top_controls = st.columns(2, gap="large")
        with top_controls[0]:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="control-card-kicker">Core generation</div>
                    <div class="control-card-title">Output size and privacy posture</div>
                    <div class="control-card-text">Set the row count and the base privacy-versus-realism balance for this generation run.</div>
                    """,
                    unsafe_allow_html=True,
                )
                preset_options = list(GENERATION_PRESETS.keys()) + ["Custom"]
                current_preset = str(controls.get("generation_preset", "Balanced"))
                if current_preset not in preset_options:
                    current_preset = "Custom"
                selected_preset = st.selectbox(
                    "Generation preset",
                    options=preset_options,
                    index=preset_options.index(current_preset),
                    help="Use a preset to quickly set a starting point for privacy, structure, and noise settings.",
                )
                if selected_preset != current_preset:
                    if selected_preset == "Custom":
                        controls["generation_preset"] = "Custom"
                    else:
                        controls = apply_generation_preset(controls, selected_preset)
                controls["fidelity_priority"] = st.slider(
                    "Privacy versus fidelity",
                    min_value=0,
                    max_value=100,
                    value=int(controls["fidelity_priority"]),
                    help="Lower values add more smoothing; higher values preserve source behavior more closely.",
                )
                controls["synthetic_rows"] = int(
                    st.number_input(
                        "Synthetic row count",
                        min_value=10,
                        value=int(controls["synthetic_rows"]),
                        step=5,
                    )
                )
                st.markdown(
                    "<div class='control-compact-note'>Use this block to set the overall size and posture of the synthetic package before tuning the detailed controls. Synthetic row count is open-ended here, so you are not capped by the source dataset size.</div>",
                    unsafe_allow_html=True,
                )
        with top_controls[1]:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="control-card-kicker">Distribution & structure</div>
                    <div class="control-card-title">Preserve patterns that matter for analysis</div>
                    <div class="control-card-text">Lock key distributions and keep more structural relationships when downstream testing needs them.</div>
                    """,
                    unsafe_allow_html=True,
                )
                controls["locked_columns"] = st.multiselect(
                    "Lock key distributions",
                    options=eligible_distribution_columns,
                    default=controls["locked_columns"],
                    help="Selected columns stay closer to source behavior during generation.",
                )
                controls["correlation_preservation"] = st.slider(
                    "Correlation preservation",
                    min_value=0,
                    max_value=100,
                    value=int(controls["correlation_preservation"]),
                    help="Higher values preserve more row-to-row structure across columns.",
                )
                controls["rare_case_retention"] = st.slider(
                    "Rare case retention",
                    min_value=0,
                    max_value=100,
                    value=int(controls["rare_case_retention"]),
                    help="Higher values give more weight to rare categories and atypical rows.",
                )
                st.markdown(
                    f"<div class='control-compact-note'>Currently locking: {locked_preview}.</div>",
                    unsafe_allow_html=True,
                )

        bottom_controls = st.columns([1.05, 0.95], gap="large")
        with bottom_controls[0]:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="control-card-kicker">Data quality & noise</div>
                    <div class="control-card-title">Control gaps, extremes, and variability</div>
                    <div class="control-card-text">Adjust how much synthetic noise is added, what happens to missing values, and how outliers are handled.</div>
                    """,
                    unsafe_allow_html=True,
                )
                controls["noise_level"] = st.slider(
                    "Noise level",
                    min_value=0,
                    max_value=100,
                    value=int(controls["noise_level"]),
                    help="Controls how much random variation is injected during generation.",
                )
                missingness_options = ["Preserve source pattern", "Reduce missingness", "Fill gaps"]
                controls["missingness_pattern"] = st.selectbox(
                    "Missingness pattern",
                    options=missingness_options,
                    index=missingness_options.index(controls["missingness_pattern"]),
                )
                outlier_options = ["Preserve tails", "Clip extremes", "Smooth tails"]
                controls["outlier_strategy"] = st.selectbox(
                    "Outlier strategy",
                    options=outlier_options,
                    index=outlier_options.index(controls["outlier_strategy"]),
                )
                st.markdown(
                    "<div class='control-compact-note'>These controls are useful when you want a cleaner testing dataset without changing the approved metadata package.</div>",
                    unsafe_allow_html=True,
                )
        with bottom_controls[1]:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="control-card-kicker">Quick metadata preset</div>
                    <div class="control-card-title">Adjust sensitive-field handling without opening the full editor</div>
                    <div class="control-card-text">Apply a metadata preset when you want to tighten privacy controls or restore a more analysis-friendly package.</div>
                    """,
                    unsafe_allow_html=True,
                )
                quick_metadata_label = st.selectbox(
                    "Metadata quick action",
                    options=list(METADATA_QUICK_PRESETS.keys()),
                    index=0,
                    help="Applying a metadata preset sends the package back to Step 3 for review and approval.",
                )
                if quick_metadata_label != "Use approved metadata":
                    if st.button("Apply metadata preset and return to review", use_container_width=True):
                        apply_bulk_metadata_profile(METADATA_QUICK_PRESETS[quick_metadata_label])
                        st.session_state.metadata_status = "Draft"
                        st.session_state.metadata_submitted_by = None
                        st.session_state.metadata_submitted_at = None
                        st.session_state.metadata_approved_by = None
                        st.session_state.metadata_approved_at = None
                        st.session_state.last_reviewed_metadata_signature = None
                        st.session_state.current_metadata_package_id = None
                        clear_generation_outputs()
                        st.session_state.current_step = 2
                        record_audit_event(
                            "Metadata preset applied",
                            f"{quick_metadata_label} was applied in synthetic generation and sent back for review.",
                            status="Updated",
                        )
                        st.rerun()
                    st.markdown(
                        "<div class='control-compact-note'>This is useful when you need a quick privacy tightening step without editing every field manually.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='control-compact-note'>The current approved metadata package will be used as-is unless you choose a quick preset.</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div class="generate-action-card">
                        <div class="control-card-kicker">Generation action</div>
                        <div class="generate-action-title">Ready to create a synthetic dataset</div>
                        <div class="generate-action-text">
                            Current plan: {controls['synthetic_rows']} rows, {len(controls.get('locked_columns', []))} locked distributions,
                            correlation preservation at {controls.get('correlation_preservation', 0)}/100, and noise at {controls.get('noise_level', 0)}/100.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                info_cols = st.columns(2)
                info_cols[0].metric("Posture", posture_label)
                info_cols[1].metric("Rare case retention", f"{controls.get('rare_case_retention', 0)} / 100")
                generate_clicked = st.button("Generate synthetic dataset", type="primary", use_container_width=True)
                st.markdown(
                    "<div class='control-compact-note'>Any metadata or control change after this run will mark the output stale until you regenerate.</div>",
                    unsafe_allow_html=True,
                )

        controls = sync_generation_preset_label(controls)
        st.session_state.controls = controls
    else:
        render_role_restriction("This role can review the generation configuration, but cannot create or replace the synthetic dataset.")
        read_cols = st.columns(3, gap="large")
        read_rows = [
            (str(controls.get("generation_preset", "Balanced")), posture_label, "Current generation preset and privacy posture."),
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
            progress = st.progress(0, text="Checking approved metadata package...")
            time.sleep(0.12)
            progress.progress(20, text="Preparing field-level generation rules...")
            time.sleep(0.12)
            progress.progress(48, text="Applying privacy and fidelity controls...")
            time.sleep(0.12)
            synthetic_df, generation_summary = generate_synthetic_data(st.session_state.source_df, metadata, controls)
            progress.progress(82, text="Writing synthetic records and summaries...")
            time.sleep(0.12)
            progress.progress(100, text="Synthetic generation completed.")
            time.sleep(0.08)
            st.session_state.synthetic_df = synthetic_df
            st.session_state.generation_summary = generation_summary
            st.session_state.validation = None
            st.session_state.last_generation_signature = build_generation_signature(metadata, controls)
            st.session_state.release_status = "Validation required"
            record_audit_event(
                "Synthetic dataset generated",
                f"{generation_summary['rows_generated']} rows generated under {generation_summary['noise_mode'].lower()} settings.",
                status="Completed",
            )
            st.rerun()

    if st.session_state.synthetic_df is None:
        st.info("No synthetic dataset has been generated yet.")
        return metadata, controls

    if has_stale_generation(metadata, controls):
        st.warning("Metadata or generation controls changed after the last run. Re-generate before relying on the current synthetic output.")

    metric_cols = st.columns(4)
    summary = st.session_state.generation_summary
    metric_cols[0].metric("Rows generated", summary["rows_generated"])
    metric_cols[1].metric("Fields included", summary["columns_generated"])
    metric_cols[2].metric("Posture", summary["noise_mode"])
    metric_cols[3].metric("Excluded fields", len(summary["excluded_columns"]))

    st.markdown("**Generation run bars**")
    run_cols = st.columns(3)
    included_fields = max(int(st.session_state.metadata_editor_df["include"].sum()), 1)
    field_coverage = int(round(summary["columns_generated"] / included_fields * 100))
    row_scale = int(round(summary["rows_generated"] / max(int(controls["synthetic_rows"]), 1) * 100))
    run_cols[0].caption(f"Records generated: {summary['rows_generated']} of target {controls['synthetic_rows']}")
    run_cols[0].progress(min(row_scale, 100))
    run_cols[1].caption(f"Approved field coverage: {summary['columns_generated']} included fields in output")
    run_cols[1].progress(min(field_coverage, 100))
    run_cols[2].caption(f"Fidelity setting: {controls['fidelity_priority']} / 100")
    run_cols[2].progress(int(controls["fidelity_priority"]))

    preview_tab, note_tab = st.tabs(["Synthetic preview", "Generation notes"])
    with preview_tab:
        st.dataframe(st.session_state.synthetic_df.head(12), use_container_width=True, hide_index=True)
    with note_tab:
        st.markdown(
            f"""
            <div class="note-card">
                <strong>Generation note</strong><br/>
                This run used the {summary.get('generation_preset', 'Balanced').lower()} preset with {summary['noise_mode'].lower()} behavior,
                {summary['missingness_pattern'].lower()} missingness handling, and {summary['outlier_strategy'].lower()} outlier handling.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(build_generation_control_rows(controls)),
            use_container_width=True,
            hide_index=True,
        )
        if summary["excluded_columns"]:
            st.dataframe(pd.DataFrame({"Excluded field": summary["excluded_columns"]}), use_container_width=True, hide_index=True)

    rollback_cols = st.columns(2)
    if has_permission("rollback"):
        if rollback_cols[0].button("Rollback synthetic run", use_container_width=True):
            clear_generation_outputs()
            record_audit_event("Synthetic run discarded", "Current synthetic output was discarded to revisit upstream controls.", status="Rolled back")
            st.rerun()
        if rollback_cols[1].button("Return to metadata review", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()

    return metadata, controls


def render_step_five(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    render_section_header(4, "Validate fidelity, compare outputs, and control release.")

    if st.session_state.synthetic_df is None:
        st.warning("Validation is locked until a synthetic dataset has been generated.")
        return

    if has_stale_generation(metadata, controls):
        st.warning("The current synthetic output is stale because upstream controls changed. Re-run Step 4 before validation or export.")

    if has_permission("validate"):
        if st.button("Run validation", type="primary", use_container_width=True):
            validation = validate_synthetic_data(st.session_state.source_df, st.session_state.synthetic_df, metadata, controls)
            st.session_state.validation = validation
            st.session_state.release_status = "Ready for request"
            st.session_state.export_requested_by = None
            st.session_state.export_policy_approved_by = None
            st.session_state.export_approved_by = None
            record_audit_event(
                "Validation completed",
                f"Overall {validation['overall_score']}/100 with privacy {validation['privacy_score']}/100.",
                status="Verified",
            )
            st.rerun()
    elif st.session_state.validation is None:
        render_role_restriction("This role can review validation output, but cannot run a new validation pass.")

    if st.session_state.validation is None:
        st.info("Run validation to unlock controlled release actions and comparison views.")
        return

    validation = st.session_state.validation
    metric_cols = st.columns(3)
    metric_cols[0].metric("Overall score", validation["overall_score"])
    metric_cols[1].metric("Fidelity score", validation["fidelity_score"])
    metric_cols[2].metric("Privacy score", validation["privacy_score"])

    dashboard_cols = st.columns(5)
    for col, metric in zip(dashboard_cols, build_validation_dashboard(metadata, controls)):
        col.markdown(
            f"""
            <div class="state-card">
                <h4>{metric['label']}</h4>
                <div class="state-value">{metric['value']}</div>
                <div class="state-text">{metric['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col.progress(int(round(metric["value"])))

    st.success(validation["verdict"])

    compare_tab, distribution_tab, checks_tab, role_tab, audit_tab = st.tabs(
        ["Source vs synthetic", "Column distributions", "Validation checks", "Role view & use cases", "Audit & release"]
    )
    with compare_tab:
        st.dataframe(
            build_comparison_table(st.session_state.source_df, st.session_state.synthetic_df, metadata),
            use_container_width=True,
            hide_index=True,
        )
    with distribution_tab:
        available_columns = [
            item["column"]
            for item in metadata
            if item["include"] and item["column"] in st.session_state.synthetic_df.columns and item["data_type"] != "identifier"
        ]
        if not available_columns:
            st.info("No comparable non-identifier columns are available for distribution review.")
        else:
            selected_column = st.selectbox("Select a column to compare", options=available_columns, key="distribution_column")
            comparison = build_distribution_comparison(metadata, selected_column)
            score_row = validation["fidelity_table"].loc[validation["fidelity_table"]["column"] == selected_column]
            meta_lookup = {item["column"]: item for item in metadata}
            role_name = meta_lookup[selected_column]["data_type"].replace("_", " ").title()
            summary_cols = st.columns(3)
            summary_cols[0].metric("Field", selected_column)
            summary_cols[1].metric("Field type", role_name)
            summary_cols[2].metric(
                "Column fidelity",
                score_row.iloc[0]["score"] if not score_row.empty else "N/A",
            )
            if comparison["kind"] == "line":
                st.line_chart(comparison["frame"], use_container_width=True)
            elif comparison["kind"] == "bar":
                st.bar_chart(comparison["frame"], use_container_width=True)
            else:
                st.info(comparison["note"])
            if not comparison["frame"].empty:
                st.dataframe(comparison["frame"].reset_index(), use_container_width=True, hide_index=True)
            st.caption(comparison["note"])
    with checks_tab:
        left_col, right_col = st.columns([1.1, 0.9], gap="large")
        with left_col:
            st.markdown("**Column-level fidelity**")
            st.dataframe(validation["fidelity_table"], use_container_width=True, hide_index=True)
        with right_col:
            st.markdown("**Privacy checks**")
            st.dataframe(validation["privacy_checks"], use_container_width=True, hide_index=True)
    with role_tab:
        render_role_guidance_panel(4, key_suffix="validation")
        st.markdown("**Recommended use cases for the current package**")
        st.dataframe(pd.DataFrame(build_use_case_rows(metadata, controls)), use_container_width=True, hide_index=True)

    with audit_tab:
        audit_cols = st.columns([1.05, 0.95], gap="large")
        with audit_cols[0]:
            st.markdown("**Recent audit activity**")
            st.dataframe(pd.DataFrame(st.session_state.audit_events), use_container_width=True, hide_index=True)
        with audit_cols[1]:
            st.markdown(
                """
                <div class="note-card">
                    <strong>Release controls</strong><br/>
                    Synthetic export remains governed through a release chain: validation, analyst request, policy approval,
                    and final Admin Office or Master authorization. If hygiene, metadata, or generation changes, the release gate resets.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Release status", effective_release_status(metadata, controls))
            st.metric("Requested by", st.session_state.export_requested_by or "Not requested")
            st.metric("Policy approved by", st.session_state.export_policy_approved_by or "Not approved")
            st.metric("Final authorization", st.session_state.export_approved_by or "Not approved")

            if has_permission("request_export"):
                if st.button("Request release review", use_container_width=True):
                    st.session_state.release_status = "Awaiting policy approval"
                    st.session_state.export_requested_by = st.session_state.current_role
                    st.session_state.export_policy_approved_by = None
                    st.session_state.export_approved_by = None
                    record_audit_event("Release review requested", f"Requested by {st.session_state.current_role}.", status="Requested")
                    st.rerun()
            if has_permission("approve_release_policy"):
                if st.button(
                    "Approve policy checkpoint",
                    use_container_width=True,
                    disabled=st.session_state.export_requested_by is None,
                ):
                    st.session_state.release_status = "Awaiting final authorization"
                    st.session_state.export_policy_approved_by = st.session_state.current_role
                    record_audit_event("Release policy approved", f"Approved by {st.session_state.current_role}.", status="Approved")
                    st.rerun()
            if has_permission("approve_export"):
                if st.button(
                    "Authorize final export",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.export_policy_approved_by is None,
                ):
                    st.session_state.release_status = "Approved"
                    st.session_state.export_approved_by = st.session_state.current_role
                    record_audit_event("Final export authorized", f"Authorized by {st.session_state.current_role}.", status="Approved")
                    st.rerun()

        st.markdown("**Approval hierarchy for release**")
        render_approval_hierarchy(build_release_approval_rows(metadata, controls), "release_approval")

    download_enabled = effective_release_status(metadata, controls) == "Approved"
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "Download synthetic dataset",
        data=st.session_state.synthetic_df.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_dataset.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not download_enabled,
    )
    export_cols[1].download_button(
        "Download validation report",
        data=build_validation_report(metadata, controls).encode("utf-8"),
        file_name="validation_report.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not download_enabled,
    )
    export_cols[2].download_button(
        "Download audit summary",
        data=pd.DataFrame(st.session_state.audit_events).to_csv(index=False).encode("utf-8"),
        file_name="audit_log.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=not download_enabled,
    )

    if not download_enabled:
        st.caption(
            "Downloads stay locked until the current package has a valid synthetic run, current validation, policy approval, and final Admin Office or Master authorization."
        )

    if has_permission("rollback"):
        rollback_cols = st.columns(2)
        if rollback_cols[0].button("Rollback to metadata review", use_container_width=True):
            clear_generation_outputs()
            st.session_state.current_step = 2
            record_audit_event("Rolled back to metadata review", "Downstream outputs cleared to revise the metadata package.", status="Rolled back")
            st.rerun()
        if rollback_cols[1].button("Rollback to hygiene review", use_container_width=True):
            clear_generation_outputs()
            st.session_state.metadata_status = "Draft"
            st.session_state.metadata_submitted_by = None
            st.session_state.metadata_submitted_at = None
            st.session_state.metadata_approved_by = None
            st.session_state.metadata_approved_at = None
            st.session_state.last_reviewed_metadata_signature = None
            st.session_state.current_metadata_package_id = None
            st.session_state.hygiene_reviewed = False
            st.session_state.current_step = 1
            record_audit_event("Rolled back to hygiene review", "Validation outputs cleared to revisit source quality.", status="Rolled back")
            st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="H", layout="wide", initial_sidebar_state="collapsed")
    inject_styles()
    initialize_app_state()

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
    render_action_center(metadata, controls)
    render_step_navigation(metadata, controls)

    current_step = st.session_state.current_step
    if current_step == 0:
        render_step_one(metadata)
    elif current_step == 1:
        render_step_two()
    elif current_step == 2:
        metadata, controls = render_step_three()
    elif current_step == 3:
        metadata, controls = render_step_four(metadata, controls)
    else:
        render_step_five(metadata, controls)

    persist_shared_workspace_state()


if __name__ == "__main__":
    main()
