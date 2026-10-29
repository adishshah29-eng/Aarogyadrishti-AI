"""
Streamlit UI for the optional lab-report upload on wizard Step 1.

Everything lives in st.session_state for this browser session only: the
uploaded bytes are never written to disk, and "Remove report" / "Start new
assessment" / closing the tab discards them.
"""
import hashlib
import math
import os

import pandas as pd
import streamlit as st

from src.report_ingest import (DEFAULT_MODEL, ReportReadError, apply_to_pdata, check_value,
                               fake_extractor_enabled, process_report)
from src.report_ingest.fields import SELF_REPORTED, SPECS, models_using
from src.report_ingest.normalize import CHECK, OK, REJECTED

STATE = "report_state"
EXT_MIME = {
    "pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "webp": "image/webp", "txt": "text/plain",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_STATUS_TEXT = {OK: "OK", CHECK: "Check", REJECTED: "Not used"}


def _secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


def configured_key():
    return _secret("GEMINI_API_KEY")


def active_key():
    return configured_key() or st.session_state.get("gemini_key_input", "").strip()


def clear_report():
    st.session_state.pop(STATE, None)
    st.session_state.get("pdata", {}).pop("_report_values", None)
    st.session_state["report_uploader_n"] = st.session_state.get("report_uploader_n", 0) + 1


def render_sidebar_key():
    if configured_key() or fake_extractor_enabled():
        return
    st.markdown('<div class="sb-section">Lab report reading</div>', unsafe_allow_html=True)
    st.text_input("Gemini API key (optional)", type="password", key="gemini_key_input",
                  help="Used only to read an uploaded lab report. Kept for this browser "
                       "session only — never saved.")


def source_badge(pdata: dict, key: str, current):
    """Small 'from your report' tag under a wizard field, shown while the
    field still holds the value read from the report."""
    report_values = pdata.get("_report_values")
    if report_values is None:
        return
    original = report_values.get(key)
    if original is None:
        if SPECS[key].required:
            st.markdown('<div class="src-badge src-missing">not on your report — enter it here</div>',
                        unsafe_allow_html=True)
        return
    same = (original == current) if isinstance(original, str) else math.isclose(
        float(original), float(current), abs_tol=0.05)
    if same:
        st.markdown('<div class="src-badge">from your report</div>', unsafe_allow_html=True)


def _review_frame(result) -> pd.DataFrame:
    rows = []
    for key, r in result.readings.items():
        spec = SPECS[key]
        if spec.step == 0 or spec.categorical:
            continue
        rows.append({
            "key": key,
            "Use": r.status != REJECTED,
            "Field": spec.label,
            "Value": float(r.value) if isinstance(r.value, (int, float)) else None,
            "Unit": spec.unit,
            "Status": _STATUS_TEXT[r.status],
            "As printed": r.printed,
            "Report line": r.evidence,
            "Note": r.note,
        })
    return pd.DataFrame(rows, columns=["key", "Use", "Field", "Value", "Unit", "Status",
                                       "As printed", "Report line", "Note"])


def _missing_panel(result):
    req, opt = result.missing(required=True), result.missing(required=False)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Still needed for a prediction**")
        if req:
            for s in req:
                st.markdown(f"- {s.label} <span class='rp-dim'>· Step {s.step}</span>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<span class='rp-dim'>Nothing — all found.</span>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Optional, improves accuracy**")
        if opt:
            for s in opt:
                st.markdown(f"- {s.label} <span class='rp-dim'>· {', '.join(models_using(s))} "
                            f"will use a population average</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='rp-dim'>Nothing — all found.</span>", unsafe_allow_html=True)
    with c3:
        st.markdown("**Only you can answer (Step 3)**")
        for q in SELF_REPORTED:
            st.markdown(f"- {q}")


def _render_review(result, pdata, digest):
    found = len(result.usable())
    total = sum(1 for s in SPECS.values() if s.step > 0)
    st.markdown(f"Found **{found} of {total}** values the models use. Check them, edit anything "
                "that's wrong, and untick anything you don't want used.")

    sex_reading = result.readings.get("sex")
    sex_choice = None
    if sex_reading and sex_reading.status != REJECTED:
        sex_choice = st.selectbox("Sex (from report)", ["Male", "Female"],
                                  index=["Male", "Female"].index(sex_reading.value),
                                  key=f"report_sex_{digest[:8]}")

    frame = _review_frame(result)
    edited = frame
    if not frame.empty:
        edited = st.data_editor(
            frame, hide_index=True, use_container_width=True, key=f"report_editor_{digest[:8]}",
            disabled=["Field", "Unit", "Status", "As printed", "Report line", "Note"],
            column_config={
                "key": None,
                "Use": st.column_config.CheckboxColumn(width="small"),
                "Value": st.column_config.NumberColumn(format="%.1f", width="small"),
                "Note": st.column_config.TextColumn(width="large"),
            },
        )

    for r in result.context_only():
        st.caption(f"{r.label} {r.printed} — shown for reference; none of the models use it.")

    _missing_panel(result)

    left, right = st.columns([1, 3])
    with left:
        if st.button("Remove report", use_container_width=True):
            clear_report()
            st.rerun()
    with right:
        if st.button("Use these values →", type="primary", use_container_width=True):
            values, problems = {}, []
            for row in edited.itertuples(index=False):
                value = row.Value
                if not row.Use or value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                value, err = check_value(SPECS[row.key], float(value))
                if err:
                    problems.append(f"{row.Field}: {err}")
                else:
                    values[row.key] = value
            if sex_choice:
                values["sex"] = sex_choice
            if problems:
                for p in problems:
                    st.error(p)
            else:
                st.session_state[STATE]["applied"] = True
                st.session_state.wizard_step = apply_to_pdata(pdata, values)
                st.rerun()


def render_report_panel(pdata: dict):
    state = st.session_state.get(STATE)
    key = active_key()
    can_read = bool(key) or fake_extractor_enabled()

    with st.container(border=True):
        st.markdown('<div class="wiz-card-title">Have a lab report? Let it fill this in</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="wiz-card-sub">Upload a PDF, photo, or Word file of your blood '
                    'test. We read the values, you check them, and we tell you what\'s still '
                    'missing. Optional — you can skip this and fill the form yourself.</div>',
                    unsafe_allow_html=True)

        if state and state.get("applied"):
            st.success(f"Values from **{state['name']}** are filled in below. Fields marked "
                       "\"from your report\" came from it.")
            with st.expander("Review report values again"):
                _render_review(state["result"], pdata, state["hash"])
            return

        if not can_read:
            st.info("Reading reports needs a Gemini API key. Add one in the sidebar, or fill "
                    "in the form below yourself.")
            return

        consent = st.checkbox(
            "I agree to send this report to Google's Gemini API to read it. This app saves "
            "nothing. On Google's free (unpaid) API tier, Google may use submitted content to "
            "improve its products; paid-tier keys aren't used that way.",
            value=st.session_state.get("report_consent_given", False),
        )
        st.session_state["report_consent_given"] = consent

        upload = st.file_uploader(
            "Lab report", type=list(EXT_MIME), disabled=not consent, label_visibility="collapsed",
            key=f"report_upload_{st.session_state.get('report_uploader_n', 0)}",
        )
        if upload is None:
            if state:
                _render_review(state["result"], pdata, state["hash"])
            return

        data = upload.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if state and state["hash"] == digest:
            _render_review(state["result"], pdata, digest)
            return

        if st.button("Read report", type="primary"):
            ext = upload.name.rsplit(".", 1)[-1].lower()
            mime = EXT_MIME.get(ext) or upload.type
            model = _secret("GEMINI_MODEL") or DEFAULT_MODEL
            try:
                with st.spinner("Reading your report…"):
                    result = process_report(data, mime, key, model)
            except ReportReadError as exc:
                st.error(str(exc))
                return
            # Only the extracted values are kept; the file bytes are dropped here.
            st.session_state[STATE] = {"hash": digest, "name": upload.name, "result": result,
                                       "applied": False}
            st.rerun()
