"""
Tests for the lab-report upload pipeline (src/report_ingest).

No network: Gemini is never called. The extractor is exercised through a fake
client and through parse_response_text, which is the same code path a real
response goes through.

Run:  python -m pytest tests/test_report_ingest.py -q
"""
import json
import os
import sys
import types as pytypes

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.report_ingest import ReportReadError, process_report, apply_to_pdata, FAKE_RESPONSE_ENV
from src.report_ingest import extract_gemini
from src.report_ingest.fields import FIELDS, SPECS, MODEL_RAW_FEATURES, models_using
from src.report_ingest.normalize import OK, CHECK, REJECTED, build_result, normalize_item, normalize_unit

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "sample_report_response.json")


def _one(**item):
    return normalize_item(item)


# ── unit conversions ──
@pytest.mark.parametrize("field,value,unit,expected", [
    ("glucose", 7.0, "mmol/L", 126),           # diagnostic threshold, both units
    ("glucose", 126, "mg/dL", 126),
    ("glucose", 110, "mg%", 110),
    ("cholesterol", 5.2, "mmol/L", 201),
    ("triglycerides", 1.7, "mmol/l", 150.6),
    ("uric_acid", 357, "µmol/L", 6.0),
    ("uric_acid", 357, "umol/L", 6.0),
    ("height", 1.72, "m", 172.0),
    ("weight", 176, "lbs", 79.8),
    ("waist_circumference", 36, "in", 91.4),
])
def test_unit_conversion(field, value, unit, expected):
    r = _one(field=field, value=value, unit=unit, qualifier="fasting")
    assert r.status == OK, r.note
    assert r.value == pytest.approx(expected, abs=0.15)


def test_integer_fields_become_ints_for_the_wizard_widgets():
    r = _one(field="glucose", value=7.4, unit="mmol/L", qualifier="fasting")
    assert isinstance(r.value, int)
    r = _one(field="bun", value=14, unit="mg/dL", qualifier="bun")
    assert isinstance(r.value, float)


# ── BUN: urea vs urea nitrogen ──
def test_blood_urea_is_converted_to_bun():
    r = _one(field="bun", value=32, unit="mg/dL", qualifier="urea", printed_name="Blood Urea")
    assert r.value == pytest.approx(14.9, abs=0.05)
    assert "urea" in r.note.lower()


def test_bun_printed_as_bun_is_not_converted():
    r = _one(field="bun", value=15, unit="mg/dL", qualifier="bun", printed_name="Blood Urea Nitrogen (BUN)")
    assert r.value == 15.0 and r.note == ""
    # qualifier missing but name says nitrogen → still BUN
    r = _one(field="bun", value=15, unit="mg/dL", printed_name="Urea Nitrogen")
    assert r.value == 15.0


def test_urea_in_mmol_is_same_molar_amount_as_bun():
    r = _one(field="bun", value=5.0, unit="mmol/L", qualifier="urea")
    assert r.value == pytest.approx(14.0, abs=0.05)


# ── glucose must be fasting ──
def test_post_prandial_glucose_is_not_used():
    r = _one(field="glucose", value=180, unit="mg/dL", qualifier="post_prandial")
    assert r.status == REJECTED


def test_unlabelled_glucose_needs_confirmation():
    r = _one(field="glucose", value=105, unit="mg/dL", qualifier="unknown")
    assert r.status == CHECK


def test_fasting_reading_wins_over_pp_reading_for_same_field():
    result = build_result([
        {"field": "glucose", "value": 200, "unit": "mg/dL", "qualifier": "post_prandial"},
        {"field": "glucose", "value": 98, "unit": "mg/dL", "qualifier": "fasting"},
    ])
    assert result.readings["glucose"].value == 98


# ── rejection paths ──
def test_mmol_value_mislabelled_as_mg_dl_is_rejected_by_range():
    """The classic misread: 7.2 (really mmol/L) labelled mg/dL would look like
    severe hypoglycaemia. It falls outside the widget range and is rejected."""
    r = _one(field="glucose", value=7.2, unit="mg/dL", qualifier="fasting")
    assert r.status == REJECTED


def test_unknown_unit_is_rejected():
    r = _one(field="cholesterol", value=5, unit="g/L")
    assert r.status == REJECTED and r.value is None


def test_missing_unit_is_assumed_but_flagged():
    r = _one(field="cholesterol", value=210)
    assert r.status == CHECK and r.value == 210


def test_unrequested_fields_are_dropped():
    result = build_result([{"field": "patient_name", "text": "A. Person"},
                           {"field": "age", "value": 40, "unit": "years"}])
    assert result.dropped == 1
    assert set(result.readings) == {"age"}


def test_inverted_blood_pressure_is_flagged():
    result = build_result([{"field": "systolic_bp", "value": 80, "unit": "mmHg"},
                           {"field": "diastolic_bp", "value": 120, "unit": "mmHg"}])
    assert result.readings["systolic_bp"].status == CHECK


def test_sex_normalization():
    assert _one(field="sex", text="F").value == "Female"
    assert _one(field="sex", text="Male").value == "Male"
    assert _one(field="sex", text="unclear").status == REJECTED


def test_normalize_unit_variants():
    assert normalize_unit(" mg / dL ") == "mg/dl"
    assert normalize_unit("μmol/L") == "umol/l"
    assert normalize_unit("mg/100ml") == "mg/dl"


# ── whole-report result + what's missing ──
def test_fixture_report_found_and_missing():
    os.environ[FAKE_RESPONSE_ENV] = FIXTURE
    try:
        result = process_report(b"", "application/pdf", api_key="")
    finally:
        del os.environ[FAKE_RESPONSE_ENV]

    usable = result.usable()
    assert usable["glucose"] == 133          # 7.4 mmol/L fasting, PP reading ignored
    assert usable["uric_acid"] == pytest.approx(6.9)
    assert usable["bun"] == pytest.approx(14.9)
    assert usable["sex"] == "Male"
    assert "hba1c" not in usable             # context only, no model uses it
    assert [r.key for r in result.context_only()] == ["hba1c"]
    assert result.dropped == 1

    missing_required = {s.key for s in result.missing(required=True)}
    assert missing_required == {"height", "weight", "systolic_bp", "diastolic_bp"}
    missing_optional = {s.key for s in result.missing(required=False)}
    assert missing_optional == {"waist_circumference", "resting_pulse"}


# ── applying to the wizard ──
def test_apply_marks_missing_optional_as_unknown_and_picks_next_step():
    pdata = {}
    step = apply_to_pdata(pdata, {"age": 52, "sex": "Male", "glucose": 133, "bun": 14.9})
    assert step == 1                      # height/weight still missing
    assert pdata["idk_bun"] is False
    assert pdata["idk_trig"] is True      # not on report, never typed → median
    assert pdata["_report_values"]["glucose"] == 133


def test_apply_keeps_a_value_the_user_already_typed():
    pdata = {"triglycerides": 140.0, "idk_trig": False}
    apply_to_pdata(pdata, {"glucose": 99})
    assert pdata["idk_trig"] is False and pdata["triglycerides"] == 140.0


def test_apply_computes_bmi_and_skips_to_lifestyle_when_complete():
    values = {"age": 50, "sex": "Female", "height": 160.0, "weight": 64.0,
              "systolic_bp": 128, "diastolic_bp": 82, "glucose": 101, "cholesterol": 190}
    pdata = {}
    assert apply_to_pdata(pdata, values) == 3
    assert pdata["bmi"] == pytest.approx(25.0)


# ── extractor, with a fake Gemini client ──
class _FakeModels:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if self.exc:
            raise self.exc
        return pytypes.SimpleNamespace(text=self.text)


def _patch_client(monkeypatch, fake_models):
    from google import genai
    monkeypatch.setattr(genai, "Client", lambda api_key: pytypes.SimpleNamespace(models=fake_models))


def test_extract_sends_inline_bytes_and_parses(monkeypatch):
    with open(FIXTURE, encoding="utf-8") as fh:
        fake = _FakeModels(text=fh.read())
    _patch_client(monkeypatch, fake)
    items = extract_gemini.extract(b"%PDF-1.4 fake", "application/pdf", api_key="k")
    assert any(i["field"] == "glucose" for i in items)
    model, contents, config = fake.calls[0]
    assert contents[0].inline_data.data == b"%PDF-1.4 fake"   # inline, not Files API
    assert config.temperature == 0


def test_extract_reads_docx_as_text(monkeypatch):
    import docx, io
    d = docx.Document()
    d.add_paragraph("Glucose Fasting 98 mg/dL")
    buf = io.BytesIO(); d.save(buf)
    fake = _FakeModels(text=json.dumps({"is_medical_report": True, "items": []}))
    _patch_client(monkeypatch, fake)
    extract_gemini.extract(buf.getvalue(), extract_gemini.DOCX_MIME, api_key="k")
    assert "Glucose Fasting 98" in fake.calls[0][1][0].text


def test_extract_maps_api_errors_to_plain_messages(monkeypatch):
    from google.genai import errors
    exc = errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    _patch_client(monkeypatch, _FakeModels(exc=exc))
    with pytest.raises(ReportReadError, match="quota"):
        extract_gemini.extract(b"x", "image/png", api_key="k")


def test_non_medical_document_is_refused():
    with pytest.raises(ReportReadError, match="medical"):
        extract_gemini.parse_response_text(json.dumps({"is_medical_report": False, "items": []}))


def test_extract_guards_before_any_network_call():
    with pytest.raises(ReportReadError, match="10 MB"):
        extract_gemini.extract(b"x" * (extract_gemini.MAX_BYTES + 1), "application/pdf", api_key="k")
    with pytest.raises(ReportReadError, match="Unsupported"):
        extract_gemini.extract(b"x", "application/zip", api_key="k")
    with pytest.raises(ReportReadError, match="API key"):
        extract_gemini.extract(b"x", "application/pdf", api_key="")


def test_prompt_forbids_identifiers_and_unit_conversion():
    prompt = extract_gemini.build_prompt()
    assert "Never output the patient's name" in prompt
    assert "Do NOT convert units" in prompt
    assert all(f'"{s.key}"' in prompt for s in FIELDS)


# ── contract: every fillable field reaches a real model input ──
def test_every_report_field_feeds_a_model_or_is_marked_context_only():
    for spec in FIELDS:
        if spec.step == 0:
            assert not spec.model_inputs, f"{spec.key} is context-only but claims model inputs"
            continue
        assert models_using(spec), f"{spec.key} would be filled from a report but no model reads it"


def test_report_field_bounds_match_the_wizard_widgets():
    """A pre-filled value outside a number_input's min/max makes Streamlit
    raise, so the registry bounds must equal the wizard's."""
    import re
    src = open(os.path.join(ROOT, "src", "dashboard", "app.py"), encoding="utf-8").read()
    label_by_key = {
        "age": '"Age"', "height": '"Height (cm)"', "weight": '"Weight (kg)"',
        "systolic_bp": '"Systolic BP (upper number)"', "diastolic_bp": '"Diastolic BP (lower number)"',
        "glucose": '"Fasting blood glucose (mg/dL)"', "cholesterol": '"Total cholesterol (mg/dL)"',
        "waist_circumference": '"Waist circumference (cm)"', "resting_pulse": '"Resting pulse (bpm)"',
        "uric_acid": '"Uric acid (mg/dL)"', "bun": '"Blood urea nitrogen / BUN (mg/dL)"',
        "triglycerides": '"Triglycerides (mg/dL)"',
    }
    for key, label in label_by_key.items():
        m = re.search(re.escape(label) + r",\s*min_value=([\d.]+),\s*max_value=([\d.]+)", src)
        assert m, f"wizard widget for {key} not found"
        lo, hi = float(m.group(1)), float(m.group(2))
        spec = SPECS[key]
        assert (spec.lo, spec.hi) == (lo, hi), f"{key}: registry {spec.lo}-{spec.hi} vs widget {lo}-{hi}"
        assert spec.integer == ("." not in m.group(1)), f"{key}: int/float mismatch with widget"
