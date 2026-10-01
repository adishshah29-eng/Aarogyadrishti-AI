"""
Deterministic post-processing of what Gemini read off a report.

Gemini only transcribes (value + unit exactly as printed). Every conversion,
range check and judgement call happens here, so the risky part is plain,
testable Python rather than model output.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from src.report_ingest.fields import SPECS, FIELDS, FieldSpec

OK, CHECK, REJECTED = "ok", "check", "rejected"
_STATUS_RANK = {OK: 0, CHECK: 1, REJECTED: 2}

# Urea is ~46.7% nitrogen by mass (28/60.06): BUN (mg/dL) = urea (mg/dL) x 0.467.
UREA_TO_BUN = 0.467


@dataclass
class FieldReading:
    key: str
    label: str
    value: object                 # canonical value (float/int/str) or None
    unit: str
    printed: str                  # value + unit as the report printed it
    evidence: str
    status: str
    note: str = ""


@dataclass
class ReportResult:
    readings: dict = field(default_factory=dict)   # key -> FieldReading (best per field)
    dropped: int = 0                                # items for fields we never asked for

    def usable(self) -> dict:
        """Wizard-field values safe to pre-fill (ok or flagged-for-check)."""
        return {k: r.value for k, r in self.readings.items()
                if r.status != REJECTED and SPECS[k].step > 0}

    def context_only(self) -> list:
        return [r for k, r in self.readings.items() if SPECS[k].step == 0 and r.status != REJECTED]

    def missing(self, required: bool) -> list:
        found = self.usable()
        return [s for s in FIELDS if s.step > 0 and s.required == required and s.key not in found]


def normalize_unit(unit: Optional[str]) -> str:
    if not unit:
        return ""
    u = unit.strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    u = u.replace("mg/100ml", "mg/dl").replace("mg%", "mg/dl").replace("mgs/dl", "mg/dl")
    u = u.replace("mmhg.", "mmhg").replace("beats/minute", "beats/min").replace("perminute", "/min")
    if u in ("permin", "/minute"):
        u = "/min"
    return u


def check_value(spec: FieldSpec, value: float):
    """Coerce a canonical value to the widget's type and verify its bounds.
    Returns (value, error_message_or_None)."""
    if spec.integer:
        value = int(round(value))
    else:
        value = round(float(value), 1)
    if not (spec.lo <= value <= spec.hi):
        return value, (f"{value:g} {spec.unit} is outside the accepted range "
                       f"{spec.lo:g}–{spec.hi:g} {spec.unit} — check the unit.")
    return value, None


def _normalize_sex(text: Optional[str]):
    t = (text or "").strip().lower()
    if t in ("m", "male", "man"):
        return "Male"
    if t in ("f", "female", "woman"):
        return "Female"
    return None


def _bun_factor(unit: str, qualifier: str, printed_name: str):
    """Multiplier to BUN mg/dL, plus a note when the report printed urea."""
    name = (printed_name or "").lower()
    q = (qualifier or "").lower()
    is_bun = q == "bun" or "nitrogen" in name or re.search(r"\bbun\b", name) is not None
    if unit == "mmol/l":
        # Urea and urea-nitrogen are the same molar quantity.
        return 2.801, ""
    if unit in ("mg/dl", ""):
        if is_bun:
            return 1.0, ""
        return UREA_TO_BUN, "Urea → BUN (×0.467)."
    return None, ""


def normalize_item(item: dict) -> Optional[FieldReading]:
    key = (item.get("field") or "").strip()
    spec = SPECS.get(key)
    if spec is None:
        return None

    raw_value, raw_unit = item.get("value"), item.get("unit") or ""
    printed = f"{raw_value:g} {raw_unit}".strip() if isinstance(raw_value, (int, float)) else (item.get("text") or "")
    evidence = (item.get("evidence") or "")[:120]

    if spec.categorical:
        sex = _normalize_sex(item.get("text"))
        if sex is None:
            return FieldReading(key, spec.label, None, "", printed, evidence, REJECTED,
                                "Couldn't tell male/female from the report.")
        return FieldReading(key, spec.label, sex, "", printed, evidence, OK)

    if not isinstance(raw_value, (int, float)):
        return FieldReading(key, spec.label, None, spec.unit, printed, evidence, REJECTED, "No number found.")

    unit = normalize_unit(raw_unit)
    notes, status = [], OK

    if key == "bun":
        factor, note = _bun_factor(unit, item.get("qualifier"), item.get("printed_name"))
        if note:
            notes.append(note)
    else:
        factor = dict(spec.units).get(unit) if unit else 1.0

    if factor is None:
        return FieldReading(key, spec.label, None, spec.unit, printed, evidence, REJECTED,
                            f"Unit '{raw_unit}' not recognised for {spec.label}.")
    if not unit:
        status = CHECK
        notes.append(f"No unit printed; assumed {spec.unit}." if spec.unit else "")
    elif factor != 1.0 and key != "bun":
        notes.append(f"Converted from {raw_unit}.")

    value, err = check_value(spec, float(raw_value) * factor)
    if err:
        return FieldReading(key, spec.label, value, spec.unit, printed, evidence, REJECTED, err)

    if key == "glucose":
        kind = (item.get("qualifier") or "unknown").lower()
        if kind in ("random", "post_prandial", "pp", "postprandial"):
            return FieldReading(key, spec.label, value, spec.unit, printed, evidence, REJECTED,
                                "This is a random/after-meal reading; the model needs fasting glucose.")
        if kind != "fasting":
            status = CHECK
            notes.append("Report doesn't say this was fasting — confirm before using.")

    return FieldReading(key, spec.label, value, spec.unit, printed, evidence, status,
                        " ".join(n for n in notes if n))


def build_result(items: list) -> ReportResult:
    result = ReportResult()
    for item in items or []:
        reading = normalize_item(item)
        if reading is None:
            result.dropped += 1
            continue
        current = result.readings.get(reading.key)
        if current is None or _STATUS_RANK[reading.status] < _STATUS_RANK[current.status]:
            result.readings[reading.key] = reading

    sbp, dbp = result.readings.get("systolic_bp"), result.readings.get("diastolic_bp")
    if sbp and dbp and REJECTED not in (sbp.status, dbp.status) and sbp.value <= dbp.value:
        for r in (sbp, dbp):
            r.status = CHECK
            r.note = (r.note + " Systolic should be higher than diastolic.").strip()
    return result


def apply_to_pdata(pdata: dict, values: dict) -> int:
    """Write confirmed report values into the wizard's PDATA dict and return
    the wizard step the user should land on next.

    Optional fields the report didn't have are switched to "I don't know"
    (population median) unless the user already typed a value themselves —
    otherwise the wizard's placeholder default would silently pass as data.
    """
    for key, value in values.items():
        pdata[key] = value
    for spec in FIELDS:
        if spec.idk_key is None:
            continue
        if spec.key in values:
            pdata[spec.idk_key] = False
        elif spec.key not in pdata:
            pdata[spec.idk_key] = True

    if pdata.get("height") and pdata.get("weight"):
        pdata["bmi"] = pdata["weight"] / ((pdata["height"] / 100) ** 2)

    pdata["_report_values"] = dict(values)

    for step in (1, 2):
        if any(s.step == step and s.required and s.key not in pdata for s in FIELDS):
            return step
    return 3
