"""Lab-report upload: read with Gemini, normalize in Python, pre-fill the wizard."""
import json
import os

from src.report_ingest.extract_gemini import DEFAULT_MODEL, ReportReadError, extract, parse_response_text
from src.report_ingest.normalize import ReportResult, apply_to_pdata, build_result, check_value

# Test hook: point this at a JSON file shaped like Gemini's response to run the
# whole upload flow (including the browser E2E test) without a network call.
FAKE_RESPONSE_ENV = "AAROGYA_FAKE_REPORT_JSON"


def fake_extractor_enabled() -> bool:
    return bool(os.environ.get(FAKE_RESPONSE_ENV))


def process_report(data: bytes, mime: str, api_key: str, model: str = DEFAULT_MODEL) -> ReportResult:
    fake = os.environ.get(FAKE_RESPONSE_ENV)
    if fake:
        with open(fake, encoding="utf-8") as fh:
            items = parse_response_text(json.dumps(json.load(fh)))
    else:
        items = extract(data, mime, api_key, model)
    return build_result(items)


__all__ = ["ReportReadError", "ReportResult", "process_report", "apply_to_pdata",
           "check_value", "fake_extractor_enabled", "DEFAULT_MODEL"]
