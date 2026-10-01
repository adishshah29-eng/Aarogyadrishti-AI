"""
End-to-end smoke test for the Streamlit wizard, using Playwright to drive a
real browser against a locally-launched dashboard.

This exists because the wizard's session-state bugs (an AttributeError
advancing past Step 2; a SHAP KeyError on engineered feature columns) were
only caught by actually clicking through the UI — none of the unit-level
contract/regression tests would have noticed either one, since both only
manifest when Streamlit reruns the script with a different `wizard_step`
branch active.

Requires `pip install playwright && playwright install chromium` (or, in
this project's sandboxed dev environment, the pre-installed browser at
PLAYWRIGHT_BROWSERS_PATH). Skipped automatically if Playwright isn't
importable, so it never blocks a plain `pytest tests/`.

Run:  python -m pytest tests/test_wizard_e2e.py -q -s
"""
import contextlib
import os
import socket
import subprocess
import sys
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(ROOT, "src", "dashboard", "app.py")

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


FAKE_REPORT = os.path.join(ROOT, "tests", "fixtures", "sample_report_response.json")


@contextlib.contextmanager
def _streamlit_server(extra_env=None):
    port = _free_port()
    env = {**os.environ, **(extra_env or {})}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", APP_PATH,
         "--server.headless", "true", "--server.port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(30):
            time.sleep(1)
            try:
                socket.create_connection(("localhost", port), timeout=1).close()
                break
            except OSError:
                continue
        else:
            pytest.fail("Streamlit server did not start in time")
        time.sleep(3)  # let the first script run finish
        yield f"http://localhost:{port}"
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def streamlit_url():
    with _streamlit_server() as url:
        yield url


@pytest.fixture(scope="module")
def streamlit_url_fake_report():
    """App with the Gemini call swapped for a canned response (no network)."""
    with _streamlit_server({"AAROGYA_FAKE_REPORT_JSON": FAKE_REPORT}) as url:
        yield url


def _chromium_executable():
    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if browsers_path:
        candidate = os.path.join(browsers_path, "chromium")
        if os.path.exists(candidate):
            return candidate
    return None


def test_full_wizard_flow_reaches_results_without_error(streamlit_url):
    with sync_playwright() as p:
        launch_kwargs = {}
        exe = _chromium_executable()
        if exe:
            launch_kwargs["executable_path"] = exe
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        try:
            page.goto(streamlit_url, timeout=30000)
            page.wait_for_timeout(3000)

            page.get_by_role("button", name="Next →").click()   # Step 1 -> 2
            page.wait_for_timeout(3000)
            page.get_by_role("button", name="Next →").click()   # Step 2 -> 3
            page.wait_for_timeout(3000)
            page.get_by_role("button", name="Calculate Risk Profile →").click()
            page.wait_for_timeout(8000)

            content = page.content()
            for keyword in ("Traceback", "AttributeError", "KeyError", "NameError", "TypeError"):
                assert keyword not in content, f"'{keyword}' found on results page — wizard crashed"

            assert "Comorbidity Risk Index" in content
            assert "Diabetes" in content and "Hypertension" in content
        finally:
            browser.close()


def _launch(p):
    launch_kwargs = {}
    exe = _chromium_executable()
    if exe:
        launch_kwargs["executable_path"] = exe
    return p.chromium.launch(**launch_kwargs)


def test_report_upload_fills_wizard_and_lists_whats_missing(streamlit_url_fake_report, tmp_path):
    report = tmp_path / "lab_report.pdf"
    report.write_bytes(b"%PDF-1.4\n% placeholder: the fake extractor ignores file contents\n")
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(viewport={"width": 1400, "height": 1400})
        try:
            page.goto(streamlit_url_fake_report, timeout=30000)
            page.wait_for_timeout(3000)

            page.get_by_text("I agree to send this report").click()
            page.wait_for_timeout(1500)
            page.locator("input[type=file]").set_input_files(str(report))
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="Read report").click()
            page.wait_for_timeout(4000)

            content = page.content()
            assert "Found <strong>7 of 13</strong>" in content
            assert "Still needed for a prediction" in content
            assert "Systolic BP" in content and "Height" in content
            assert "will use a population average" in content
            page.screenshot(path=str(tmp_path / "review.png"), full_page=True)

            page.get_by_role("button", name="Use these values →").click()
            page.wait_for_timeout(3000)
            # Height/weight weren't on the report → back on Step 1 with badges.
            assert page.get_by_label("Age").input_value() == "52"
            assert page.locator(".src-badge").count() >= 2

            page.get_by_role("button", name="Next →").click()
            page.wait_for_timeout(3000)
            assert page.get_by_label("Fasting blood glucose (mg/dL)").input_value() == "133"
            assert page.get_by_label("Blood urea nitrogen / BUN (mg/dL)").input_value() == "14.9"
            # BP wasn't on the report: both fields are tagged for the user to fill in.
            assert page.locator(".src-missing").count() == 2
            page.screenshot(path=str(tmp_path / "step2.png"), full_page=True)

            page.get_by_role("button", name="Next →").click()
            page.wait_for_timeout(3000)
            page.get_by_role("button", name="Calculate Risk Profile →").click()
            page.wait_for_timeout(8000)
            content = page.content()
            for keyword in ("Traceback", "AttributeError", "KeyError", "NameError", "TypeError",
                            "StreamlitAPIException"):
                assert keyword not in content, f"'{keyword}' found after report upload flow"
            assert "Comorbidity Risk Index" in content
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
