"""Publication figures and tables for an IEEE paper.

Trains a fresh model for each shipped recipe on a stratified 80% split
(random_state=42) and scores the untouched 20%. Features, engineering,
chaining, and XGBoost settings come from scripts/generate_eval_cache.py
and src/models. Nothing here is an in-sample score of a model trained on
the full dataset. Outputs are written to reports/paper/.

Requires matplotlib.
"""
import importlib.util
import inspect
import os
import platform
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

from src.chaining.cri import compute_cri

OUT_DIR = os.path.join(ROOT, "reports", "paper")
FIG_SIZE = (3.4, 2.8)
DPI = 300

# Plot order requested for the paper. Internal keys match generate_eval_cache.
PLOT_ORDER = ["Diabetes", "Heart Disease", "CKD", "Hypertension"]
LEGEND_NAME = {
    "Diabetes": "Diabetes",
    "Heart Disease": "Heart",
    "CKD": "CKD",
    "Hypertension": "Hypertension",
}
# Distinct dash patterns so the curves stay separable in grayscale.
LINE_STYLES = {
    "Diabetes": ("-", "0.0"),
    "Heart Disease": ("--", "0.0"),
    "CKD": ("-.", "0.15"),
    "Hypertension": ((0, (1, 1)), "0.0"),
}


def _load_eval_module():
    path = os.path.join(ROOT, "scripts", "generate_eval_cache.py")
    spec = importlib.util.spec_from_file_location("generate_eval_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _use_serif_font():
    candidates = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\Times.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            return name
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
    return "Times New Roman"


def _style_axes(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(length=2.5, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.grid(False)


def _save_figure(fig, stem):
    png = os.path.join(OUT_DIR, stem + ".png")
    pdf = os.path.join(OUT_DIR, stem + ".pdf")
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _new_figure():
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.97)
    return fig, ax


def plot_roc(results):
    fig, ax = _new_figure()
    ax.plot([0, 1], [0, 1], color="0.65", linewidth=0.6, linestyle="-", zorder=0)
    for name in PLOT_ORDER:
        y_test = results[name]["y_test"]
        probs = results[name]["probs"]
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = auc(fpr, tpr)
        style, color = LINE_STYLES[name]
        ax.plot(
            fpr, tpr, linestyle=style, color=color, linewidth=1.1,
            label=f"{LEGEND_NAME[name]} (AUC = {score:.3f})",
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    _style_axes(ax)
    ax.legend(
        loc="lower right", frameon=True, fancybox=False, edgecolor="0.8",
        facecolor="white", framealpha=1, handlelength=2.4, borderpad=0.3,
    )
    return _save_figure(fig, "roc_curves")


def plot_calibration(results):
    fig, ax = _new_figure()
    ax.plot([0, 1], [0, 1], color="0.65", linewidth=0.6, linestyle="-", zorder=0)
    for name in PLOT_ORDER:
        y_test = results[name]["y_test"]
        probs = results[name]["probs"]
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_test, probs)
        style, color = LINE_STYLES[name]
        ax.plot(
            mean_pred, frac_pos, linestyle=style, color=color, linewidth=1.1,
            label=f"{LEGEND_NAME[name]} (Brier = {brier:.3f})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    _style_axes(ax)
    ax.legend(
        loc="upper left", frameon=True, fancybox=False, edgecolor="0.8",
        facecolor="white", framealpha=1, handlelength=2.4, borderpad=0.3,
    )
    return _save_figure(fig, "calibration_curves")


def _at_threshold(y_test, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y_test, pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }


def diabetes_thresholds(y_test, probs):
    """Default 0.5, Youden's J, and the highest threshold with sensitivity >= 0.85."""
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    youden_idx = int(np.argmax(tpr - fpr))
    eligible = np.where(tpr >= 0.85)[0]
    if len(eligible) == 0:
        raise RuntimeError("No diabetes threshold reaches 85% sensitivity on the held-out split.")
    # roc_curve thresholds decrease as sensitivity rises, so the first hit is
    # the highest threshold that still meets the sensitivity floor.
    sens_idx = int(eligible[0])
    rows = [
        ("default_0.5", _at_threshold(y_test, probs, 0.5)),
        ("youden_j", _at_threshold(y_test, probs, thresholds[youden_idx])),
        ("sensitivity_at_least_0.85", _at_threshold(y_test, probs, thresholds[sens_idx])),
    ]
    path = os.path.join(OUT_DIR, "diabetes_thresholds.csv")
    header = "rule,threshold,sensitivity,specificity,precision,f1,accuracy"
    lines = [header]
    for rule, metrics in rows:
        lines.append(
            "{rule},{threshold:.6f},{sensitivity:.6f},{specificity:.6f},"
            "{precision:.6f},{f1:.6f},{accuracy:.6f}".format(rule=rule, **metrics)
        )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return path, rows


def test_set_summary(results):
    path = os.path.join(OUT_DIR, "test_set_summary.csv")
    lines = ["model,n_test,n_positive,positive_rate"]
    for name in PLOT_ORDER:
        y_test = results[name]["y_test"]
        n_test = int(len(y_test))
        n_positive = int(y_test.sum())
        rate = float(y_test.mean())
        lines.append(f"{LEGEND_NAME[name]},{n_test},{n_positive},{rate:.6f}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _package_version(dist_name):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "not installed"


def _cpu_model():
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _ram_bytes():
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys)
    if hasattr(os, "sysconf"):
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page * pages)
    return None


def _cri_block():
    """Weights are the literals in compute_cri(); the numeric check guards drift."""
    weights = {
        "P(Diabetes)": 0.30,
        "P(CKD)": 0.20,
        "P(Heart)": 0.25,
        "P(Hypertension)": 0.25,
        "P(Diabetes)*P(Heart)": 0.15,
        "P(Hypertension)*P(Heart)": 0.10,
        "P(Diabetes)*P(Hypertension)": 0.05,
    }
    sample = compute_cri(0.7, 0.1, 0.8, 0.1)
    expected = float(np.clip(
        0.30 * 0.7 + 0.20 * 0.1 + 0.25 * 0.8 + 0.25 * 0.1
        + 0.15 * 0.7 * 0.8 + 0.10 * 0.1 * 0.8 + 0.05 * 0.7 * 0.1,
        0.0, 1.0,
    ))
    if abs(sample - expected) > 1e-9:
        raise RuntimeError("Printed CRI weights do not match compute_cri().")
    lines = [
        "CRI formula (src/chaining/cri.py, compute_cri):",
        "  CRI = clip(0.30*P(Diabetes) + 0.20*P(CKD) + 0.25*P(Heart) + 0.25*P(Hypertension)",
        "            + 0.15*P(Diabetes)*P(Heart)",
        "            + 0.10*P(Hypertension)*P(Heart)",
        "            + 0.05*P(Diabetes)*P(Hypertension), 0, 1)",
        "Weights:",
    ]
    for term, weight in weights.items():
        lines.append(f"  {term}: {weight:.2f}")
    lines.append("Source:")
    lines.append(inspect.getsource(compute_cri).rstrip())
    return "\n".join(lines)


def _fmt_param(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return repr(value)


def environment_info(results):
    ram = _ram_bytes()
    ram_text = f"{ram / (1024 ** 3):.2f} GB ({ram} bytes)" if ram else "unavailable"
    lines = [
        f"Python: {platform.python_version()}",
        f"xgboost: {_package_version('xgboost')}",
        f"scikit-learn: {_package_version('scikit-learn')}",
        f"shap: {_package_version('shap')}",
        f"imbalanced-learn: {_package_version('imbalanced-learn')}",
        f"pandas: {_package_version('pandas')}",
        f"numpy: {_package_version('numpy')}",
        f"streamlit: {_package_version('streamlit')}",
        f"plotly: {_package_version('plotly')}",
        f"OS: {platform.platform()}",
        f"CPU: {_cpu_model()}",
        f"Cores: {os.cpu_count()}",
        f"RAM: {ram_text}",
        "",
        "XGBoost hyperparameters (model.get_params() on the held-out training fit):",
    ]
    for name in PLOT_ORDER:
        lines.append(f"[{name}]")
        params = results[name]["model"].get_params()
        for key in sorted(params):
            lines.append(f"  {key}: {_fmt_param(params[key])}")
        lines.append("")
    lines.append(_cri_block())
    text = "\n".join(lines).rstrip() + "\n"
    path = os.path.join(OUT_DIR, "environment_info.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(text)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    font = _use_serif_font()
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    print(f"Font: {font}")

    results = _load_eval_module().evaluate_held_out()
    missing = [name for name in PLOT_ORDER if name not in results]
    if missing:
        raise RuntimeError(f"Held-out results missing: {missing}")

    created = []
    created.extend(plot_roc(results))
    created.extend(plot_calibration(results))
    threshold_path, rows = diabetes_thresholds(results["Diabetes"]["y_test"], results["Diabetes"]["probs"])
    created.append(threshold_path)
    created.append(test_set_summary(results))
    created.append(environment_info(results))

    print("Diabetes operating points (held-out):")
    for rule, metrics in rows:
        print(
            f"  {rule}: threshold={metrics['threshold']:.4f}  "
            f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
            f"prec={metrics['precision']:.3f}  F1={metrics['f1']:.3f}  "
            f"acc={metrics['accuracy']:.3f}"
        )
    print("\nFiles created:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
