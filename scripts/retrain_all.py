"""
Retrain every model in dependency order and dump a metrics summary.

Order matters: the downstream Heart and Hypertension models chain on the
upstream Diabetes and CKD predictions, so the upstream models must be trained
(and their .pkl files written) before the downstream ones are trained.

Outputs:
  * models/*.pkl                 — refreshed model artifacts
  * reports/model_metrics.json   — machine-readable CV metrics for the docs/dashboard
"""
import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models import diabetes_model, ckd_model, heart_model, hypertension_model


def _norm(metrics):
    """Normalise a per-model result (tuple- or dict-shaped) to JSON-safe dicts."""
    def one(m):
        if isinstance(m, dict):
            acc, auc, f1, conf = m['accuracy'], m['auc'], m['f1'], m['confusion']
        else:  # (acc, auc, f1, conf) tuple
            acc, auc, f1, conf = m
        return {
            'accuracy': round(float(acc), 4),
            'auc': round(float(auc), 4),
            'f1': round(float(f1), 4),
            'confusion': [int(x) for x in conf.ravel().tolist()],
        }
    return {k: one(v) for k, v in metrics.items()}


def main():
    summary = {}

    print("\n########## 1/4  DIABETES (upstream) ##########")
    summary['diabetes'] = _norm(diabetes_model.train_and_evaluate())

    print("\n########## 2/4  CKD (upstream) ##########")
    summary['ckd'] = _norm(ckd_model.train_and_evaluate())

    print("\n########## 3/4  HEART DISEASE (chained) ##########")
    summary['heart'] = _norm(heart_model.train_and_evaluate())

    print("\n########## 4/4  HYPERTENSION (chained) ##########")
    summary['hypertension'] = _norm(hypertension_model.train_and_evaluate())

    out = os.path.join(ROOT, "reports", "model_metrics.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics summary written to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
