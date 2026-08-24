"""
Loads Optuna-tuned XGBoost hyperparameters (see scripts/tune_hyperparameters.py)
on top of each model's hand-picked defaults.

Tuning is opt-in: configs/tuned_hyperparams.json is not required to exist, and
a model whose key is missing from it just keeps its defaults. Monotonic
constraints are never read from this file — they encode a clinical
correctness requirement, not a performance knob, and are always set
separately by each model.
"""
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "tuned_hyperparams.json")


def load_tuned(model_key: str, defaults: dict) -> dict:
    """Return `defaults` overridden by any tuned values found for `model_key`."""
    if not os.path.exists(CONFIG_PATH):
        return dict(defaults)
    with open(CONFIG_PATH) as f:
        tuned = json.load(f).get(model_key, {})
    tuned = {k: v for k, v in tuned.items() if not k.startswith("_")}
    return {**defaults, **tuned}
