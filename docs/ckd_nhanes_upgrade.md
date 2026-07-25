# CKD data upgrade: UCI (400 rows) → NHANES (both sexes, thousands)

The shipped CKD model currently trains on the 400-row UCI set, which shows
near-perfect separation on its full lab panel — a tiny, over-optimistic,
single-source dataset. This guide replaces it with an NHANES-derived cohort.
The hard part (eGFR, the CKD definition, module joins, schema mapping) is
already implemented and unit-tested in `scripts/build_ckd_nhanes.py`.

## Why it wasn't done automatically
NHANES XPT files are served from `wwwn.cdc.gov`. Some execution environments
(including the sandbox this repo was last edited in) block `cdc.gov` and the
GitHub API by egress policy, so the files can't be fetched there. Download them
from any machine with normal internet, or in an environment whose network
policy allows CDC.

## Step 1 — fetch the NHANES modules (2017-2018 cycle)
```bash
mkdir -p data/raw/nhanes/2017-2018 && cd data/raw/nhanes/2017-2018
for f in DEMO_J BMX_J BPX_J BIOPRO_J TCHOL_J GLU_J SMQ_J ALB_CR_J; do
  curl -sSLO "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/$f.XPT"
done
cd -
```
(Pool more cycles by adding them to `CYCLES` in the build script — e.g.
`2015-2016` with suffix `_I`.)

## Step 2 — build the cleaned dataset
```bash
python scripts/build_ckd_nhanes.py            # -> data/processed/ckd_nhanes_clean.csv
python scripts/build_ckd_nhanes.py --selftest # optional: verify the eGFR math
```
CKD is labelled positive when **eGFR < 60 mL/min/1.73m²** (race-free CKD-EPI
2021 from serum creatinine) **or** **urine albumin/creatinine ratio ≥ 30 mg/g**
(KDIGO markers). NHANES is a single visit, so this flags CKD *markers*, not the
3-month persistence a clinical diagnosis needs — keep that caveat in the docs.

## Step 3 — point the CKD model at the new data
In `src/models/ckd_model.py`:
- `data_path = os.path.join(DATA_PROCESSED, "ckd_nhanes_clean.csv")`
- Use a **checkup-safe** feature set only — the lab columns define the label and
  would leak, exactly like HbA1c does for diabetes:
  `checkup_safe_features = ['age', 'sex', 'bmi', 'systolic_bp', 'diastolic_bp', 'glucose', 'cholesterol', 'smoking']`
- Drop the UCI-specific `baseline_features` (sg/al/su/sc/…); if you want a
  "with-labs" baseline for comparison, build it from `serum_creatinine/egfr/acr`
  and label it clearly as leaky/illustrative.
- Median-impute as today (fasting glucose is a subsample, so it will be partly
  missing).

## Step 4 — retrain and refresh everything
```bash
python scripts/retrain_all.py          # CKD + downstream re-chain on new ckd_risk
python scripts/generate_eval_cache.py  # leakage-free held-out curves
python src/chaining/chain_engine.py    # refresh the external ablation
```
Then update the CKD numbers in `reports/baseline_metrics.md`, the dashboard
metric tables, and the README, from `reports/model_metrics.json` (same flow used
for the diabetes upgrade).

## Expected outcome
A both-sex CKD screening model on thousands of adults, with a **realistic**
AUC (no perfect separation), predicting CKD markers from routine checkup
features only — a far more honest upstream model for the chain than the 400-row
UCI set.
