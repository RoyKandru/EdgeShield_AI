# URL detector v2 evaluation (auto-generated)

Config: {"max_train": 250000, "max_fpr_val": 0.005, "chosen": "char+lexical (v1)", "components": ["char", "lexical"], "val_pr_auc": {"char+lexical (v1)": 0.9684466370473285, "char+host+lexical (v2)": 0.9657431131897279}, "threshold_ml": 0.8561406074698348, "threshold_hybrid": 0.8561406074698348, "allowlist_size": 49512, "pool_by_source": {"corpus": 798727, "phishdb": 141224, "tranco_train": 98385}, "openphish_overlap_removed_domains": 255, "split_sizes": {"train": 250000, "val": 110520, "test": 209015}}

## Held-out mixed test (domain-grouped)

| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| ml_only | 0.9960 [0.996-0.996] | 0.6167 [0.614-0.620] | 0.7617 | 0.0024 [0.0021-0.0026] | 62689 | 254 | 107108 | 38964 |
| hybrid_allowlist+ml | 0.9982 [0.998-0.999] | 0.5948 [0.592-0.598] | 0.7454 | 0.0010 [0.0008-0.0012] | 60467 | 110 | 107252 | 41186 |
| rules_any_flag | 0.9659 [0.963-0.969] | 0.1063 [0.105-0.108] | 0.1916 | 0.0036 [0.0032-0.0039] | 10810 | 382 | 106980 | 90843 |

## Test split by source

- corpus_ml_only: `{"n": 158658, "precision": 0.9965462339252908, "recall": 0.57232890201871, "f1": 0.7270842641408276, "fpr": 0.00161008530026378}`
- corpus_hybrid: `{"n": 158658, "precision": 0.9972301261551633, "recall": 0.5571217556446507, "f1": 0.7148684994313976, "fpr": 0.0012560949150994029}`
- phishdb: `{"n": 30568, "positives": 30568, "flag_rate_ml_only": 0.7198704527610573, "flag_rate_hybrid": 0.6825438366919655}`
- tranco_train: `{"n": 19789, "positives": 0, "flag_rate_ml_only": 0.005710243064328668, "flag_rate_hybrid": 0.0}`

## External checks

```json
{
  "tranco_bare_domain": {
    "n": 10000,
    "note": "Tranco ranks 50,001-60,000 (disjoint from training/allowlist/threshold calibration)",
    "ml_only_fpr": 0.0188,
    "hybrid_fpr": 0.0188,
    "rules_fpr": 0.0
  },
  "tranco_https_root": {
    "n": 10000,
    "note": "Tranco ranks 50,001-60,000 (disjoint from training/allowlist/threshold calibration)",
    "ml_only_fpr": 0.0039,
    "hybrid_fpr": 0.0039,
    "rules_fpr": 0.0
  },
  "openphish_live": {
    "n": 300,
    "file": "openphish_20261122.txt",
    "ml_only_recall": 0.6866666666666666,
    "hybrid_recall": 0.68,
    "rules_recall": 0.04666666666666667,
    "fraction_on_allowlisted_domains": 0.023333333333333334
  }
}
```