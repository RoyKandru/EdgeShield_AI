# Detector evaluation (auto-generated 2026-11-22T13:10:39+00:00)

Operating point: best validation F1 with validation FPR <= 1%. Positive class = malicious. FPR = FP/(FP+TN). 95% CIs: 300-sample bootstrap over test items.

## SMS
Dataset: UCI SMS Spam Collection (CC BY 4.0); stratified random 70/10/20 on de-duplicated messages.

Test items: 1032 (129 malicious). Model: tfidf(word1-2 + char_wb2-5) + LogisticRegression, threshold 0.2999.

| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| ml_tuned_threshold | 0.9766 [0.945-1.000] | 0.9690 [0.933-0.993] | 0.9728 | 0.0033 [0.0000-0.0079] | 125 | 3 | 900 | 4 |
| ml_threshold_0.5 | 0.9837 [0.959-1.000] | 0.9380 [0.890-0.969] | 0.9603 | 0.0022 [0.0000-0.0056] | 121 | 2 | 901 | 8 |
| rules_any_flag | 1.0000 [1.000-1.000] | 0.0698 [0.032-0.114] | 0.1304 | 0.0000 [0.0000-0.0000] | 9 | 0 | 903 | 120 |
| rules_strong_flag | 0.0000 [0.000-0.000] | 0.0000 [0.000-0.000] | 0.0000 | 0.0000 [0.0000-0.0000] | 0 | 0 | 903 | 129 |
| HYBRID_shipped | 0.9766 [0.945-1.000] | 0.9690 [0.933-0.993] | 0.9728 | 0.0033 [0.0000-0.0079] | 125 | 3 | 900 | 4 |

- ml_tuned_threshold: ROC-AUC 0.9988, PR-AUC 0.9938
- ml_threshold_0.5: ROC-AUC 0.9988, PR-AUC 0.9938
- rules_any_flag: ROC-AUC 0.6144, PR-AUC 0.3185
- rules_strong_flag: ROC-AUC 0.6144, PR-AUC 0.3185
- HYBRID_shipped: ROC-AUC 0.9988, PR-AUC 0.9938

## URL
Dataset: ealvaradob/phishing-dataset (urls.json) (Apache-2.0); 70/10/20 grouped by registered domain (no domain in two splits); scheme and 'www.' stripped before features.

Test items: 177671 (69119 malicious). Model: ensemble, threshold 0.7204.

| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| ml_tuned_threshold | 0.9654 [0.964-0.967] | 0.7079 [0.705-0.711] | 0.8168 | 0.0162 [0.0155-0.0169] | 48930 | 1756 | 106796 | 20189 |
| ml_threshold_0.5 | 0.8732 [0.871-0.876] | 0.9151 [0.913-0.917] | 0.8937 | 0.0846 [0.0829-0.0862] | 63253 | 9182 | 99370 | 5866 |
| rules_any_flag | 0.9510 [0.947-0.956] | 0.1127 [0.111-0.115] | 0.2015 | 0.0037 [0.0033-0.0040] | 7787 | 401 | 108151 | 61332 |
| rules_strong_flag | 0.9882 [0.970-1.000] | 0.0024 [0.002-0.003] | 0.0048 | 0.0000 [0.0000-0.0000] | 167 | 2 | 108550 | 68952 |
| HYBRID_shipped | 0.9653 [0.964-0.967] | 0.7079 [0.705-0.711] | 0.8168 | 0.0162 [0.0155-0.0169] | 48930 | 1757 | 106795 | 20189 |

- ml_tuned_threshold: ROC-AUC 0.9741, PR-AUC 0.9620
- ml_threshold_0.5: ROC-AUC 0.9741, PR-AUC 0.9620
- rules_any_flag: ROC-AUC 0.6347, PR-AUC 0.5483
- rules_strong_flag: ROC-AUC 0.6347, PR-AUC 0.5483
- HYBRID_shipped: ROC-AUC 0.9741, PR-AUC 0.9618

### External sanity sets (independent of training corpus)

```json
{
  "tranco_bare_domain": {
    "n": 10000,
    "ml_fpr": 0.0973,
    "rules_fpr": 0.0,
    "note": "all treated as benign; ranks 50001-60000 of Tranco list"
  },
  "tranco_https_root": {
    "n": 10000,
    "ml_fpr": 0.009,
    "rules_fpr": 0.0,
    "note": "all treated as benign; ranks 50001-60000 of Tranco list"
  },
  "openphish_live": {
    "n": 300,
    "ml_recall": 0.44,
    "rules_recall": 0.04666666666666667,
    "sha256": "c1dab8b4f19812c4473fc420121ac3a5b4b18610192a77099797b6a5baa1454e",
    "fetched": "2026-11-22T13:10:15.811548+00:00"
  }
}
```

### SMS cross-corpus transfer (model trained on UCI only; threshold NOT re-tuned)
CloveAI/india-spam-sms (templated; indicative only); 19230 unique rows, 5558 malicious.

| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| ml_uci_threshold | 0.4010 [0.395-0.409] | 0.8989 [0.890-0.907] | 0.5546 | 0.5458 [0.5380-0.5533] | 4996 | 7462 | 6210 | 562 |
| rules_any_flag | 0.8528 [0.840-0.865] | 0.3555 [0.344-0.366] | 0.5018 | 0.0249 [0.0225-0.0269] | 1976 | 341 | 13331 | 3582 |
| rules_strong_flag | 1.0000 [1.000-1.000] | 0.0356 [0.032-0.040] | 0.0688 | 0.0000 [0.0000-0.0000] | 198 | 0 | 13672 | 5360 |
| HYBRID_shipped | 0.4010 [0.395-0.409] | 0.8989 [0.890-0.907] | 0.5546 | 0.5458 [0.5380-0.5533] | 4996 | 7462 | 6210 | 562 |

## NETWORK (CIC-IDS2017, per-flow)
Flows: 2300825. Features: packets, bytes, duration_s, dst_port, syn_count, bytes_per_pkt, pkts_per_s, bytes_per_s, syn_ratio, sensitive_port, well_known_port, ephemeral_port.

### A_random_split
stratified random 70/10/20 over all days. In-distribution; flows from the same attack burst are near-duplicates, so this is an UPPER bound.

Test flows: 460166 (111529 attack), threshold 0.5723

| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| ml_tuned_threshold | 0.9919 [0.991-0.992] | 0.9875 [0.987-0.988] | 0.9897 | 0.0026 [0.0024-0.0028] | 110134 | 898 | 347739 | 1395 |
| ml_threshold_0.5 | 0.9913 [0.991-0.992] | 0.9882 [0.988-0.989] | 0.9897 | 0.0028 [0.0026-0.0030] | 110211 | 970 | 347667 | 1318 |
| rules | 0.0000 [0.000-0.000] | 0.0000 [0.000-0.000] | 0.0000 | 0.0001 [0.0001-0.0001] | 0 | 30 | 348607 | 111529 |
| HYBRID_shipped | 0.9916 [0.991-0.992] | 0.9875 [0.987-0.988] | 0.9896 | 0.0027 [0.0025-0.0028] | 110134 | 928 | 347709 | 1395 |

Per-family recall (ML tuned / rules):

- Bot (n=428): 0.381 / 0.000
- DDoS (n=25549): 0.999 / 0.000
- DoS_GoldenEye (n=2065): 0.930 / 0.000
- DoS_Hulk (n=46192): 0.992 / 0.000
- DoS_Slowhttptest (n=1058): 0.983 / 0.000
- DoS_slowloris (n=1110): 0.977 / 0.000
- FTP-Patator (n=1559): 0.997 / 0.000
- Heartbleed (n=4): 0.500 / 0.000
- Infiltration (n=9): 0.444 / 0.000
- PortScan (n=31901): 0.999 / 0.000
- SSH-Patator (n=1184): 0.986 / 0.000
- Web_Attack___Brute_Force (n=330): 0.058 / 0.000
- Web_Attack___Sql_Injection (n=6): 0.167 / 0.000
- Web_Attack___XSS (n=134): 0.015 / 0.000

### B_train_Tue-Thu_test_Fri
train on Tuesday-Thursday, test on Friday. Bot, PortScan and DDoS are NEVER seen in training: measures generalisation to new attack families.

Test flows: 703245 (288923 attack), threshold 0.5132

| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| ml_tuned_threshold | 0.9840 [0.983-0.985] | 0.3589 [0.357-0.361] | 0.5260 | 0.0041 [0.0039-0.0043] | 103693 | 1685 | 412637 | 185230 |
| ml_threshold_0.5 | 0.9840 [0.983-0.985] | 0.3589 [0.357-0.361] | 0.5260 | 0.0041 [0.0039-0.0043] | 103693 | 1686 | 412636 | 185230 |
| rules | 0.0000 [0.000-0.000] | 0.0000 [0.000-0.000] | 0.0000 | 0.0001 [0.0000-0.0001] | 0 | 27 | 414295 | 288923 |
| HYBRID_shipped | 0.9838 [0.983-0.985] | 0.3589 [0.357-0.361] | 0.5259 | 0.0041 [0.0039-0.0043] | 103693 | 1712 | 412610 | 185230 |

Per-family recall (ML tuned / rules):

- Bot (n=1966): 0.000 / 0.000
- DDoS (n=128027): 0.806 / 0.000
- PortScan (n=158930): 0.003 / 0.000
