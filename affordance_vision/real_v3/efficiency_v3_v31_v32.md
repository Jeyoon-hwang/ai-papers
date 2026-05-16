# Efficiency analysis — v3 / v3.1 / v3.2

| Run | Samples | Train min | AUROC mean | AUROC max | s / 0.01 AUROC |
|---|---|---|---|---|---|
| v3 | 1,020 | 0.0 | 0.8424 | 0.8632 | 0.0 |
| v3.1 | 1,020 | 3.1 | 0.8416 | 0.8786 | 5.4 |
| v3.2 | 10,000 | 37.4 | 0.8572 | 0.8849 | 62.8 |

_s / 0.01 AUROC_ = training seconds divided by 100 × (mean AUROC − 0.5), the marginal training cost per percentage point of AUROC above random.