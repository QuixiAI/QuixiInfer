# setd-chunk (2026-11-07T04:38:54)

git 0dad410 + working tree 580ba849bffb, host `rtx-5090_ryzen-9-9950x_121g`, driver 595.99.02, correctness pass

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode256k | 3/3 | **53.20** | 18.8 | 53.10 | 53.60 | 0.9% | 136.70 | 0.01 |
| prefill256k | 3/3 | **6.80** | 147.1 | 6.80 | 7.10 | 4.4% | 4.30 | 0.00 |
| serve256k1 | 3/3 | **42.92** | 23.3 | 42.57 | 47.85 | 12.3% | 0.00 | nan |
| serve256k3 | 3/3 | **112.99** | 8.9 | 111.23 | 113.51 | 2.0% | 0.00 | nan |
| decode | 3/3 | **35.40** | 28.2 | 35.20 | 38.20 | 8.5% | 45.70 | 18.76 |
| prefill128 | 3/3 | **11.40** | 87.7 | 11.10 | 11.60 | 4.4% | 27.30 | 16.69 |
| serve1 | 3/3 | **41.25** | 24.2 | 40.36 | 42.57 | 5.4% | 0.00 | nan |
| serve3 | 3/3 | **99.11** | 10.1 | 98.04 | 102.04 | 4.0% | 0.00 | nan |

score (geomean of the case medians, ms/token): **34.447**
