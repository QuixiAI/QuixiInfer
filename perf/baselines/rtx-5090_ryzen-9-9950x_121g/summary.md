# baseline-setb (2026-11-06T07:22:36)

git 0dc3dbd + working tree 5a3b184bd7eb, host `rtx-5090_ryzen-9-9950x_121g`, driver 595.99.02, correctness pass

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode | 3/3 | **33.10** | 30.2 | 32.90 | 33.90 | 3.0% | 37.30 | 15.61 |
| prefill64 | 3/3 | **12.10** | 82.6 | 12.00 | 12.60 | 5.0% | 21.10 | 13.89 |
| long16k | 3/3 | **36.40** | 27.5 | 36.20 | 36.40 | 0.5% | 21.00 | 15.64 |
| long16k-prefill | 3/3 | **20.88** | 47.9 | 20.28 | 20.99 | 3.4% | 21.00 | 15.64 |
| serve1 | 3/3 | **40.49** | 24.7 | 40.49 | 42.02 | 3.8% | 0.00 | nan |
| serve3 | 3/3 | **95.24** | 10.5 | 94.34 | 95.24 | 0.9% | 0.00 | nan |

score (geomean of the case medians, ms/token): **34.327**
