# action-squashing

Clipping out-of-range Gaussian samples against squashing them with tanh.

| Variant | Seeds | Eval laps | Crash rate | Mean training laps | vs first | Steps to 5 laps |
| :-- | --: | --: | --: | --: | --: | --: |
| clipped | 3 | 4.08 ± 4.41 | 67% ± 47% | 8.14 ± 0.10 | – | 81k ± 0 |
| tanh | 3 | 10.28 ± 0.03 | 0% ± 0% | 8.07 ± 0.03 | -1% | 90k ± 0 |
