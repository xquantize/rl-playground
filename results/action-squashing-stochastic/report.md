# action-squashing

Clipping out-of-range Gaussian samples against squashing them with tanh.

| Variant | Seeds | Eval laps | Crash rate | Mean training laps | vs first | Steps to 5 laps |
| :-- | --: | --: | --: | --: | --: | --: |
| clipped | 3 | 9.59 ± 0.43 | 17% ± 12% | 8.14 ± 0.10 | – | 81k ± 0 |
| tanh | 3 | 10.16 ± 0.02 | 0% ± 0% | 8.07 ± 0.03 | -1% | 90k ± 0 |
