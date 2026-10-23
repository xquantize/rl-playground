# ppo-ablations

From-scratch PPO with one implementation detail removed at a time.

| Variant | Seeds | Eval laps | Crash rate | Mean training laps | vs first | Steps to 5 laps |
| :-- | --: | --: | --: | --: | --: | --: |
| full | 3 | 9.59 ± 0.43 | 17% ± 12% | 8.14 ± 0.10 | – | 81k ± 0 |
| no-advantage-norm | 3 | 10.16 ± 0.04 | 3% ± 5% | 8.40 ± 0.18 | +3% | 90k ± 0 |
| no-orthogonal-init | 3 | 9.56 ± 0.83 | 13% ± 19% | 7.92 ± 0.11 | -3% | 84k ± 3k |
| no-grad-clipping | 3 | 10.22 ± 0.02 | 0% ± 0% | 8.30 ± 0.11 | +2% | 84k ± 3k |
| no-clipped-objective | 3 | 5.24 ± 4.00 | 63% ± 45% | 6.44 ± 1.94 | -21% | 76k ± 7k |
| no-truncation-bootstrap | 3 | 9.77 ± 0.09 | 3% ± 5% | 8.32 ± 0.05 | +2% | 84k ± 3k |
