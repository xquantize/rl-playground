# ppo-ablations

From-scratch PPO with one implementation detail removed at a time.

| Variant | Seeds | Eval laps | Crash rate | Mean training laps | vs first | Steps to 5 laps |
| :-- | --: | --: | --: | --: | --: | --: |
| full | 3 | 4.08 ± 4.41 | 67% ± 47% | 8.14 ± 0.10 | – | 81k ± 0 |
| no-advantage-norm | 3 | 10.26 ± 0.05 | 0% ± 0% | 8.40 ± 0.18 | +3% | 90k ± 0 |
| no-orthogonal-init | 3 | 7.96 ± 3.30 | 27% ± 38% | 7.92 ± 0.11 | -3% | 84k ± 3k |
| no-grad-clipping | 3 | 10.02 ± 0.42 | 3% ± 5% | 8.30 ± 0.11 | +2% | 84k ± 3k |
| no-clipped-objective | 3 | 4.71 ± 4.05 | 67% ± 47% | 6.44 ± 1.94 | -21% | 76k ± 7k |
| no-truncation-bootstrap | 3 | 9.86 ± 0.45 | 3% ± 5% | 8.32 ± 0.05 | +2% | 84k ± 3k |
