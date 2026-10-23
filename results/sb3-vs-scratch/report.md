# sb3-vs-scratch

Stable-Baselines3 PPO against the from-scratch implementation, identical hyperparameters.

| Variant | Seeds | Eval laps | Crash rate | Mean training laps | vs first | Steps to 5 laps |
| :-- | --: | --: | --: | --: | --: | --: |
| sb3 | 3 | 4.10 ± 4.37 | 67% ± 47% | 8.07 ± 0.13 | – | 84k ± 3k |
| scratch | 3 | 4.08 ± 4.41 | 67% ± 47% | 8.14 ± 0.10 | +1% | 81k ± 0k |
