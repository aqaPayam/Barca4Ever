# Destination-specific change detection: external results

This is an original extension of V3, independent of the 2023 trajectory model.
It retains V2 possession probabilities but predicts background/start-H/start-A/
start-D rather than binary change/no-change. The 177 boundary features, 140
boosting iterations, 15 leaves, rate 0.08, L2 3 and positive weight 10 are retained.
The cost of switching to j is 16/(1+4*b[t,j]). The scores are uncalibrated.

Validation selected base16/scale4 from four predeclared configurations. Mean
spell F1 was 0.513826 versus V3 0.513153: essentially tied on one match.

| Evaluation | Tracking | Model | Spell F1 | Macro F1 | Short recovery |
|---|---|---|---:|---:|---:|
| cross-validation | clean | v3 | 0.6245 | 0.9235 | 111/463 |
| cross-validation | clean | state_boundary | 0.6326 | 0.9260 | 104/463 |
| cross-validation | noise | v3 | 0.4178 | 0.8392 | 61/463 |
| cross-validation | noise | state_boundary | 0.4402 | 0.8448 | 64/463 |
| reused-test regression | clean | v3 | 0.5896 | 0.9139 | 81/362 |
| reused-test regression | clean | state_boundary | 0.5939 | 0.9150 | 82/362 |
| reused-test regression | noise | v3 | 0.4079 | 0.8279 | 46/362 |
| reused-test regression | noise | state_boundary | 0.4191 | 0.8329 | 44/362 |

Overall spell F1 improves in all four aggregate comparisons, but short-spell
recall does not: clean cross-validation drops 111 -> 104 and noisy regression
drops 46 -> 44. This is a trade-off, not an across-the-board improvement.
Short recovery means under two seconds, correct team and interval IoU >= 0.5.

Both halves stay within their match. Folds fit WOH/WOY and evaluate WPY/WQQ,
then reverse; exact evaluated noise inputs are shared with V3. The cached V2
emission classifier was fitted only within each fold; the new boundary classifier
is also refitted within each fold. WR9 selects parameters. WMX/WN1 are reused
regression matches. All seven games have appeared in development; no fresh
blind-test, statistical-significance or professional-system superiority claim.

Only IDSSE external data were used, CC BY 4.0; see EXTERNAL_DATA.md. V3 and its
competition CSV remain preserved pending the complete model comparison.
