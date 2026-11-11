# Team possession from automatic football tracking

V5 technical report, Barca4Ever, More Than A Hack 2027 Initial Challenge.

This is the browser-readable version. The submission paper is compiled from LaTeX. [PDF](submission/Barca4Ever.pdf) · [LaTeX source](submission/Barca4Ever.tex)

## Task and evaluation

We estimate home possession H, away possession A and dead ball D from player and ball tracking. The output has 199,319 rows, one for each tactical-video frame. Long complete player-tracking outages receive X.

The ranking metric, spellF1, matches predicted and reference possession spells one-to-one within the same team by temporal IoU. A spell is a maximal uninterrupted H or A interval within a scored reference span. F1 is averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. Dead-ball spells are excluded. Unknown reference spans are excluded. Predicted X values are forward-filled, with leading abstentions back-filled within a valid span. Frame-level macroF1 breaks ranking ties.

Short recovery is a separate count of possessions shorter than two seconds matched at IoU 0.5. It is distinct from the multi-threshold headline score.

## Method

V5 combines a 59-feature geometry classifier with a trajectory network inspired by Borghesi et al. (2023). We implemented and trained the branch ourselves. It predicts dead probability and conditional home probability from provider possession and ball-status labels, without touch-level supervision.

Geometry and neural probabilities receive equal weight. A directed transition classifier estimates the six changes between H, A and D. Viterbi decoding maximizes summed log possession probabilities minus the switch cost 16/(1+8*b), where b is the score for the specific source/destination pair. The PDF gives the full equations and pipeline figure.

Source tracks use 25 Hz. Neural inputs use 20 channels at 16 Hz, with 64 samples and zero-based target index 47. The window includes about one second of future trajectories. Observation-gap features can use ten seconds of future evidence, so the method is offline.

The loader maps source record 1 to output frame 0 and converts normalized positions to pitch-centred metres. All observed actors are kept in padded team groups and missing actors are masked. Goalkeeper-role inputs stay zero. Ball gaps up to two seconds are interpolated between observed endpoints, while raw missingness is retained. Longer ball gaps stay missing. Complete player outages of at least ten seconds split inference blocks and receive X.

The implementation uses tactical XY tracking. Video, Smart Tagging, broadcast tracking and homography are outside this model. Detailed settings, training seeds and checkpoint lineage are in the model guide. [V5 model guide](V5_MODEL_GUIDE.md)

## External data and selection

The deployed artifact was fitted on IDSSE, using DFL/Sportec BallPossession and BallStatus references under CC BY 4.0. Inactive status maps to D, live possession 1/2 to H/A, and unknown to X. Labels are isolated from features.

| Role | Whole matches |
| --- | --- |
| Final fitting | J03WOH, J03WOY, J03WPY, J03WQQ |
| Selection | J03WR9 |
| Reused regression | J03WMX, J03WN1 |

Cross-validation within the four fitting matches trains on WOH/WOY and evaluates WPY/WQQ, then reverses. Every learned fold component excludes its evaluated matches. Both halves stay with their match. All seven matches have been inspected repeatedly during development, including the original regression holdouts.

The selected composite had the highest WR9 spellF1 among candidates passing their predeclared cross-validation and regression checks. Neural fine-tuning used two epochs of 100,000 fresh references, sampled 75% by eligible frames and 25% by team spells. [Selection protocol](MULTIAGENT_SELECTION_PROTOCOL.md) · [Development experiments](MULTIAGENT_RESULTS.md)

## Results and limitations

V4.1 is the preceding geometry/trajectory pipeline with destination-based boundary scores. V5 adds raw observation confidence, elapsed-time velocity correction, fresh training references and directed transitions. These comparisons assess complete pipelines.

| Comparison | View | V4.1 spellF1 | V5 spellF1 | V4.1 short | V5 short |
| --- | --- | --- | --- | --- | --- |
| Whole-match CV | Clean | 0.7105 | 0.7133 | 191/463 | 206/463 |
| Whole-match CV | Noisy | 0.5329 | 0.5344 | 79/463 | 80/463 |
| Reused regression | Clean | 0.6808 | 0.6847 | 128/362 | 132/362 |
| Reused regression | Noisy | 0.4968 | 0.5140 | 64/362 | 71/362 |

Scores average halves equally and short hits are pooled. Synthetic corruption adds 0.25 m player jitter, 0.35 m ball jitter, 5% player dropout and ball gaps of 15-84 source frames. Clean and noisy runs share labels.

The largest difference is in noisy reused regression. The noisy cross-validation gain is only 0.0015, with one additional short possession recovered. These are repeated development comparisons on one provider. They provide limited independent evidence, and competition performance awaits the organizer's private evaluation.

Noisy short possessions remain difficult. Fragmented identities, missing ball observations, body orientation, aerial contacts, whistles and provider possession definitions limit transfer. Exact nearest-player ties retain the trained slot-order convention.

## Reproduction

From the repository root, use Python 3.12 and the local organizer bundle.

```text
python -m pip install -r requirements-contact.txt
python refined_infer.py --model models/external_v5.joblib --data-dir /local/bundle --out local_outputs/Barca4Ever.csv
python evaluation/validate_submission.py local_outputs/Barca4Ever.csv
```

The bundle contains tactical/BAR-ATH_tactical_atd.xml and its TXT tracking file. The artifact is included, CPU inference uses four threads, and the local manifest records model/output hashes. The November 11 release passed 85 synthetic tests and a full-length generated-data inference/format check. Historical evidence includes the original 82-test suite and four external CPU parity comparisons. These are software checks. [Verification summary](external_reports/v5_software_verification_2026_11_11.json)

All fitting, selection and quality evaluation use external labels. The supplied match is used for local inference. Challenge assets, predictions and logs stay local. The organizers prohibit uploading their data or extracts to hosted AI services.

## References

- Matteo Borghesi, Lorenzo Dusty Costa, Lia Morra and Fabrizio Lamberti. Using Temporal Convolutional Networks to estimate ball possession in soccer games. Expert Systems with Applications 223, 119780 (2023). [Paper](https://doi.org/10.1016/j.eswa.2023.119780)
- Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert. An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data 12, 195 (2025). [Data paper](https://doi.org/10.1038/s41597-025-04505-y)
- IDSSE, original data from DFL / Sportec Solutions, under CC BY 4.0. Preserve attribution with trained artifacts. [Dataset](https://doi.org/10.6084/m9.figshare.28196177) · [License](https://creativecommons.org/licenses/by/4.0/)
