# Paper-based possession model: implementation and external comparison

Independent PyTorch implementation of Borghesi et al. (2023), ESWA 223,
119780: https://doi.org/10.1016/j.eswa.2023.119780. Original trained weights,
code and confidential data were unavailable. This tests a retrained
paper-based adaptation; it does not establish the authors original score.

## Implementation fidelity

The strongest published architecture (two branches, separate ball-query
self-attention, shared per-object trajectory TCN) is implemented directly,
rather than the earlier hand-feature TCN. It uses 64 timesteps at 16 Hz,
target offset 47, 36 object slots, 11 input channels, causal dilated
convolution blocks, batch normalization, dropout, and ELU 64/32-unit FFNs.
Home/away loss is zero on externally labelled dead frames. See
PAPER_2023_PROTOCOL.md for every unspecified setting and adaptation.

IDSSE lacks referee trajectories: a missing referee slot is retained.
Goalkeepers use external metadata. Missing participants are padded.
Predictions are interpolated back to all 25-Hz frames before scoring.

The paper recipe keeps clean training, Adam 1e-5 and decay 0.7 per epoch,
100,000 windows and validation-accuracy checkpoint selection. The adapted
recipe changes optimization to Adam 0.001/cosine, adds simulated noise,
team swaps and short-spell weights, and selects by validation spell F1.
The relative recipe also adds six ball-relative contact/motion channels,
scaled velocity inputs, short-gap interpolation, a target-frame geometry
MLP skip, distance-biased attention and 25% uniform-spell training references.
These are bundled original extensions, not components of the paper.

The rolefree recipe initializes from the contact model and fine-tunes two
epochs with keeper-role channels zero. This matches the available tactical
input and avoids guessed roles. Each fold is adapted only from its own
eight-epoch checkpoint and only on its own fitting matches. The final
role-free fusion is selected and checked separately from the role-based
research model. No supplied metadata contents or possession observations
were displayed in the local input compatibility check.

The published architecture does not prescribe our Viterbi decoder.
paper_raw is its original hierarchical threshold prediction; paper
adds a decoder chosen on external validation. State-boundary is an
independent tree-based extension of V3, not part of the paper.

## Results

Means weight evaluated halves equally. Short recall counts possessions
under two seconds recovered at IoU 0.5. Clean/noisy rows share labels.

| Evaluation | Tracking | Model | Spell F1 | Macro F1 | Short recovery |
|---|---|---|---:|---:|---:|
| validation | mixed | v3 | 0.5132 | 0.8857 | 40/188 (21.3%) |
| validation | mixed | paper | 0.1023 | 0.3975 | 3/188 (1.6%) |
| validation | mixed | paper_raw | 0.0616 | 0.3975 | 4/188 (2.1%) |
| validation | mixed | adapted | 0.1694 | 0.6420 | 3/188 (1.6%) |
| validation | mixed | relative | 0.5720 | 0.8937 | 38/188 (20.2%) |
| validation | mixed | rolefree | 0.5675 | 0.8934 | 31/188 (16.5%) |
| validation | mixed | fusion | 0.6446 | 0.9193 | 51/188 (27.1%) |
| validation | mixed | fusion_rolefree | 0.6371 | 0.9166 | 55/188 (29.3%) |
| validation | mixed | state_boundary | 0.5138 | 0.8880 | 38/188 (20.2%) |
| validation | mixed | boundary_blend | 0.5167 | 0.8883 | 38/188 (20.2%) |
| cross-validation | clean | fusion | 0.7192 | 0.9390 | 188/463 (40.6%) |
| cross-validation | noise | fusion | 0.5561 | 0.8822 | 85/463 (18.4%) |
| cross-validation | clean | relative | 0.6283 | 0.9087 | 141/463 (30.5%) |
| cross-validation | noise | relative | 0.5213 | 0.8599 | 74/463 (16.0%) |
| cross-validation | clean | v3 | 0.6245 | 0.9235 | 111/463 (24.0%) |
| cross-validation | noise | v3 | 0.4178 | 0.8392 | 61/463 (13.2%) |
| cross-validation | clean | fusion_rolefree | 0.7050 | 0.9364 | 171/463 (36.9%) |
| cross-validation | noise | fusion_rolefree | 0.5315 | 0.8773 | 68/463 (14.7%) |
| cross-validation | clean | rolefree | 0.6257 | 0.9092 | 120/463 (25.9%) |
| cross-validation | noise | rolefree | 0.4929 | 0.8555 | 50/463 (10.8%) |
| cross-validation | clean | state_boundary | 0.6326 | 0.9260 | 104/463 (22.5%) |
| cross-validation | noise | state_boundary | 0.4402 | 0.8448 | 64/463 (13.8%) |
| reused-test regression | clean | paper | 0.1063 | 0.5325 | 3/362 (0.8%) |
| reused-test regression | noise | paper | 0.0086 | 0.1868 | 0/362 (0.0%) |
| reused-test regression | clean | paper_raw | 0.0852 | 0.5332 | 4/362 (1.1%) |
| reused-test regression | noise | paper_raw | 0.0099 | 0.1862 | 0/362 (0.0%) |
| reused-test regression | clean | v3 | 0.5896 | 0.9139 | 81/362 (22.4%) |
| reused-test regression | noise | v3 | 0.4079 | 0.8279 | 46/362 (12.7%) |
| reused-test regression | clean | adapted | 0.2218 | 0.7124 | 6/362 (1.7%) |
| reused-test regression | noise | adapted | 0.1238 | 0.5388 | 3/362 (0.8%) |
| reused-test regression | clean | relative | 0.6151 | 0.9114 | 95/362 (26.2%) |
| reused-test regression | noise | relative | 0.4364 | 0.8458 | 38/362 (10.5%) |
| reused-test regression | clean | rolefree | 0.6082 | 0.9103 | 87/362 (24.0%) |
| reused-test regression | noise | rolefree | 0.4469 | 0.8476 | 39/362 (10.8%) |
| reused-test regression | clean | fusion | 0.6770 | 0.9330 | 111/362 (30.7%) |
| reused-test regression | noise | fusion | 0.5072 | 0.8637 | 61/362 (16.9%) |
| reused-test regression | clean | fusion_rolefree | 0.6705 | 0.9291 | 109/362 (30.1%) |
| reused-test regression | noise | fusion_rolefree | 0.4863 | 0.8607 | 53/362 (14.6%) |
| reused-test regression | clean | state_boundary | 0.5939 | 0.9150 | 82/362 (22.7%) |
| reused-test regression | noise | state_boundary | 0.4191 | 0.8329 | 44/362 (12.2%) |

## Interpretation limits

All seven IDSSE matches have appeared in development. Whole-match folds
exclude their evaluated matches from fitting, but architecture acceptance
uses development evidence. This is not nested validation or a new blind
test. The paper used a different private dataset and accuracy metric.
Beating this implementation here cannot establish superiority over the
original trained system, current professional products or all models.

No competition video, labels or tracking were used in these experiments.
Competition predictions remain local and cannot establish a quality score.

## Reproduction

Install requirements-paper2023.txt for GPU training, or use the CPU
requirements-contact.txt with the same torch version. The final model
and fold-specific training commands are in V4_MODEL_GUIDE.md. Run the
relative whole-match crosscheck before the role-free adaptation so each
fold has its own external-only initializer. Reproduce in a fresh experiment
directory; existing artifact/report guards preserve original evidence.
Data licence and attribution: EXTERNAL_DATA.md (IDSSE, CC BY 4.0).

Synthetic tests cover permutation invariance, loss masking, missingness,
target alignment, direct-window versus dense predictions and exact
destination-cost decoding. Training curves, seeds and artifact hashes
are retained with the models. Source hashes are in the JSON summary.
