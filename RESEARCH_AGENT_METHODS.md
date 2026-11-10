# Paper-driven implementation guidance

Reviewed primary papers and official code on 6 November 2026. No competition
input, video, ground truth or prediction contents were opened. Production code
was not edited. Detailed notes: PAPER_RESEARCH_NOTES_2026_11_06.md.

## Immediate experiments

1. **Native 25-Hz contact adaptation.** Our 25-to-16-Hz rounded sampling produces
   source time steps of 1/25 or2/25 seconds, while velocity is multiplied by16.
   Constant-speed motion therefore receives about0.64x/1.28x alternating estimates.
   Native25Hz avoids this approximation; a16Hz actual-delta-velocity control
   would isolate it. A64-frame window at25Hz spans2.56s, with0.64s future context,
   rather than the paper's4s/~1s. Declare this adaptation explicitly.

2. **Retain raw observation information.** `object_features` computes
   `raw_ball_seen` but supplies the network only post-interpolation missingness.
   Filled observations become indistinguishable from real ball measurements.
   Add a raw-observation/interpolation channel; zero-pad pretrained input weights
   for an initially function-preserving extension, then fine-tune externally.
   Do not simply replace the old missingness flag: `relative_features` would
   then remove otherwise useful filled geometry.

3. **Touch/reception and structured control/flight states.** Use local minima
   of player-ball distance, signed approach/recession slopes, acceleration and
   direction changes, pitch-boundary proximity, raw visibility and gap duration
   as native-rate boundary features. A later auxiliary path can distinguish
   player control, in-flight passes and outside/dead states, preserving the
   passing team's ownership during flight until opposing control is inferred.
   These are proposed adaptations requiring matched external checks.

## Paper and code findings

**Borghesi2023:** the shared per-object TCN and separate attention/FFN branches
are correctly present in our architecture. The paper specifies velocity as
coordinate differences divided by frame period but does not unambiguously
settle before/after-scaling units. Normalized velocity is therefore an assumption,
not a proven replication bug. Avoid calling our unspecified kernel/width/block
choices an exact replication.

Primary: https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf

**ELASTIC2026:** its contact candidates combine distance valleys, acceleration
peaks and boundary valleys. Published physical filters are3m player-ball distance
and4m ball height. Its full algorithm additionally needs a known event sequence,
so it is not a drop-in possession detector. Borrow soft physics features; our
XY adapter cannot reproduce its height filter. Official code is MPL-2.0 and
benchmark event data is explicitly CC BY4.0 with Sportec/DFL attribution.
The re-annotated matches WMX/WN1/WPY are already in our development data.

Paper: https://arxiv.org/html/2608.30227v1
Code/data licence: https://github.com/hyunsungkim-ds/elastic#license-and-attribution

**PathCRF2026:** promising ball-free fallback: player control self-loops,
player-to-player flight edges and outside states; Set Attention plus dynamic
masked CRF, with auxiliary sender/receiver/emission supervision. Official code
and weights are available under MPL-2.0. Its event F1 is not our spell F1.

Pretrained-checkpoint warning: `saved/140/args.json` specifies LOMOCV test idx6.
The split code sorts the seven match-ID files: test WR9, validation WQQ,
training WMX/WN1/WOH/WOY/WPY. This overlaps our regression and CV matches.
Retrain inside our folds for a matched comparison. Official preprocessing uses
true live-play episodes and skips short/NaN-possessor episodes; evaluation also
corrects episode endings and skips red-card phases. Those shortcuts must not
be inherited into full-match H/A/D/X inference or its claimed comparison.

Paper: https://arxiv.org/html/2602.12080v2
Code/licence: https://github.com/hyunsungkim-ds/pathcrf
Config: https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/saved/140/args.json
Split: https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/train.py
Dataset shortcuts: https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/dataset.py
Evaluation: https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/evaluate.py

**Peral/Ferraz2025:** player-centric video tubes, ResNet50 temporal shifts and
ConvTasNet/TDNN-style possessor selection support a background/in-flight state
between consecutive players' touches. I found no official code/weights link
in the paper/author page. Transfer its task decomposition, not a claim of video
model reproduction. A video implementation needs licensed external video/data.

Primary: https://www.iri.upc.edu/files/scidoc/3004-Temporally-Accurate-Events-Detection-Through-Ball-Possessor-Recognition-in-Soccer.pdf

**PLOS2024 validation:** dependent tracking/event feeds can inflate apparent
event performance. This does not prove IDSSE leakage, but means our repeated
single-provider scores cannot establish cross-provider professional quality.

Primary: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298107
