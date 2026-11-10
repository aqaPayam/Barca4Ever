# Observability and sampling-rate neural adaptation

This is an original extension informed by Borghesi2023 trajectory/attention,
Ferraz2025 temporally accurate contact/reception detection, and an audit of our
own preprocessing. Source research notes are recorded separately by the research
agent. No released weights trained on evaluated games are used. No competition
inputs, video, labels or prediction quality are examined or used for fitting.

## Declare before fitting

Two fixed recipes: 16Hz-observability and 25Hz-observability. Both retain the
64-sample input and target47, with16 future samples. Thus25Hz intentionally
uses2.56s trajectory context and0.64s future trajectory context, versus4s/1s at16Hz. Time-until-observation may use up to10s of future evidence, as an offline feature analogous to the V4 tree gap feature. The published
architecture's two attention branches remain unchanged; these are adaptations.

Keep the first17 channels/roles and append three channels broadcast to objects:
raw ball-observed flag, time since and until a raw ball observation, capped at
10s and divided by10. Existing filled-geometry visibility is retained. Ball
gaps are filled on the original25Hz source before sampling, using V2's fixed
2s short-gap rule. Velocities use actual source-index elapsed time rather than
assuming every rounded16Hz interval is exactly1/16s. No label interpolation.

Initialize from the corresponding frozen role-free17-channel checkpoint. Copy
old stem/instant-MLP weights and zero-pad only new input-channel columns. This
preserves the initial network function when fed the same old17 channels. All
other weights/buffers are retained. Keeper-role channel remains zero.

Fine-tune two epochs of100k windows, batch256, Adam1e-4 with cosine decay to3e-5,
seed20261116. Use fresh frame/spell references EACH epoch,75%uniform frames and
25%uniform team spells, the old clipped spell-duration loss weights, clean/noisy
external inputs and team swaps. Fixed old synthetic-corruption seeds are reused
to compare rates fairly; varying noise and identity churn are separate future
experiments, not silently mixed into this trial.

Fit base weights only on WOH/WOY/WPY/WQQ. On WR9, evaluate neural mixing weights
0.25/0.5/0.75 with unchanged V4.1 trees/state-boundary classifier and16/8 decoder.
Select the highest mean spell F1 (macro tie-break) among choices improving WR9
spell F1 and avoiding a short-hit decline against V4.1. If none qualifies,
preserve V4.1. Only the selected rate/mix proceeds to whole-match CV and reused
WMX/WN1 regression. Each CV fine-tune starts only from its own corresponding
frozen fold checkpoint, with the same fit-match list and no evaluated game in
either stage. All learned trees also remain fold-specific.

Promotion requires higher spell F1 and no loss of short hits in BOTH clean/noisy
CV and regression. No fallback is chosen after inspecting regression. All seven
games are reused development data, not fresh blind tests. Retain failures.
