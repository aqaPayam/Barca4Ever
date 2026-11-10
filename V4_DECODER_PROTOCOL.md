# Frozen V4 decoder experiment

Before inspecting results, test six decoder settings: base switch cost 8, 12 or
16, crossed with boundary scale 4 or 8. V4's 16/4 is the unchanged control.
Keep every trained weight and the 0.5 neural/tree mixture fixed. Select by
mean organizer spell F1 on external WR9 clean/noisy halves, macro F1 breaking
ties. This is development tuning on previously inspected external data.

Motivation: a frozen V4 external validation diagnostic found 11 noisy short
spells matched by raw labels but lost after decoding. It also found cases where
one component's correct majority was lost by averaging; that is a separate
future hypothesis. The counters overlap and do not establish achievable gains.

Only if validation improves, evaluate the single selected setting against V4
in both original whole-match folds and reused WMX/WN1 regression. Retain each
fold's own trained role-free network, possession classifier and state detector;
no component may be fitted to an evaluated fold match. Use the original seeds.
Reproduce the baseline score exactly in every case before accepting results.

Promotion requires spell F1 improvement and no loss of short-spell hits in
both clean and noisy scenarios of both comparisons, plus mixed validation.
If this fails, preserve V4 and report the tradeoff. Do not choose another
setting after examining cross-validation or regression results. No competition
match, video or ground truth is accessed. No model is refitted in this trial.
