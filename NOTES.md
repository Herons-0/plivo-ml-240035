# NOTES

The model reads the last second of *actual speech* before each pause — the key fix was
discovering that `pause_start` lags the true speech offset, so every tail feature is
re-anchored at the last energy-detected speech frame. Signals used: final pitch level and
slope (semitones, per-speaker normalized), energy decay into the pause, final-voiced-run
lengthening, tail voicing fraction, spectral-centroid trail-off, pooled log-mel tail
statistics, plus causal turn context (pause index, time so far, previous-pause durations,
and deltas of this tail vs earlier tails of the same turn). Training drops holds shorter
than 0.4s because they can never cause a false cutoff — capacity goes to the only
distinction the metric pays for: true ends vs LONG hesitations. A human listening round
found en__091's 3-second "hold" is a thud before the speaker's first word; that became a
shipped rule: a prefix whose only speech evidence is one short low-frequency burst can
never be an end of turn (p_eot ≤ 0.02). An ensemble (LR + 2 GBMs + MLP) trained on both
languages beats per-language models (248 samples each is too few), giving grouped-OOF
1255ms English and 840ms Hindi vs the 1600/850 silence baseline (AUC 0.66/0.70).
Verified remaining failures: holds after trailing conjunctions ("and…") and
flat-inflection dates — lexical completion that prosody cannot see (ASR is banned) —
plus fast, mid-pitch-range endings like phone numbers. With one more day: a
formant-stability filler detector, pYIN-quality pitch fast enough to ship (4s/pause was
not), a small 1D-CNN over the mel tail with augmentation, and hazard-style calibration
over pause index.
