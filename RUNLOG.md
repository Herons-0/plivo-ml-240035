# RUNLOG

All scores = official `score.py` metric: mean response delay (ms) @ ≤5% interrupted turns.
Model-development scores are **5-fold grouped out-of-fold (OOF) by turn** — no turn is ever
scored by a model that saw it. In-sample scores are marked as such and not used for decisions.

| # | run | english | hindi | change / why |
|---|-----|---------|-------|--------------|
| 1 | silence-only baseline (given) | **1600** | **850** | Reference. Hindi holds are short (p90 0.63s) so a fixed 850ms timer already survives the 5% budget; English holds reach 3.0s, forcing 1600ms. AUC ≈ 0.5 both. |
| 2 | v1 prosody: 20 features (F0 tail/slope, energy slope, rate, turn context) + LR/GBM | 1270 | 850 | First learned model (GBM). AUC en .64 / hi .67. Weak: listened to feature values — tail windows anchored at `pause_start` were full of silence. |
| 3 | **v2 anchor fix**: found `pause_start` LAGS the true speech offset (some "pre-pause" windows were pure silence, energy −108dB). Re-anchored every tail feature at the last real speech frame (energy VAD); octave-folded + median-filtered F0 | 1222 (GBM, dur-weighted) | 850 | AUC hi jumped .67→.74. Biggest single insight of the session. |
| 4 | + log-mel tail block: 40-mel × last 1s of speech, causal CMN, pooled stats (240 dims) | 1305 | 850 | AUC en .71. Mel pooling adds a little; flattened-mel PCA did not help (496 samples). |
| 5 | metric-aware training: drop holds with dur <0.4s from TRAIN only (they can never cause a cutoff at delay ≥ dur) | 1264 | **792** (MLP) | Best Hindi so far. Uses label-side info only — not a feature; causality intact. |
| 6 | + within-turn delta features (tail pitch/energy/lengthening vs previous pauses of same turn, "new low" flag) | 1278 | 850 | AUC hi .738. Deltas helped AUC but not the operating point; kept (cheap, interpretable). |
| 7 | per-language training test | 1301 | 865 | WORSE — 248 samples/language is too few; cross-language transfer wins. Train on both, add a language flag from filename (defaults to Hindi for unknown ids — hidden set is mostly Hindi). |
| 8 | ensemble without train filter: LR + GBM(d3) + GBM(d2) + MLP, prob-averaged, + lang flag | 1353 | 850 | Ensemble over single GBM (1315/850) for variance on unseen data; run-to-run differences are within top-few-long-holds noise (~±80ms). |
| 9 | row 8 + row 5 combined: same ensemble, holds <0.4s dropped from training | 1282 | 826 | AUC en .663 / hi .699. Both languages below baseline. |
| 10 | **human listening round** (worst OOF errors): en__091's 3.0s "hold" is a THUD + silence before the speaker's first word; en__007 ends on "and…" (thinking); en__018 ends on a flat-inflection date; en__072/hi__016 are normal assertive endings the model under-scores | — | — | Direct outcome: pre-speech guard idea (row 11). Conjunction/date holds confirmed lexical — out of reach without ASR (banned). |
| 11 | **FINAL: + thud guard + speech-evidence features** (speech_time, n_speech_runs, hf_ratio). Guard: single burst <0.6s with <10% energy above 1kHz ⇒ p_eot ≤ 0.02 (a thud rings periodically and fools the pitch tracker, but has no high-frequency speech content) | **1255** | **840** | Guard catches 7 english pre-speech noise holds incl. en__091 (en 1294→1255); costs 2 quiet one-word eots (en__084, hi__062 → hi 830→840). Kept: for a live agent this failure mode is a slower response, never an interruption. Shipped as `eot_model.pkl`. |
| 12 | shipped predictions_{english,hindi}.csv via predict.py | 264 (in-sample) | 375 (in-sample) | Inflated — the final model trains on all provided turns (in-sample AUC ~0.94–1.0 is memorization, NOT a claim). Honest expectation on unseen data = row 11: ~1255 en / ~840 hi. |

## Second research round (post-deadline-pressure, literature-informed)

All numbers below are 5-fold grouped OOF, same protocol. Shipped model unchanged (row 11) —
these runs informed the roadmap but were not integrated under submission time constraints
(the winning config needs extra inference-time models and a ~0.3 s/pause pitch tracker).

| # | run | english | hindi | change / why |
|---|-----|---------|-------|--------------|
| 13 | future-VAD aux model v1 (VAP-style, Ekstedt & Skantze 2022): self-supervised "speech resumes within 300/600/1200 ms?" from causal pooled-mel at 142k frames (fold-safe: aux nets never see eval-fold files); outputs at speech offset + pause_start added as 6 features | 1234 | 853 | Standalone aux AUC only 0.51–0.54 — pooled context genuinely lacks the cue — but as features: best english so far. |
| 14 | aux v2: offset-region anchors only + TEMPORAL 8×120 ms sliced-mel input (88k frames) | 1236 | 852 | Standalone AUC up (hi 0.63 @pause_start) yet ensemble saturated. |
| 15 | time-sliced mel as direct pause features (360 dims) | 1277 | 856 | Overfits 496 samples — dropped. |
| 16 | librosa yin F0 on anchored tails (better tracker; 0.3 s/pause) | 1249 | 840 | Strongest single hindi features yet (slope-800ms AUC 0.676) but ensemble flat. |
| 17 | within-turn stage-2 recalibration (p vs running max/mean of previous pauses' p — causal) | 1222 | 844 | Small consistent english gain. |
| 18 | single GBM over ALL blocks (hand+mel+aux1+aux2+yin) replaces 4-model average | **1175** | 857 | LR/MLP heads were dragging the average. 3-seed bagged: identical. +stage-2 blend: **1159**. |
| 19 | nested per-language head-weight selection (leave-one-fold-out) | 1209 | 857 | Weights unstable across folds — small-sample noise, not signal. |

**Why hindi is pinned at ~850:** the operating point is frozen by ~8 turns whose long holds
score 0.82–0.94 across every model; human listening (hi__051, hi__036) finds no audible cue
— the speaker just pauses mid-sentence with neutral prosody. This is lexical-completion
information (ASR is banned). English retains headroom (best honest config 1159 ms) via the
aux/yin blocks; integrating them into predict.py (ship aux models + yin at inference) is the
first thing we would do with one more day.

**Where it still fails:** confident cue-less long holds (above); fast mid-pitch-range
endings (phone numbers). The full second-round code (aux_vad.py, aux2.py, yin_tail.py)
is reproducible and fold-safe.
