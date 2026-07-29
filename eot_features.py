"""Causal prosodic features for end-of-turn detection (v2).

CAUSALITY: every feature for a pause uses ONLY audio[0 : pause_start].
Per-turn normalizers (median pitch/energy, noise floor) come from that
same prefix, never from the full file. pause_end of the CURRENT pause is
never used. Boundaries of PREVIOUS pauses in the turn are past
information (they ended before pause_start) and are allowed.

v2 insight: labels' pause_start lags the true speech offset, so all tail
features are anchored at the last actual speech frame (energy-based),
with octave-corrected, median-filtered F0.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10
F0_FRAME_MS = 40
PREFIX_CAP_S = 12.0
HOP_S = HOP_MS / 1000.0


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def _frames(x, sr, frame_ms, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def _energy_db(x, sr):
    fr = _frames(x, sr, FRAME_MS)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def _f0_frame(f, sr, lo, fmin=60.0, thresh=0.30):
    f = f - f.mean()
    if np.max(np.abs(f)) < 1e-4:
        return 0.0
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(sr / lag) if ac[lag] >= thresh else 0.0


def _f0_contour(x, sr, fmax=400.0):
    fr = _frames(x, sr, F0_FRAME_MS)
    lo = int(sr / fmax)
    return np.array([_f0_frame(f, sr, lo) for f in fr], dtype=np.float32)


def _medfilt3(v):
    if len(v) < 3:
        return v
    out = v.copy()
    out[1:-1] = np.median(np.stack([v[:-2], v[1:-1], v[2:]]), axis=0)
    return out


def _octave_fold(f0, med):
    """Fold halving/doubling errors toward the speaker median."""
    f = f0.copy()
    nz = f > 0
    f[nz & (f > 1.6 * med)] /= 2.0
    f[nz & (f < med / 1.6)] *= 2.0
    return f


def _slope(y, dt):
    if len(y) < 3:
        return 0.0
    t = np.arange(len(y)) * dt
    t = t - t.mean()
    denom = (t ** 2).sum()
    return float((t * (y - y.mean())).sum() / denom) if denom > 0 else 0.0


def _runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def _centroid(x, sr):
    fr = _frames(x, sr, FRAME_MS)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    w = np.hanning(fr.shape[1]).astype(np.float32)
    sp = np.abs(np.fft.rfft(fr * w, axis=1))
    freqs = np.fft.rfftfreq(fr.shape[1], 1.0 / sr)
    return (sp * freqs).sum(axis=1) / (sp.sum(axis=1) + 1e-9)


BASE_NAMES = [
    # anchored at true speech end
    "gap_before_pause",        # pause_start minus last real speech (causal!)
    "e_end_rel", "e_slope_off", "e_slope_off2",
    "f0_end_st", "f0_slope_tail", "f0_slope_run", "f0_tail_med_st",
    "f0_range_tail", "f0_min_is_end",
    "final_run_dur", "final_run_rel", "tail_voiced_frac",
    "centroid_tail_rel", "flat_tail",       # filler-ish cues
    "burst_dur",                             # speech since previous pause
    # turn context (past only)
    "pause_index", "t_start", "prev_pause_max", "prev_pause_mean",
    "speech_frac", "syl_rate",
    # total voiced evidence so far — a pause before ANY real speech can
    # never be an end of turn (found by listening: en__091 is a thud +
    # silence before the speaker even starts)
    "voiced_total", "speech_time", "n_speech_runs", "hf_ratio",
]

# within-turn contrasts vs PREVIOUS pause tails (past info, causal).
# The human cue: "the voice dropped lower / trailed off more than at any
# earlier pause of this same speaker in this same turn."
DELTA_NAMES = [
    "d_f0_end_prev",     # tail pitch minus previous pause's tail pitch
    "d_f0_end_min",      # tail pitch minus MIN of previous tails
    "f0_new_low",        # 1 if lowest tail pitch so far
    "d_e_end_prev", "d_e_end_min",
    "d_slope_prev",      # f0 tail slope vs previous
    "d_run_prev",        # final lengthening vs previous
    "d_burst_prev",      # burst duration vs previous
]

FEATURE_NAMES = BASE_NAMES + DELTA_NAMES


def extract(x, sr, pause_start, pause_index, prev_pauses):
    f = dict.fromkeys(BASE_NAMES, 0.0)
    f["pause_index"] = float(pause_index)
    f["t_start"] = float(pause_start)
    if prev_pauses:
        durs = [e - s for s, e in prev_pauses]
        f["prev_pause_max"] = max(durs)
        f["prev_pause_mean"] = float(np.mean(durs))
        f["speech_frac"] = max(0.0, 1.0 - sum(durs) / max(pause_start, 1e-6))
    else:
        f["speech_frac"] = 1.0

    end = int(pause_start * sr)
    prefix = x[max(0, end - int(PREFIX_CAP_S * sr)):end]
    if len(prefix) < sr // 10:
        return f

    e = _energy_db(prefix, sr)
    floor = np.percentile(e, 10)
    loud = np.percentile(e, 95)
    speech_th = max(floor + 12.0, loud - 25.0)
    smask = e > speech_th
    if not smask.any():
        return f

    last_sp = int(np.max(np.nonzero(smask)))          # last speech frame idx
    f["gap_before_pause"] = (len(e) - 1 - last_sp) * HOP_S

    # how much actual speech has happened at all (thud-vs-word evidence)
    rs_all = [(a, b) for a, b in _runs(smask) if (b - a) * HOP_S >= 0.06]
    f["speech_time"] = sum((b - a) for a, b in rs_all) * HOP_S
    f["n_speech_runs"] = float(len(rs_all))
    fr_sp = _frames(prefix, sr, FRAME_MS)
    nn = min(len(fr_sp), len(smask))
    fr_sp = fr_sp[:nn][smask[:nn]]
    if len(fr_sp):
        w_h = np.hanning(fr_sp.shape[1])
        spp = np.abs(np.fft.rfft(fr_sp * w_h, axis=1)) ** 2
        fq = np.fft.rfftfreq(fr_sp.shape[1], 1.0 / sr)
        f["hf_ratio"] = float(spp[:, fq > 1000].sum() / (spp.sum() + 1e-12))

    # ---- tail = last 700 ms of actual speech ----
    tail_n = 70
    t0 = max(0, last_sp - tail_n + 1)
    e_tail = e[t0:last_sp + 1]
    f["e_end_rel"] = float(e[max(0, last_sp - 4):last_sp + 1].mean()
                           - np.median(e[smask]))
    f["e_slope_off"] = _slope(e[max(0, last_sp - 15):last_sp + 1], HOP_S)
    f["e_slope_off2"] = _slope(e[max(0, last_sp - 40):last_sp + 1], HOP_S)

    # ---- pitch on speech region ----
    f0_raw = _f0_contour(prefix, sr)
    n = min(len(f0_raw), len(e))
    f0_raw = f0_raw[:n]
    sp_f0 = f0_raw[:last_sp + 1]
    voiced_all = sp_f0[sp_f0 > 0]
    f["voiced_total"] = float(len(voiced_all)) * HOP_S
    if len(voiced_all) >= 5:
        med = float(np.median(voiced_all))
        f0c = _medfilt3(_octave_fold(sp_f0, med))
        st = lambda hz: 12.0 * np.log2(np.maximum(hz, 1e-3) / med)
        tail_f0 = f0c[t0:last_sp + 1]
        vt = tail_f0[tail_f0 > 0]
        f["tail_voiced_frac"] = float((tail_f0 > 0).mean())
        if len(vt) >= 3:
            f["f0_end_st"] = float(np.median(st(vt[-4:])))
            f["f0_tail_med_st"] = float(np.median(st(vt)))
            f["f0_slope_tail"] = _slope(st(vt), HOP_S)
            f["f0_range_tail"] = float(st(vt.max()) - st(vt.min()))
            f["f0_min_is_end"] = 1.0 if np.argmin(st(vt)) >= len(vt) - 3 else 0.0
            f["flat_tail"] = 1.0 if (len(vt) >= 8 and np.std(st(vt)) < 0.7
                                     and len(vt) * HOP_S > 0.25) else 0.0
            vmask = f0c[:last_sp + 1] > 0
            runs = _runs(vmask)
            if runs:
                ls, le = runs[-1]
                f["final_run_dur"] = (le - ls) * HOP_S
                run_durs = [(b - a) * HOP_S for a, b in runs]
                f["final_run_rel"] = f["final_run_dur"] / max(np.median(run_durs), 1e-3)
                f["f0_slope_run"] = _slope(st(f0c[ls:le][f0c[ls:le] > 0]), HOP_S)

    # ---- spectral centroid of tail vs speech median ----
    c = _centroid(prefix, sr)[:n]
    csp = c[smask[:len(c)]]
    if len(csp) > 10 and last_sp < len(c):
        ct = c[max(0, last_sp - 20):last_sp + 1]
        f["centroid_tail_rel"] = float(np.log1p(ct.mean()) - np.log1p(np.median(csp)))

    # ---- rhythm ----
    runs_sp = _runs(smask)
    if runs_sp:
        f["syl_rate"] = len(runs_sp) / max(len(e) * HOP_S, 1e-3)
        f["burst_dur"] = (runs_sp[-1][1] - runs_sp[-1][0]) * HOP_S if prev_pauses else \
                         (last_sp - runs_sp[0][0]) * HOP_S

    return f


def add_deltas(f, hist):
    """hist: list of BASE dicts from previous pauses of this turn."""
    d = dict.fromkeys(DELTA_NAMES, 0.0)
    if hist:
        prev = hist[-1]
        p_f0 = [h["f0_end_st"] for h in hist]
        p_e = [h["e_end_rel"] for h in hist]
        d["d_f0_end_prev"] = f["f0_end_st"] - prev["f0_end_st"]
        d["d_f0_end_min"] = f["f0_end_st"] - min(p_f0)
        d["f0_new_low"] = 1.0 if f["f0_end_st"] < min(p_f0) - 0.3 else 0.0
        d["d_e_end_prev"] = f["e_end_rel"] - prev["e_end_rel"]
        d["d_e_end_min"] = f["e_end_rel"] - min(p_e)
        d["d_slope_prev"] = f["f0_slope_tail"] - prev["f0_slope_tail"]
        d["d_run_prev"] = f["final_run_rel"] - prev["final_run_rel"]
        d["d_burst_prev"] = f["burst_dur"] - prev["burst_dur"]
    return d


def vec(f, d):
    out = {**f, **d}
    return np.array([out[k] for k in FEATURE_NAMES], dtype=np.float32)


def build_dataset(data_dir, labels_df):
    import os
    X, keys = [], []
    for (tid, af), g in labels_df.groupby(["turn_id", "audio_file"], sort=False):
        x, sr = load_wav(os.path.join(data_dir, af))
        g = g.sort_values("pause_index")
        prev, hist = [], []
        for _, r in g.iterrows():
            f = extract(x, sr, float(r["pause_start"]), int(r["pause_index"]), list(prev))
            X.append(vec(f, add_deltas(f, hist)))
            keys.append((tid, int(r["pause_index"])))
            hist.append(f)
            try:
                pe = float(r["pause_end"])
                if np.isfinite(pe):
                    prev.append((float(r["pause_start"]), pe))
            except (KeyError, TypeError, ValueError):
                pass
    return np.array(X), keys
