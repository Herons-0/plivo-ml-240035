"""Log-mel tail representation, causally anchored at the true speech end.

For each pause: take audio[0:pause_start], find the last actual speech
frame (energy-based, same logic as eot_features), then represent the
last 1.0 s of speech ending there as a log-mel patch.  Cepstral-mean
normalization uses ONLY prefix speech frames (causal).
Outputs per pause:
  pooled: per-band mean/std over last 300 ms and last 1 s, plus deltas (160 dims)
  flat:   40 mels x 100 frames flattened (4000 dims) for PCA
"""
import numpy as np
from eot_features import load_wav, _frames, _energy_db, FRAME_MS, HOP_MS, PREFIX_CAP_S

N_MELS = 40
N_FRAMES = 100  # 1.0 s at 10 ms hop


def _mel_fb(sr, n_fft, n_mels=N_MELS, fmin=50.0, fmax=7600.0):
    def hz2mel(h): return 2595.0 * np.log10(1 + h / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(min(fmax, sr / 2)), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[i, l:c] = np.linspace(0, 1, c - l, endpoint=False)
        fb[i, c:r] = np.linspace(1, 0, r - c, endpoint=False)
    return fb.astype(np.float32)


_FB_CACHE = {}


def logmel(x, sr):
    fr = _frames(x, sr, FRAME_MS)
    if len(fr) == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    n_fft = fr.shape[1]
    key = (sr, n_fft)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_fb(sr, n_fft)
    w = np.hanning(n_fft).astype(np.float32)
    sp = np.abs(np.fft.rfft(fr * w, axis=1)) ** 2
    m = sp @ _FB_CACHE[key].T
    return np.log(m + 1e-8).astype(np.float32)


def mel_for_pause(x, sr, pause_start):
    end = int(pause_start * sr)
    prefix = x[max(0, end - int(PREFIX_CAP_S * sr)):end]
    if len(prefix) < sr // 5:
        return np.zeros((N_FRAMES, N_MELS), dtype=np.float32)
    e = _energy_db(prefix, sr)
    floor = np.percentile(e, 10); loud = np.percentile(e, 95)
    smask = e > max(floor + 12.0, loud - 25.0)
    M = logmel(prefix, sr)
    n = min(len(M), len(e))
    M, e, smask = M[:n], e[:n], smask[:n]
    if not smask.any():
        return np.zeros((N_FRAMES, N_MELS), dtype=np.float32)
    last = int(np.max(np.nonzero(smask)))
    cmn = M[smask].mean(axis=0)          # causal: prefix speech only
    M = M - cmn
    seg = M[max(0, last + 1 - N_FRAMES):last + 1]
    if len(seg) < N_FRAMES:
        seg = np.vstack([np.zeros((N_FRAMES - len(seg), N_MELS), np.float32), seg])
    return seg


def pool(seg):
    tail = seg[-30:]                      # last 300 ms
    d = np.diff(seg, axis=0)
    return np.concatenate([
        seg.mean(0), seg.std(0),
        tail.mean(0), tail.std(0),
        d[-30:].mean(0), np.abs(d[-30:]).mean(0),
    ]).astype(np.float32)


def build_mel_dataset(data_dir, labels_df):
    import os
    flats, pools, keys = [], [], []
    for (tid, af), g in labels_df.groupby(["turn_id", "audio_file"], sort=False):
        x, sr = load_wav(os.path.join(data_dir, af))
        for _, r in g.sort_values("pause_index").iterrows():
            seg = mel_for_pause(x, sr, float(r["pause_start"]))
            flats.append(seg.ravel())
            pools.append(pool(seg))
            keys.append((tid, int(r["pause_index"])))
    return np.array(flats), np.array(pools), keys
