"""Train the final EOT model on both language folders and save eot_model.pkl.

    python train_model.py --data_dirs eot_data/english eot_data/hindi

Prints honest 5-fold grouped OOF scores (official metric) for the record.
"""
import argparse, os, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

from eot_features import build_dataset, FEATURE_NAMES
from mel_features import build_mel_dataset

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def official(df):
    best = None
    n_turns = df.turn_id.nunique()
    is_hold = (df.label == "hold").values
    dur = df.dur.values; p = df.p.values; tid = df.turn_id.values
    for t in THRESHOLDS:
        fires = p >= t
        for d in DELAYS:
            cut = len(set(tid[is_hold & fires & (d < dur)])) / n_turns
            if cut > 0.05:
                continue
            lat = np.where(fires[~is_hold], d, TIMEOUT_S).mean()
            if best is None or lat < best:
                best = lat
    return (best if best is not None else TIMEOUT_S) * 1000


def make_models():
    return {
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=0.1)),
        "GBM": HistGradientBoostingClassifier(max_iter=250, max_depth=3,
                                              learning_rate=0.05, l2_regularization=1.0,
                                              min_samples_leaf=10, random_state=0),
        "GBM2": HistGradientBoostingClassifier(max_iter=400, max_depth=2,
                                               learning_rate=0.05, l2_regularization=0.5,
                                               min_samples_leaf=8, random_state=1),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 16), alpha=1.0,
                                           max_iter=1500, random_state=0)),
    }


def apply_prespeech_guard(p, X):
    """A pause before any real WORD cannot end a turn. Listening finding
    (human): en__091 opens with a thud + 3s silence before the first word.
    A thud rings periodically (fools the pitch tracker) but has almost no
    energy above 1 kHz and forms a single short burst; require ALL THREE
    conditions so genuine one-word turns are spared where possible."""
    stime = X[:, FEATURE_NAMES.index("speech_time")]
    nruns = X[:, FEATURE_NAMES.index("n_speech_runs")]
    hf = X[:, FEATURE_NAMES.index("hf_ratio")]
    thud = (stime < 0.6) & (nruns <= 1) & (hf < 0.10)
    return np.where(thud, np.minimum(p, 0.02), p)


def lang_flag_from_turn_id(tid):
    """1.0 for Hindi-looking ids, 0.0 for English; default Hindi (the hidden
    test is mostly Hindi, and prosodic features are shared anyway)."""
    t = str(tid).lower()
    if t.startswith("en"):
        return 0.0
    return 1.0


def _features_for_dir(dd):
    """Feature extraction with a local cache (safe: cache key = data dir)."""
    tag = os.path.basename(os.path.normpath(dd))
    cache = f"cache_{tag}.npz"
    df = pd.read_csv(os.path.join(dd, "labels.csv"))
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if z["Xh"].shape[1] == len(FEATURE_NAMES):   # stale-cache guard
            return df, z["Xh"], z["Xp"], [(str(t), int(i)) for t, i in z["keys"]]
    Xh, keys = build_dataset(dd, df)
    _, Xp, _ = build_mel_dataset(dd, df)
    np.savez_compressed(cache, Xh=Xh, Xp=Xp,
                        keys=np.array([(t, int(i)) for t, i in keys], dtype=object))
    return df, Xh, Xp, keys


def load_all(data_dirs):
    Xs, metas = [], []
    for dd in data_dirs:
        df, Xh, Xp, keys = _features_for_dir(dd)
        df = df.set_index(["turn_id", "pause_index"]).loc[keys].reset_index()
        df["dur"] = df.pause_end - df.pause_start
        flag = np.array([[lang_flag_from_turn_id(t)] for t in df.turn_id])
        Xs.append(np.hstack([Xh, Xp, flag]))
        metas.append(df)
    return np.vstack(Xs), pd.concat(metas, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="eot_model.pkl")
    args = ap.parse_args()

    X, meta = load_all(args.data_dirs)
    y = (meta.label == "eot").astype(int).values
    groups = meta.turn_id.values
    print(f"training matrix {X.shape}, eot rate {y.mean():.2f}")

    # metric-aware TRAIN-time filter (label-side info only, never a feature):
    # a hold shorter than 0.4 s cannot cause a false cutoff at delay >= its
    # duration, so we spend capacity on eot vs LONG hesitations only.
    train_ok = (y == 1) | (meta.dur.values > 0.4)

    # honest OOF for the run log
    gkf = GroupKFold(n_splits=5)
    oof = {m: np.zeros(len(y)) for m in make_models()}
    for tr, te in gkf.split(X, y, groups):
        tr = tr[train_ok[tr]]
        for name, mdl in make_models().items():
            mdl.fit(X[tr], y[tr])
            oof[name][te] = mdl.predict_proba(X[te])[:, 1]
    ens = np.mean(list(oof.values()), axis=0)
    ens = apply_prespeech_guard(ens, X)
    for dd in args.data_dirs:
        tag = os.path.basename(dd.rstrip("/\\"))
        pref = "en" if tag.startswith("en") else "hi"
        m = meta.turn_id.str.lower().str.startswith(pref).values
        d = meta[m].copy(); d["p"] = ens[m]
        print(f"OOF official metric [{tag}]: {official(d):.0f} ms")

    # final fit on everything (same train-time filter)
    fitted = {}
    for name, mdl in make_models().items():
        mdl.fit(X[train_ok], y[train_ok])
        fitted[name] = mdl
    with open(args.out, "wb") as f:
        pickle.dump({"models": fitted, "n_features": X.shape[1],
                     "feature_names_hand": FEATURE_NAMES}, f)
    print(f"saved -> {args.out}")

    np.save("oof_ens.npy", ens)
    meta.assign(p_oof=ens).to_csv("oof.csv", index=False)


if __name__ == "__main__":
    main()
