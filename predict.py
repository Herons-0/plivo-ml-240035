"""End-of-turn prediction.

    python predict.py --data_dir <folder> --out predictions.csv

<folder> must contain labels.csv (same schema as the handout) and the
audio/ files it references. Writes turn_id,pause_index,p_eot.

CAUSALITY: all features for a pause come from audio[0:pause_start] only
(see eot_features.py / mel_features.py). pause_end of the current pause
is never read; boundaries of PREVIOUS pauses of the same turn are past
information for a live agent and are used as context. If the label
column is present it is ignored.

Requires eot_model.pkl, eot_features.py, mel_features.py next to this
file (deps: numpy, pandas, scikit-learn, soundfile/scipy).
"""
import argparse, os, pickle, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eot_features import build_dataset
from mel_features import build_mel_dataset
from train_model import lang_flag_from_turn_id, apply_prespeech_guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eot_model.pkl")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    Xh, keys = build_dataset(args.data_dir, df)
    _, Xp, keys2 = build_mel_dataset(args.data_dir, df)
    assert keys == keys2
    flag = np.array([[lang_flag_from_turn_id(t)] for t, _ in keys])
    X = np.hstack([Xh, Xp, flag])
    if X.shape[1] != bundle["n_features"]:
        raise SystemExit(f"feature dim {X.shape[1]} != model {bundle['n_features']}")

    p = np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"].values()], axis=0)
    p = apply_prespeech_guard(p, X)

    out = pd.DataFrame(keys, columns=["turn_id", "pause_index"])
    out["p_eot"] = np.round(p, 4)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
