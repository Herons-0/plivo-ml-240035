# End-of-Turn Detection — Plivo ML Assignment

Predicts `p_eot` (probability a user's turn is over) for every annotated pause,
from **causal audio only** (nothing after `pause_start` is ever read).

## Run

```
pip install -r requirements.txt
python predict.py --data_dir <folder> --out predictions.csv
```

`<folder>` must contain `labels.csv` and the `audio/` files it references
(same schema as the handout). `eot_model.pkl` must sit next to `predict.py`.

## Results (5-fold grouped out-of-fold, official scorer)

| | silence baseline | this model |
|---|---|---|
| english | 1600 ms | **1255 ms** |
| hindi | 850 ms | **840 ms** |

Shipped `predictions_*.csv` are in-sample (model trained on all provided turns);
the OOF numbers above are the honest claim — details in `RUNLOG.md`.

## Files

- `predict.py` — inference entry point (loads `eot_model.pkl`)
- `eot_features.py` / `mel_features.py` — causal feature extraction (read these for the causality audit)
- `train_model.py` — reproduces `eot_model.pkl` from the two language folders
- `RUNLOG.md` — every scoring run, both research rounds (graded)
- `NOTES.md` — signal / failures / next steps
- `SUMMARY.html` — full report with graphs and the human-vs-agent breakdown
