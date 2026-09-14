# Attention-Based Heart-Rate Forecasting

Real-time BPM streamed from a wearable pulse sensor while playing two mobile games with very different pacing, for
ecast with a pretrained attention-based transformer (PatchTST) and evaluated against ground-truth moving-average B
PM.

## Overview

- **Sensor**: MAX30105 pulse oximeter/heart-rate sensor, read over I2C from an Arduino (`iot_project.ino`), stream
ing instantaneous BPM over serial.
- **Data collection**: 3 recording sessions each while playing **Geometry Dash** (fast, reflex-driven) and **Clash
 Royale** (slower, strategy-driven) — `data/g1.txt`–`g3.txt` and `data/cr1.txt`–`cr3.txt`, concatenated into `data
/g_total.txt` and `data/cr_total.txt`.
- **Model**: a pretrained 2-input-channel [PatchTST](https://huggingface.co/chungimungi/PatchTST-2-input-channels)
 transformer (`PatchTSTForPrediction`) takes both BPM streams as parallel channels and forecasts future BPM values
 from past values — no fine-tuning, used directly for inference (`scripts/model.py`).
- **Evaluation**: predicted BPM is smoothed with a moving average and compared against the moving average of the g
round-truth BPM.

## Results

| Task | Actual moving-avg BPM | Predicted moving-avg BPM | Accuracy |
|---|---|---|---|
| Geometry Dash | 90.93 | 79.61 | **87.55%** |
| Clash Royale | 87.75 | 75.81 | **86.39%** |

## Repo structure

```
iot_project.ino           Arduino sketch — reads MAX30105, streams BPM over serial
data/                     Raw BPM logs per session, plus concatenated per-task totals
scripts/model.py          Loads the pretrained PatchTST model, runs inference, computes accuracy
scripts/plotting.py       Plotting helpers for actual-vs-predicted BPM
notebooks/                Exploration and results notebooks (final.ipynb, view_bpm_cr.ipynb, view_bpm_gd.ipynb)
final_ARIMA.ipynb         Classical ARIMA baseline exploration
images/                   Pipeline/results figures
```

## Reproducing

1. Wire a MAX30105 to an Arduino and flash `iot_project.ino`; log the serial BPM stream to a `.txt` file.
2. `pip install torch transformers numpy matplotlib`
3. Run `scripts/model.py` (point it at your BPM `.txt` files) to load the pretrained PatchTST model, generate fore
casts, and print/plot accuracy.

## Limitations

- Evaluated against a smoothed moving-average baseline, not raw per-sample ground truth — accuracy numbers reflect
 trend-following ability, not sample-level precision.
- Small sample (3 sessions/task) from a single subject — not validated across people or activities beyond these tw
o games.
- Uses the pretrained PatchTST model as-is; no task-specific fine-tuning was performed.
