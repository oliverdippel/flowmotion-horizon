# Training a bigger model on a free GPU (Colab / Kaggle)

`flowmotion train` already picks up CUDA automatically (`train.py` builds its device
from `torch.cuda.is_available()` — no code changes needed to run on GPU). The only
things that don't come for free are getting AMASS data onto the GPU environment and
not letting dependency installation clobber the preinstalled CUDA build of torch. This
doc covers both, plus a concrete config for a run that's meaningfully bigger than the
CPU-trained `real_v2` checkpoint already in this repo (d_model=384, n_layers=6,
n_heads=6, 20k steps, ~42 min on CPU).

**What a bigger/longer run will and won't fix.** More steps and a larger model should
lower training loss and likely improve jerk/drift numbers the same way the existing
ablation shows more training does. It will **not** narrow the wide bootstrap CIs at
rollout length 300 in the Results section — those are wide because only 7 of the 20
held-out subjects have a sequence long enough to evaluate at that length, which is a
property of which subjects `split_subjects` happened to hold out, not of how well the
model is trained. Fixing that needs more/longer held-out sequences, not more compute.

**Heads up on local disk:** this machine has ~10GB free, and the local AMASS copy at
`/Users/oliver/data/amass_real` is 5.7GB. Don't create a local zip/tar of it as an
upload staging step — sync/copy the existing folder directly (options below) instead
of doubling it on disk first.

## 1. Get AMASS data into Google Drive (or Kaggle Datasets)

You don't need a local archive — either of these streams the existing folder without
creating a second copy on disk:

**Option A — `rclone` (fastest, resumable, scriptable):**
```bash
brew install rclone
rclone config  # add a remote named "gdrive", follow the OAuth prompt in your browser
rclone copy /Users/oliver/data/amass_real gdrive:amass_real -P
```
`-P` shows live progress; if it's interrupted, re-running the same command resumes
(rclone skips files already present on the remote).

**Option B — Google Drive desktop app:** if it's already running and syncing a local
folder, just add `/Users/oliver/data/amass_real` (or move/symlink it under your
existing synced folder) — it uploads in the background with no extra CLI setup.

**Kaggle instead of Colab:** Kaggle Datasets don't have an rclone-style push from a
Mac; the practical path is zipping smaller pieces and uploading through the Kaggle
web UI's dataset creator (which itself stores a copy, so the local-disk caveat above
still applies — zip one sub-dataset directory at a time into e.g. `/tmp` and delete
each zip after its upload completes, rather than zipping all 5.7GB at once).
Given that, Colab + Drive is the more disk-friendly path from this machine.

## 2. Open Colab, get a GPU runtime

New notebook at [colab.research.google.com](https://colab.research.google.com) →
**Runtime → Change runtime type → T4 GPU** (the free-tier option). Free Colab GPU
sessions have a wall-clock/idle limit that varies (often several hours, sometimes
cut short) and **there is no mid-training resume in this codebase** (`train()`
always initializes a fresh optimizer and model) — so pick a step count you're
confident finishes in one sitting rather than a very long run you'd need to resume.

## 3. Mount Drive and copy data onto Colab's local disk

Run training against Colab's local SSD, not directly against the mounted Drive path
— Drive is a FUSE-mounted network filesystem, and `MotionWindowDataset`'s per-sequence
loading (plus the full streaming pass `Normalizer.fit_streaming` makes over every
window before training starts) will be much slower reading over that mount than off
local disk.

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/amass_real
!rsync -a --info=progress2 /content/drive/MyDrive/amass_real/ /content/amass_real/
```

## 4. Clone the repo and install without touching torch

```python
!git clone https://github.com/oliverdippel/flowmotion-horizon.git
%cd flowmotion-horizon
!pip install -e . --no-deps -q
```

`--no-deps` is the important part: `pyproject.toml` pins a CPU-only torch wheel index
for `uv` (so CI and local dev stay lightweight without a GPU), and a plain `pip
install -e .` without `--no-deps` would otherwise try to satisfy `torch>=2.2` from
PyPI and potentially reinstall over Colab's preinstalled CUDA-enabled build.
`--no-deps` skips all dependency resolution and just installs this package against
whatever numpy/pandas/matplotlib/torch Colab already has (all comfortably newer than
this project's minimums). Verify the GPU is actually visible before training:

```python
import torch

print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If `cuda.is_available()` is `False`, stop and fix the runtime type (step 2) before
continuing — training will silently fall back to Colab's CPU otherwise.

## 5. Train a bigger model

A concrete step up from `real_v2` — deeper and wider (6.38M → roughly 25-30M
params), more steps, bigger batch (GPU throughput at this sequence length, K=H=10,
is high enough that batch size is no longer CPU-memory-bound):

```python
!python -m flowmotion.cli train --data-root /content/amass_real \
    --out /content/runs/real_v3 --steps 40000 --batch-size 128 \
    --d-model 512 --n-layers 8 --n-heads 8 --dim-ff 2048 \
    --cache-size 3500 --val-every 1000
```

`--val-every 1000` turns on held-out-subject validation-loss tracking
(`val_log.csv` in the output dir) — worth having for a run you're going to call the
new headline result, since `real_v2` didn't use it (noted as a limitation in the
README). Watch `train_log.csv`/the printed loss; if it's still dropping fast when
`--steps` runs out, that's a sign to raise `--steps` on the next attempt rather than
call the run done.

## 6. Download the checkpoint (and don't run eval on Colab)

```python
from google.colab import files

files.download("/content/runs/real_v3/model.pt")
files.download("/content/runs/real_v3/train_log.csv")
files.download("/content/runs/real_v3/val_log.csv")  # if --val-every was set
```

Run `flowmotion eval` / `eval-baseline` / `visualize` back on this machine, not on
Colab: eval isn't the expensive step here (22 min for 9,363 trials on this machine's
11 CPU cores with `--workers 8`, per the README), Colab's free-tier CPU allocation is
typically only 2 cores, and keeping eval on the same machine keeps it numerically
comparable to the existing `real_v2`/baseline numbers, which were also run here.

## 7. Reproduce the Results-section workflow locally

```bash
uv run flowmotion eval --checkpoint /path/to/downloaded/model.pt \
    --data-root /Users/oliver/data/amass_real --lengths 30,60,90,150,300 \
    --seeds-per-subject 3 --noise-samples 3 --workers 8 \
    --out-dir ./runs/real_v3/eval_report

uv run flowmotion eval-baseline --checkpoint /path/to/downloaded/model.pt \
    --data-root /Users/oliver/data/amass_real --lengths 30,60,90,150,300 \
    --seeds-per-subject 3 --out-dir ./runs/real_v3/baseline_report

uv run python scripts/compare_runs.py \
    --run "real_v2 (CPU, small)=runs/real_v2/eval_report_v2/horizon_eval.csv" \
    --run "real_v3 (GPU, large)=runs/real_v3/eval_report/horizon_eval.csv" \
    --out assets/scale_comparison.png --report-out assets/scale_comparison.md
```

That last comparison is the interesting one to fold into the README: it's the same
"does more training/scale actually move the metrics, with CIs" question the existing
well-trained-vs-undertrained ablation asks, but for model scale instead of step count
— and it's directly comparable to the existing baseline table using the same
`scripts/compare_runs.py` machinery.
