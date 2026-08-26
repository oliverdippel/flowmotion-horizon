# flowmotion-horizon

[![CI](https://github.com/oliverdippel/flowmotion-horizon/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverdippel/flowmotion-horizon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

Conditional flow matching for human motion on AMASS, plus a **horizon-stability
evaluation harness** — the actual point of this repo.

## The problem

Generative motion models are usually judged on how a rollout looks for the first
second or two. That's not the failure mode that burns teams shipping to partners: a
model whose autoregressive rollout looks clean for 30 frames and quietly collapses
by 90 is a much more common and much more expensive surprise. This repo builds a
small conditional flow-matching model as something real to evaluate, then spends
its actual effort on a harness that measures *how rollout quality degrades as a
function of rollout length*, on subjects the model never saw during training.

## What's here

- **Model**: a conditional flow-matching (rectified-flow) velocity field over
  windows of AMASS body pose, conditioned on a past-frame window plus subject and
  action/dataset embeddings (`src/flowmotion/model/`).
- **Data**: a real AMASS-format loader (`src/flowmotion/data/loader.py`), lazily
  loaded with a bounded LRU cache so training scales past what fits in memory, and a
  deterministic synthetic fixture generator that reproduces the same on-disk format
  (`src/flowmotion/data/synthetic.py`) — used so this repo runs and tests fully
  without needing AMASS itself, which is license-gated and can't be auto-downloaded.
- **The harness** (`src/flowmotion/eval/`): for held-out subjects, runs matched
  *free* rollouts (fed their own predictions back in) and *teacher-forced* rollouts
  (re-anchored to real data every step) from the same seed, and reports, as a
  function of rollout length:
  - **foot skate** — horizontal drift of a foot joint during ground contact, with
    the floor height calibrated from real reference data rather than assumed
  - **jerk** — mean squared third derivative of joint position (smoothness /
    jitter proxy)
  - **free-vs-teacher-forced divergence** — the primary "does it collapse" signal:
    since a generative rollout has no ground truth beyond the seed, this isolates
    compounding self-conditioning error by comparing a rollout against a version of
    itself that's kept honest with real data at every step
  - **distributional drift** — z-score deviation of the rollout's trailing-window
    speed/acceleration from real held-out statistics

  All four are reported per held-out subject and aggregated (mean ± std) across
  subjects, so it's visible whether collapse is universal or subject-specific.
- **Visualization** (`src/flowmotion/viz.py`): renders a free-rollout-vs-teacher-
  forced comparison as an animated GIF, so collapse is something you watch, not just
  a number in a CSV.

## Results (real AMASS data)

Trained on real AMASS data — ACCAD + BMLhandball + HumanEva (929 sequences, 33
subjects, 28 train / 5 held out) — for 15,000 steps (~14 min on CPU; training loss
2.29 → ~0.3). Evaluated on the 5 held-out real subjects:

![Horizon-stability metrics on real AMASS data](assets/horizon_curves_example.png)

The headline metric — free-vs-teacher-forced divergence — climbs cleanly with
rollout length (0.96 → 1.50 → 1.56 → 2.00 across lengths 30/60/90/150): the core
signal this harness exists to catch is showing up on real data with a real,
lightly-trained model. Foot-skate/jerk/drift are noisier at this scale — only 5
held-out subjects, shrinking further at longer lengths as sequences run out of
frames — so treat those three as directional, not final, until the held-out pool
grows.

![Free rollout vs. teacher-forced rollout](assets/rollout_comparison_example.gif)

*(Sample visualization, not cherry-picked for quality — this is what a real, briefly
trained checkpoint actually produces. Re-run `flowmotion visualize` on a longer-
trained checkpoint to see it improve.)*

Two real bugs were caught and fixed by actually running this against real data
rather than only the synthetic fixture (see commit history): a non-motion
`shape.npz` file in real AMASS subject directories that crashed the loader, and a
foot-contact detector that silently reported zero contact because it assumed
"ground" was at world-height 0 — which held for the synthetic fixture by
construction but not for this codebase's approximate skeleton on real geometry.

## Quickstart

```bash
uv sync --extra dev
uv run flowmotion demo --out ./runs/demo
```

`demo` generates a synthetic fixture, trains a tiny model for a couple hundred
steps, runs the horizon eval, and writes `./runs/demo/eval_report/horizon_curves.png`
— one panel per metric, x-axis = rollout length. That plot is the point.

Or run each stage by hand:

```bash
uv run flowmotion prepare-fixture --out ./data/synthetic_amass
uv run flowmotion train --data-root ./data/synthetic_amass --out ./runs/tiny --steps 500
uv run flowmotion eval --checkpoint ./runs/tiny/model.pt --data-root ./data/synthetic_amass \
    --lengths 30,60,90,150,300 --out-dir ./runs/tiny/eval_report
uv run flowmotion rollout --checkpoint ./runs/tiny/model.pt --data-root ./data/synthetic_amass \
    --subject Dataset0/subject00 --length 90 --out ./runs/tiny/rollout.npz
uv run flowmotion visualize --checkpoint ./runs/tiny/model.pt --data-root ./data/synthetic_amass \
    --subject Dataset0/subject00 --length 90 --out ./runs/tiny/rollout.gif
```

## Using real AMASS data

Download AMASS body-pose data (SMPL+H format) from https://amass.is.tue.mpg.de
(registration + license acceptance required — this repo cannot and does not
automate that), extract each downloaded dataset archive into one common directory
so you get `<root>/<dataset_name>/<subject>/<sequence>.npz`, then either pass
`--data-root /path/to/amass` or `export AMASS_ROOT=/path/to/amass`. No code changes
needed — this has been verified end to end against real ACCAD/BMLhandball/HumanEva
downloads (see amass_format.py for the exact key/shape contract, confirmed against
real files).

For a larger real corpus, pass `--cache-size` to `train` large enough to comfortably
exceed your total sequence count (the default of 64 is sized for quick/small runs;
raise it once training data no longer fits that scope) so the dataset stays fully
cached after the first pass instead of repeatedly re-decompressing files.

If you use AMASS data, cite it per their license:

> Mahmood, N., Ghorbani, N., Troje, N. F., Pons-Moll, G., & Black, M. J. (2019).
> AMASS: Archive of Motion Capture as Surface Shapes. *ICCV*.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src/flowmotion
```

Optional pre-commit hooks (ruff lint/format + basic hygiene checks) are configured
in `.pre-commit-config.yaml`:

```bash
uv run pre-commit install
```

## Limitations / open design calls

This started as a 2-3 day build; the items below are corners cut on purpose or
gaps found and not yet closed — not silent bugs:

- **No real SMPL body model.** SMPL/SMPL-H shape (`betas`) is license-gated
  separately from AMASS mocap data, so forward kinematics uses a hardcoded
  approximate rest-pose bone-offset template (`flowmotion/data/skeleton.py`), not a
  real per-subject shape. Verified against real data: absolute geometry is
  noticeably off (e.g. a foot joint's true "floor" sits around world-height
  0.5-0.6m under this rig, not 0 — `estimate_foot_floor` calibrates around this),
  though the harness's metrics measure *relative* degradation with rollout length,
  which survives this. `betas` are still loaded and threaded through the pipeline
  so a real SMPL FK can be dropped in later.
- **Joint parent topology** in `skeleton.py` follows the standard published SMPL
  kinematic tree but hasn't been cross-checked against a canonical SMPL joint table
  with a real body model — worth verifying if absolute joint positions start to matter.
- **Up-axis Z-up** — verified against real AMASS files (root translation height sits
  in a plausible ~0.8-1.0m band consistent with pelvis height above a Z=0 floor).
- **Root canonicalization** uses simple per-window XY recentering, not full
  yaw-alignment (rotating each window to face +x) as used in some motion-generation
  papers (HuMoR, MDM-style). Cheaper to implement correctly; yaw-alignment is a
  natural next step.
- **Conditioning injection** uses extra tokens in the transformer sequence, not
  AdaLN-style modulation (DiT-style) — stock `nn.TransformerEncoder`, smaller
  surface area for subtle bugs, at some cost to architectural elegance.
- **Divergence metric uses a single noise seed per trial**, not averaged over
  multiple draws — a known variance gap, flagged rather than hidden.
- **10-step Euler ODE integration** at sampling time is a standard default for
  flow-matching, not empirically tuned.
- **Held-out sample size**: the results above use 5 held-out real subjects, which
  is enough for the divergence-vs-length trend to read clearly but not enough for
  foot-skate/jerk/drift to be statistically solid on their own — more held-out
  subjects (more AMASS sub-datasets) would tighten those.

## Citation

```bibtex
@software{flowmotion_horizon,
  author = {Dippel, Oliver},
  title = {flowmotion-horizon: Conditional Flow Matching on AMASS with a Horizon-Stability Evaluation Harness},
  url = {https://github.com/oliverdippel/flowmotion-horizon},
  year = {2026}
}
```

See `CITATION.cff`. If you use AMASS data with this code, also cite AMASS itself
(above) — that's a condition of their license, not just courtesy.
