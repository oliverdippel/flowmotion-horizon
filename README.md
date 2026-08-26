# flowmotion-horizon

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
- **Data**: a real AMASS-format loader (`src/flowmotion/data/loader.py`) and a
  deterministic synthetic fixture generator that reproduces the same on-disk format
  (`src/flowmotion/data/synthetic.py`) — used so this repo runs and tests fully
  without needing AMASS itself, which is license-gated and can't be auto-downloaded.
- **The harness** (`src/flowmotion/eval/`): for held-out subjects, runs matched
  *free* rollouts (fed their own predictions back in) and *teacher-forced* rollouts
  (re-anchored to real data every step) from the same seed, and reports, as a
  function of rollout length:
  - **foot skate** — horizontal drift of a foot joint during ground contact
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
```

## Using real AMASS data

Nothing above requires AMASS. To point at a real download instead of the synthetic
fixture: download AMASS body-pose data from https://amass.is.tue.mpg.de (registration
+ license acceptance required — this repo cannot and does not automate that), then
either pass `--data-root /path/to/amass` or `export AMASS_ROOT=/path/to/amass`. The
loader (`flowmotion.data.loader`) expects the same `<dataset>/<subject>/<sequence>.npz`
layout AMASS itself distributes (see `flowmotion/data/amass_format.py`) — no code
changes needed.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## Limitations / open design calls

This was built to a 2-3 day scope; these are the corners cut on purpose, not bugs:

- **No real SMPL body model.** SMPL/SMPL-H shape (`betas`) is license-gated
  separately from AMASS mocap data, so forward kinematics uses a hardcoded
  approximate rest-pose bone-offset template (`flowmotion/data/skeleton.py`), not a
  real per-subject shape. This doesn't undermine the harness's metrics, which
  measure *relative* degradation with rollout length rather than absolute geometry
  — but absolute distances/positions are approximate. `betas` are still loaded and
  threaded through the pipeline so a real SMPL FK can be dropped in later.
- **Joint parent topology** in `skeleton.py` follows the standard published SMPL
  kinematic tree but hasn't been cross-checked against a canonical SMPL joint table
  with real data — worth verifying before trusting it against real AMASS.
- **Up-axis assumed Z-up.** Common for AMASS-derived pipelines but varies by tool;
  configurable but not verified against real files (no real files available yet).
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
  flow-matching, not empirically tuned (no real-data benchmark to tune against yet).
- **Foot-contact thresholds** (height/velocity) are heuristic constants tuned to
  the synthetic fixture's scale and will need recalibrating against real AMASS's
  actual units.
