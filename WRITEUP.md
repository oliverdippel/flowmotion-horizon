# My eval harness told me my headline metric was measuring the wrong thing

I built [flowmotion-horizon](https://github.com/oliverdippel/flowmotion-horizon) to
answer a question that bothered me about how motion-generation models get evaluated:
almost every demo, paper figure, and eval script I could find judges a model on short
rollouts — 30, maybe 60 frames. That's exactly the regime where autoregressive error
hasn't had time to compound. A model can look flawless at 30 frames and fall apart at
300, and a fixed-length eval will never tell you.

So I built a small conditional flow-matching model for human motion (trained on
AMASS) and, more importantly, a harness that runs every metric across a *range* of
rollout lengths instead of one. At each length, it runs two rollouts from the same
seed window with the same noise draws: one **free** (fed back on its own output, the
way you'd actually deploy it) and one **teacher-forced** (re-conditioned on real data
every step, but still recording what the model predicted). The L2 distance between
those two trajectories — which only differ in whether the model is trusting itself —
is what I called **divergence**, and I built the whole project around the idea that
watching it grow with rollout length was the useful signal.

Then I ran an ablation to check that assumption, and it didn't hold up.

## The ablation

I trained two versions of the exact same architecture on the exact same data with the
exact same seed: one for 20,000 steps (loss 2.24 → 0.28), one for 1,500 steps
(deliberately undertrained). I evaluated both through the identical protocol — 3 seed
windows × 3 noise draws per trial, 9,363 trials each — and compared them with 95%
bootstrap confidence intervals over held-out subjects, so I could actually claim a
difference instead of eyeballing two lines on a chart.

Jerk and distributional drift separated the two models cleanly at every rollout
length — 7–10× apart on jerk, 3–6× apart on drift, no overlapping interval anywhere
from length 30 to 300. Divergence — the metric I built the entire repository around —
did not. Its confidence intervals overlapped at every single length. Statistically,
this evaluation could not tell the well-trained model from the undertrained one using
the metric that was supposed to be the point.

That's not a bug I found and fixed. It's a real result: at this scale, free-vs-
teacher-forced divergence tracks *rollout length* far more than it tracks *training
quality*. Both models drift from their own teacher-forced trajectory as length grows,
regardless of how good they are, because a fixed-step Euler sampler compounds
whatever noise it has at every step — trained or not. Jerk and drift, it turns out,
are the metrics actually doing the work of catching a bad model. Divergence catches
something else: that autoregressive rollouts drift, period.

## The baseline, and a second surprise

A negative result about your own metric is worth having, but it raised an obvious
follow-up: none of that ablation says whether the model is any good, only that two
versions of it differ. So I added the standard sanity check from the older motion-
prediction literature — the "zero-velocity" baseline (Martinez et al., CVPR 2017):
repeat the last observed frame forever. No model, no learning, just holding still.
I ran it through the identical matched protocol, same held-out subjects, same window
sizes.

Jerk gave the baseline a perfect score — exactly zero, at every length, no motion to
be jerky about. That's obviously not "the baseline is better than the model," but it
is the exact failure mode the ablation's use of jerk never had to face: jerk rewards
absence of motion as readily as it rewards *good* motion. A low jerk number only
means something once you've independently ruled out the model producing nothing.

The real surprise was in distributional drift. On speed, the trained model was
closer to real motion statistics than the frozen baseline at the middle rollout
lengths — a genuine win. But on *acceleration*, the baseline beat the model at
**every single length tested**. The model's acceleration profile was measurably
more anomalous than a character standing perfectly still. That's not a flattering
result, and I don't think it would have shown up any other way: it isn't a
compounding-error story (it's a property of the model's very first predicted frame,
not something that emerges over a long rollout), and it isn't something a fixed-
length visual check would catch either — a person watching a 30-frame GIF isn't going
to notice "the acceleration distribution has a slightly heavier tail than reality."

## What this changes about how I read this project

Three lessons, in order of how much they surprised me:

1. **A metric built around a plausible-sounding mechanism (self-conditioning
   drift compounding over time) isn't automatically the metric doing the
   discriminative work.** You have to check, with confidence intervals, not
   intuition.
2. **No single metric is trustworthy in isolation** — jerk is the sharpest
   signal for training quality in the ablation, and also the metric most
   easily gamed by producing nothing. You need a metric that can veto a
   degenerate baseline (distributional drift, here) before you trust a metric
   that's good at grading a non-degenerate one.
3. **A baseline isn't a formality.** I could have shipped this project with an
   impressive-looking divergence-vs-length curve and no comparison point, and
   it would have read fine. It took one trivial baseline, run through the same
   harness with no special-casing, to find a real weakness in the model that
   the rest of the eval suite was silent about.

None of this makes the underlying model impressive — it's a small transformer
trained for 42 minutes on a CPU. What I'd stand behind is the harness: it's built so
that adding a new thing to compare against (a baseline, an ablation, eventually a
bigger model) means writing one small adapter, not touching the trial loop, and it
found real, sometimes uncomfortable things about its own subject without me going
looking for them. That's the property I wanted out of it going in, and it's the part
of this project I'd want a reviewer to actually poke at.

Full writeup of the method, results, and known limitations is in the
[README](README.md); the code is at
[github.com/oliverdippel/flowmotion-horizon](https://github.com/oliverdippel/flowmotion-horizon).
