<div align="center">

# Strideline

**A markerless goniometer: certified running-gait kinematics from one camera.**

<img alt="python" src="https://img.shields.io/badge/python-3.11%2B-blue">
<img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
<img alt="tests" src="https://img.shields.io/badge/tests-18%20passing-brightgreen">

</div>

A goniometer is the protractor a physiotherapist straps to your leg to measure a joint
angle by hand. Strideline is the video version: point a phone at a runner from the side,
and it turns the raw, jittery output of an off-the-shelf pose model into gait kinematics
that come with an actual error bound, not just a number.

<p align="center"><img src="assets/figures/overlay_demo.gif" width="640" alt="Raw pose overlay in red, Strideline-corrected skeleton in blue, on the same running clip"></p>

<div align="center">

| | |
|---|---|
| red skeleton | raw output of a COCO-17 pose model, frame by frame |
| blue skeleton | Strideline's corrected trajectory: same video, same model, one more layer |

</div>

## What Strideline measures

| metric | what it is | why it's here |
|---|---|---|
| cadence | steps per minute, both feet | the number every running watch already gives you |
| ground contact instant | per foot, per stride, with a certified error window | needs no force plate; recovered from ankle kinematics alone |
| contact-time asymmetry | left vs. right stride-time gap | a real gait asymmetry signal, not assumed to be zero |
| knee flexion at contact | mean and spread, in degrees | the post that inspired this project's "average knee shape," made rigorous |
| stride-time variability | coefficient of variation | a standard fatigue / consistency marker |
| hip vertical oscillation | peak-to-peak, normalized by leg length | camera-scale-free, so it's comparable across clips |
| Form Consistency Index | a single 0-100 number, defined below | proposed here, explicitly not a clinical score |

None of this is a diagnostic device. The Form Consistency Index is
`100 / (1 + stride_CV% + knee_angle_CV%)`, a transparent formula, not a
validated clinical index - it is reported as an engineering summary number,
not a medical claim.

## The constraint nobody's pose model enforces

A pose model reports a hip, a knee, and an ankle keypoint independently, frame by
frame. Nothing stops it from reporting a thigh that is 8% longer this frame than
last frame - a runner's femur does not do that. Strideline treats the runner as a
two-segment sagittal linkage with two *unknown but constant* segment lengths, and
recovers both the smooth joint-angle trajectory and those lengths at once, by
cycling four block updates on one energy function:

<div align="center">

| block | fixed | update | guarantee |
|---|---|---|---|
| thigh angle | shank angle, both lengths | one Iterated Extended Kalman Smoother step | never increases the energy (Levenberg-Marquardt guarded) |
| shank angle | thigh angle, both lengths | same, against the ankle residual | same guard |
| thigh length | both angles | closed-form least squares | exact minimizer |
| shank length | both angles | closed-form least squares | exact minimizer |

</div>

Every block either exactly minimizes its slice of the energy or is guarded so it
provably cannot increase it - so the whole alternating scheme's energy is
monotonically non-increasing, checked directly (not just argued) in
`tests/test_smoother.py` across dozens of random seeds and noise levels.

<p align="center"><img src="assets/figures/energy_trace_left.png" width="440" alt="Energy trace decreasing monotonically across outer iterations, real clip"></p>

## The contact-time certificate

Ground contact is the zero-crossing of the ankle's vertical velocity. Linear
interpolation between the two bracketing frames gives an estimate; a textbook
interpolation-error bound gives a *window* around it:

```
|t_hat - t*|  <=  (M2 * h^2) / (8 * |v'(t*)|)
```

`h` is the frame period, `M2` bounds the local curvature (estimated from the
smoothed trajectory itself, via Savitzky-Golay differentiation rather than
repeated finite-differencing, which would amplify noise faster at higher frame
rates instead of slower), and `v'(t*)` is the deceleration at contact. A single
narrow window can't be proven to bound the true curvature outright, so the
constant multiplying this formula is calibrated on a held-out synthetic sweep
and re-checked on a second, disjoint one - not asserted, measured twice.

<p align="center"><img src="assets/figures/bound_validation.png" width="440" alt="Every point on or below the diagonal (log-log): empirical timing error never exceeds the certified bound, within the validated envelope"></p>

<div align="center">

| | value |
|---|---|
| validated envelope | 30-120 fps, up to 6px synthetic pixel noise |
| certified-bound hold rate, held-out sweep | 645 / 648 events (99.5%) |
| detection recall / precision | 100% / 100% |
| average bound-to-error ratio | 67x (the certificate is conservative by design, not just correct) |

</div>

A 240 fps stress test (`scripts/calibrate.py`) still detects every event, but its
timing tail needs a calibration constant roughly 27x looser than the one used
here to fully cover - which is exactly why the certificate's headline envelope
stops at 120 fps rather than quietly absorbing that case (see "Where the
certificate runs out").

## Synthetic ground truth: what correction actually buys you

No motion-capture lab was available, so validation runs against a procedural
running-gait simulator with exact, known joint angles and contact instants
(`simulate.py`) - every number below is a comparison against a truth the
benchmark code never sees, averaged over 6 random seeds per cell.

<div align="center">

| fps | noise (px) | thigh RMSE raw (deg) | thigh RMSE corrected (deg) | shank RMSE raw (deg) | shank RMSE corrected (deg) |
|---|---|---|---|---|---|
| 30 | 1 | 0.70 | **0.50** | 1.11 | 1.32 |
| 30 | 3 | 1.82 | **0.99** | 2.47 | **1.67** |
| 30 | 6 | 3.59 | **1.93** | 4.75 | **2.53** |
| 60 | 1 | 0.68 | **0.41** | 0.80 | 1.01 |
| 60 | 3 | 1.83 | **0.97** | 2.37 | **1.22** |
| 60 | 6 | 3.62 | **2.17** | 4.76 | **2.55** |
| 120 | 1 | 0.67 | **0.37** | 0.78 | 0.89 |
| 120 | 3 | 1.82 | **0.98** | 2.34 | **1.22** |
| 120 | 6 | 3.61 | **2.07** | 4.69 | **2.67** |

</div>

Thigh angle improves at every noise level tested. Shank angle improves
substantially from moderate noise upward (roughly 2-4x lower RMSE at 3-6px), but
at the lowest noise level (1px) the correction is very slightly worse than the
raw estimate - the smoothing prior costs a small amount of accuracy when the raw
signal is already close to noise-free. Reported as found, not smoothed over.

## Case study: one runner, one camera

The clip below is a Pexels-license stock video (see
`assets/clips/LICENSE-NOTICE.txt`), run through the exact same pipeline as the
synthetic benchmark above, with no parameters retuned for it. The runner is
only continuously visible for about 2.4 seconds of the source clip (a handful
of strides per leg) - long enough to demonstrate every mechanism below, not
long enough for the cadence and variability numbers to be precision
measurements. Treat this section as "does the machinery work on a real, messy
video," and the synthetic table above as the actual accuracy claim.

<p align="center"><img src="assets/figures/phase_portrait_left.png" width="380" alt="Knee angle vs angular velocity phase portrait, real clip"> <img src="assets/figures/symmetry_radar.png" width="380" alt="Left vs right symmetry radar, real clip"></p>

<p align="center"><img src="assets/figures/constraint_violation_left.png" width="760" alt="Where the raw pose broke limb-length constancy over time, real clip"></p>

<p align="center"><img src="assets/figures/certificate_timeline_left.png" width="760" alt="Detected contact events with certified timing windows, real clip"></p>

<div align="center">

| metric | left | right |
|---|---|---|
| detected contacts (of ~2.4s visible) | 7 | 6 |
| knee angle at contact | 38.8 +/- 17.5 deg | 44.3 +/- 14.2 deg |
| stride-time CV | 45.5% | 31.3% |
| Form Consistency Index | 52.5 | 61.2 |
| kinematic consistency violation: raw -> corrected | 12.6% -> ~0% | 10.0% -> ~0% |

</div>

Cadence over the visible window: 330 steps/min combined; step-time asymmetry:
52%. Both numbers are computed exactly as in the synthetic benchmark, from
only 13 total detected contacts - reported honestly, not rounded into
something more reassuring.

## Where the certificate runs out

- Sagittal-plane, 2D only: a side-on, roughly-perpendicular, fixed-distance
  camera (treadmill or a planted tripod at a track) is the assumption behind
  every angle in this repository. Panning, zooming, or an oblique camera
  angle breaks the constant-projection-scale assumption the length constraint
  relies on.
- COCO-17 has no toe or heel keypoint, so ground contact is an ankle-kinematics
  proxy, not a true force-plate event.
- The certified bound's calibration stops at 120 fps. A 240 fps stress test
  still detects every event (perfect recall and precision) but needs a safety
  factor around 27x looser than the one used here to fully cover its timing
  tail, so it is reported separately rather than folded into the headline
  certificate.
- A short clip (a handful of strides) makes cadence, asymmetry, and
  variability numbers illustrative, not precise - the case study above is
  explicit about this rather than presenting a two-second clip as a full
  gait assessment.
- The Form Consistency Index is an engineering summary, not a clinical or
  diagnostic score, and nothing in this repository should be used to make a
  medical decision.

## Try it

```
pip install -e ".[dev]"
pytest                          # 18 tests, all synthetic, no download needed
python scripts/benchmark.py     # regenerates the synthetic table and figure above
python scripts/run.py assets/clips/running.mp4   # regenerates the case study
```

`scripts/calibrate.py` reproduces `CALIBRATED_SAFETY_FACTOR` from scratch on a
held-out seed range disjoint from every test in this repository, and prints
the 240 fps stress-test numbers cited above.

## Layout

| path | contents |
|---|---|
| `src/strideline/rts.py` | generic linear Kalman filter / RTS smoother |
| `src/strideline/kinematics.py` | keypoints <-> joint angles, forward kinematics |
| `src/strideline/smoother.py` | the constrained kinematic smoother |
| `src/strideline/events.py` | contact detection and the certified bound |
| `src/strideline/metrics.py` | cadence, asymmetry, variability, Form Consistency Index |
| `src/strideline/simulate.py` | the synthetic ground-truth gait generator |
| `src/strideline/pose.py` | local YOLO11-pose keypoint extraction |
| `src/strideline/viz.py` | every figure in this README |

## License

[Apache-2.0](LICENSE).
