# Deferred shaping review, 2026-09-11

The user paused this investigation to save and push the current repository.
Resume only when requested. No controller changes or new experiments were
authorized as part of this checkpoint.

## Current experiment

The active study is `experiments/robotics/x_consistent_force_study.py`, with
results in `out/x_consistent_force_study_20260911`. Each identified material
gets 32 evaluations of four nominal openings for y, x, y, x pinches. The
selected simulation supplies both nominal velocity and force histories
directly, with no retiming. Execution adds PI force feedback to nominal
finger velocity. Entire plans are exchanged between the two materials.

The sole active manuscript and PDF are `paper/icra2027/paper.tex` and
`paper/icra2027/paper.pdf`. Rebuild with `./paper/icra2027/build.sh` after
saving any TeX or figure changes, then inspect the affected pages. The
manuscript is intentionally excluded from Git in this clone; do not assume
a code checkpoint includes it. Preserve the current Table II numbers until
the colleagues' timing discrepancy is resolved.

## Unresolved interpretation

Matched plans track their force references accurately at baseline, but
exchanged plans do not. Per-pinch raw-force relative L2 errors over the
executed portions are:

| Actual material | Plan | Baseline tracking errors | Stops |
| --- | --- | --- | --- |
| A | A | 0.1733–0.3048% | All four profiles complete |
| B | B | 0.0218–0.1412% | All four profiles complete |
| A | B | 79.6436–87.5379% | 12 mm opening guard on pinches 1, 2, 4 |
| B | A | 50.5469–113.5398% | All four durations complete |

Matched errors increase with grid refinement; baseline tracking does not
establish robust control under material mismatch. B executing A's lower
force profiles has poor tracking without travel-guard or speed-limit hits.
These data do not establish that those profiles are physically infeasible.
A executing B's plan fails under the current controller and constraints;
that is not proof that every possible controller would fail.

The figure currently demonstrates the outcome of the entire planning and
execution pipeline. It does not demonstrate accurately applying identical
force histories to different materials and observing different shapes.
Both material response and controller tracking contribute to the result.

The defensible scientific motivation is to evaluate whether identified
models provide motion and loading plans for plastic shaping. Neither
necessity nor superiority of force feedback over motion replay has been
established. Larger visual differences alone do not justify the controller.
The current implementation does not enforce a hard force limit, establish
damage prevention, or validate force control on the physical Franka Hand.
The user reports having no finger-force sensor for the proposed hardware
experiment, which would use a new material rather than physical A or B.

## Proposed next checks, not yet performed

1. Diagnose exchanged-profile tracking and assess one common controller by
   force accuracy over the feasible range. Keep material-independent rules
   and physical limits; do not tune to enlarge shape differences. Report
   infeasible plans explicitly and distinguish them from controller failures.
2. Replay each of the two frozen plans on each true material with feedback
   disabled: four baseline simulations, preserving nominal motion and
   timing. Compare against feedback execution without rerunning planning.

Choose the next step with the user. Do not revise the paper to imply these
checks have passed. Keep frozen archives and unrelated work intact.
