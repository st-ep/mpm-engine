# Small material-parameter pilot, 2026-09-10

The paper keeps its existing materials, figure and PDF. Two one-at-a-time
20 percent yield-threshold changes were screened without modifying the adopted
experiment. Neither provided enough visual benefit to justify a new paper run.

| Candidate | E (kPa) | nu | Yield threshold (kPa) | Matched shape error, current (mm) | Matched shape error, screen (mm) |
|---|---:|---:|---:|---:|---:|
| Softer-yielding A | 80 | 0.30 | 1 to 0.8 | 1.043612 | 0.990795 |
| Higher-yield B | 80 | 0.30 | 10 to 12 | 1.312329 | 1.341445 |

Both still yield under the exact original shared 72 to 32 mm squeeze at
20 mm/s per finger. Supplied-state identification recovers 0.799242 kPa for
modified A and 12.001333 kPa for modified B. The independent outgoing-trial
activity check finds 60.51 percent and 34.96 percent of particles, respectively,
crossing the true elastic strain cap in at least one sampled outgoing trial.
This simulator-truth diagnostic is excluded from fitting.

The symmetric mean distance between released A and B surfaces is 0.742307 mm
for the current pair, 0.789259 mm with only A changed, and 0.847313 mm with
only B changed. This is a response-separation diagnostic, not a shape error
against the X target. Identical physical scale and camera views show a small
change for B and almost no visible change for A.

Screening reuses the current matching model's selected four gaps, predicts a
new work budget in each newly identified model, and executes that budget in
the candidate's true material. It also cross-executes the candidate and
unchanged counterpart's work budgets. Thus it does not punish the candidate
by retaining its previous work values, but it is not a fresh optimized plan.
No result here is labeled a new 32-evaluation planning run. A full search was
not launched because the visible gain was marginal, and B also worsened the
matched error. This does not establish that a better plan for B cannot exist.

All original initial particles, geometry, density, contact, target, four-pinch
structure, work rule, opening guard, observation time and metric are fixed.
No shape is rescaled, no component is removed, and no result is concealed.
The maximum parameter change was declared before either run. Parameters are
synthetic soft-solid perturbations, not measurements of a new hardware sample;
the archive includes the short physical-scale literature check and its limits.

Entry point: `experiments/robotics/x_material_sensitivity.py`.
Archive: `out/x_material_sensitivity_20260910`.
Its README gives exact reproduction commands. The archive includes source
snapshots and hashes, the declared protocol, complete probe states, fitted
laws, yield-activity audits, all eight shaping rollouts, forces, work stopping
records, quantitative comparisons, and scripts for rendering and auditing.
The old experiment archive and all four active manuscript/figure files are
checked against their hashes. No TeX edit or PDF rebuild was needed.

Cross-execution errors also change very little:

| Pair | A with B's work (mm) | B with A's work (mm) |
|---|---:|---:|
| Current A and B | 1.503309 | 2.895978 |
| Only A changed | 1.499470 | 2.908606 |
| Only B changed | 1.511638 | 2.907331 |

All six screened true-material executions complete without inversion. The
matched executions reach all four work targets. In both alternatives, B's
work on A reaches the unchanged 12 mm guard on pinches 1, 2 and 4, as in the
current paper experiment. B with A's work reaches all four work targets.
The numerical-resolution and hardware-kinematic audits are not repeated for
these unadopted screening candidates.
