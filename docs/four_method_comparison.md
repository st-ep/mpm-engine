
## Addendum (2026-08-21): reproduced on their engine, by the user

The user regenerated NCLaw's dataset and reran both their method and their
sys-id baseline (differentiable MPM, known constitutive form, parameters
optimized) on their own engine, tasks (a) time, (b) velocity, (c) geometry:

| method on their engine | jelly a/b/c | sand a/b/c | plasticine a/b/c | water a/b/c |
| --- | --- | --- | --- | --- |
| NCLaw (reproduced) | 9.6e-4 / 2.3e-4 / 3.6e-4 | 4.0e-5 / 5.8e-5 / 3.7e-4 | 1.5e-4 / 4.4e-5 / 1.6e-4 | 1.2e-3 / 1.4e-4 / 3.8e-4 |
| sys-id (reproduced) | 2.5e-8 / 3.8e-9 / 6.9e-9 | 1.2e-12 / 3.5e-13 / 3.3e-13 | 2.2e-10 / 1.1e-10 / 1.6e-10 | 1.1e-8 / 3.2e-10 / 6.0e-10 |

Readings: the NCLaw reproduction matches their published cells for jelly,
sand and plasticine; water reproduces 3 to 7x worse than published (their
Table 7 attributes water's published quality to an auxiliary velocity loss,
so this is plausibly a training-variant or seed difference). The sys-id
rows show their trajectories are fittable to 1e-12 inside their own engine,
which confirms that the cross-engine cells measured from their data are
dominated by the engine difference and pins the conclusion both engines now
agree on: the known constitutive form is worth four to nine orders of
magnitude over the learned network on their own benchmark, whether reached
by gradient descent (their sys-id, our diff-sim) or by the convex solve in
seconds. The remaining number is the truth-theta engine-gap floor on their
trajectories; the cross stage computes and stores it on arrival
(identification_excess_over_floor in cross_<material>_<scene>.json).
