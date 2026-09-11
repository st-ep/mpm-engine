"""Finish the requested 200 mL pilot row after the eight-run search limit.

Use a separate, frozen continuation protocol with at most two additional runs.
The original stopped protocol/results and all numerical tolerances are retained.
"""
from experiments.pour import pour_coulomb_extend_200 as extension


if __name__ == "__main__":
    extension.WORK = extension.OUT / "pilot_extension_200_finish"
    extension.main()
