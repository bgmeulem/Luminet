from black_hole import *
import numpy as np
import pytest


@pytest.mark.parametrize("mass", [1., 2.])
def test_varying_mass(mass):
    """
    Test if black hole can be created with a mass other than 1
    """

    for incl in (0, 45, 90, 135):
        bh = BlackHole(inclination=incl, mass=mass)
        bh.calc_isoradials([10, 20, 30], [10, 20, 30])  # calculate some isoradials, should be quick enough
        for radius, ghost_and_direct in bh.isoradials.items():
            ghost_ir, direct_ir = ghost_and_direct[1], ghost_and_direct[0]
            for isoradial in (ghost_and_direct[1], ghost_and_direct[0]):
                assert not any(np.isnan(isoradial.radii_b)), "Isoradials contain nan values"
                assert not any(np.isnan(isoradial.angles)), "Isoradials contain nan values"
        bh.calc_isoredshifts([-.1, 0, .2])  # test some isoredshifts, somewhat slower
    return None