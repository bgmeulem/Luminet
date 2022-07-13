import pytest
import numpy as np
from black_hole_math import *

N_inclines = 3
N_angles = 5
N_radii = 4


@pytest.mark.parametrize('mass', [1., 2.5, 4.])
@pytest.mark.parametrize('inclination', [np.random.randint(0, np.pi) for _ in range(N_inclines)])
@pytest.mark.parametrize('angle', [np.random.randint(0., 2 * np.pi) for _ in range(N_angles)])
@pytest.mark.parametrize('radius', [np.random.randint(6.01, 60.) for _ in range(N_radii)])
@pytest.mark.parametrize('order', [0., 1., 2.])  # test potential higher orders as well
class TestParametrized:

    def test_calc_periastron(self, mass, inclination, angle, radius, order):
        """
        Test the method for calculating the impact parameter with varying input parameters
        """
        p = calc_periastron(_r=radius*mass, incl=inclination, _alpha=angle, bh_mass=mass, n=order)
        return None

    def test_get_b_from_periastron(self, mass, inclination, angle, radius, order):
        b = calc_impact_parameter(_r=radius, incl=inclination, _alpha=angle, n=order, bh_mass=mass)
        assert (not np.isnan(b)) and (b is not None), f"Calculating impact parameter failed. with" \
                                                      f"M={mass},    incl={inclination}, alpha={angle},  R={radius}, " \
                                                      f"n={order}"
        return None
