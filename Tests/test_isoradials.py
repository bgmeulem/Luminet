from black_hole import BlackHole, Isoradial
import numpy as np
from tqdm import tqdm


def test_bh_isoradials() -> None:
    incl = np.random.randint(180)
    n_isoradials = np.random.randint(2, 10)
    radii = np.linspace(3.01, 60, n_isoradials)
    bh = BlackHole(inclination=incl, mass=1.)
    bh.calc_isoradials(radii, radii)
    assert len(bh.isoradials.keys()) == n_isoradials, f"Not enough isoradials were calculated. Expected " \
                                                      f"{n_isoradials}, but calculated only {len(bh.isoradials.keys())}"
    direct_ir = [bh.isoradials[r][0] for r in radii]
    ghost_ir = [bh.isoradials[r][1] for r in radii]
    return None


def test_isoradial_precision():

    return None
