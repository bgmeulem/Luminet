from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc
# import mpmath

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


def calc_q(periastron: float, bh_mass: float, tol=1e-3) -> float:
    """
    Convert Periastron distance P to the variable Q (easier to work with)
    """
    # limits give no substantial speed improvement
    # if periastron - 2. * bh_mass < tol:
    #     # limit for small values
    #     return .5 * (periastron - 2. * bh_mass) * (periastron + 6. * bh_mass)
    # if 1/periastron < tol:
    #     # limit for large values
    #     return periastron
    # if periastron <= 2*bh_mass:
    #     raise ValueError("Non-physical periastron found (P <= 2M, aka the photon sphere)."
    #                      "If you want to calculate non-physical values, you should implement the mpmath library")
    q = np.sqrt((periastron - 2. * bh_mass) * (periastron + 6. * bh_mass))
    # Q is complex if P < 2M = r_s
    return q


def calc_b_from_periastron(periastron: float, bh_mass: float, tol: float = 1e-5) -> float:
    """
    Get impact parameter b from Periastron distance P
    """
    # limits give no substantial speed improvement
    # if abs(periastron) < tol:  # could physically never happen
    #     print("tolerance exceeded for calc_b_from_P(P_={}, M={}, tol={}".format(periastron, bh_mass, tol))
    #     return np.sqrt(3 * periastron ** 2)
    # WARNING: the paper most definitely has a typo here. The fracture on the right hand side equals b², not b.
    # Just fill in u_2 in equation 3, and you'll see. Only this way do the limits P -> 3M and P >> M hold true,
    # as well as the value for b_c
    return np.sqrt(periastron ** 3 / (periastron - 2. * bh_mass))  # the impact parameter


def k(periastron: float, bh_mass: float) -> float:
    """
    Calculate modulus of elliptic integral
    """
    q = calc_q(periastron, bh_mass)
    # adding limits does not substantially improve speed, nor stability
    # if q < 10e-3:  # numerical stability
    #     return np.sqrt(.5)
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return np.sqrt((q - periastron + 6 * bh_mass) / (2 * q))  # the modulus of the elliptic integral


def k2(periastron: float, bh_mass: float, tol: float = 1e-6):
    """Calculate the squared modulus of elliptic integral"""
    q = calc_q(periastron, bh_mass)
    # adding limits does not substantially improve speed
    # if 1 / periastron <= tol:
    #     # limit of P -> inf, Q -> P
    #     return 0.
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return (q - periastron + 6 * bh_mass) / (2 * q)  # the modulus of the ellipitic integral


def zeta_inf(periastron: float, bh_mass: float, tol: float = 1e-6) -> float:
    """
    Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)
    """
    q = calc_q(periastron, bh_mass)  # Q variable, only call to function once
    arg = (q - periastron + 2 * bh_mass) / (q - periastron + 6 * bh_mass)
    z_inf = np.arcsin(np.sqrt(arg))
    return z_inf


def zeta_r(periastron: float, r: float, bh_mass: float) -> float:
    """
    Calculate the elliptic integral argument Zeta_r for a given value of P and r
    """
    q = calc_q(periastron, bh_mass)
    a = (q - periastron + 2 * bh_mass + (4 * bh_mass * periastron) / r) / (q - periastron + (6 * bh_mass))
    s = np.arcsin(np.sqrt(a))
    return s


def cos_gamma(_a: float, incl: float, tol=10e-5) -> float:
    """
    Calculate the cos of the angle gamma
    """
    if abs(incl) < tol:
        return 0
    return np.cos(_a) / np.sqrt(np.cos(_a) ** 2 + 1 / (np.tan(incl) ** 2))  # real


def cos_alpha(phi: float, incl: float) -> float:
    """Returns cos(angle) alpha in observer frame given angles phi (black hole frame) and
    inclination (black hole frame)"""
    return np.cos(phi) * np.cos(incl) / np.sqrt((1 - np.sin(incl) ** 2 * np.cos(phi) ** 2))


def alpha(phi: float, incl: float):
    """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)"""
    return np.arccos(cos_alpha(phi, incl))


def filter_periastrons(periastron: [], bh_mass: float, tol: float = 10e-3) -> []:
    """
    Removes instances where P == 2*M
    returns indices where this was the case
    """
    return [e for e in periastron if abs(e - 2. * bh_mass) > tol]


def eq13(periastron: float, ir_radius: float, ir_angle: float, bh_mass: float, incl: float, n: int = 0,
         tol=10e-6) -> float:
    """
    Relation between radius (where photon was emitted in accretion disk), a and P.
    P can be converted to b, yielding the polar coordinates (b, a) on the photographic plate

    This function get called almost everytime when you need to calculate some black hole property
    """
    z_inf = zeta_inf(periastron, bh_mass)
    q = calc_q(periastron, bh_mass)
    m_ = k2(periastron, bh_mass)  # modulus of the elliptic integrals. mpmath takes m = k² as argument.
    ell_inf = ellipkinc(z_inf, m_)  # Elliptic integral F(zeta_inf, k)
    g = np.arccos(cos_gamma(ir_angle, incl))

    # Calculate the argument of sn (mod is m = k², same as the original elliptic integral)
    # WARNING: paper has an error here: \sqrt(P / Q) should be in denominator, not numerator
    # There's no way that \gamma and \sqrt(P/Q) can end up on the same side of the division
    if n:  # higher order image
        ell_k = ellipk(m_)  # calculate complete elliptic integral of mod m = k²
        ellips_arg = (g - 2. * n * np.pi) / (2. * np.sqrt(periastron / q)) - ell_inf + 2. * ell_k
    else:  # direct image
        ellips_arg = g / (2. * np.sqrt(periastron / q)) + ell_inf

    # sn is an Jacobi elliptic function: elliptic sine. ellipfun() takes 'sn'
    # as argument to specify "elliptic sine" and modulus m=k²
    sn, cn, dn, ph = ellipj(ellips_arg, m_)
    sn2 = sn * sn
    term1 = -(q - periastron + 2. * bh_mass) / (4. * bh_mass * periastron)
    term2 = ((q - periastron + 6. * bh_mass) / (4. * bh_mass * periastron)) * sn2

    return 1. - ir_radius * (term1 + term2)  # solve this for zero


def midpoint_method(func, args: Dict, __x, __y, __ind):
    new_x = __x
    new_y = __y

    x_ = [new_x[__ind], new_x[__ind + 1]]  # interval of P values
    inbetween_x = np.mean(x_)  # new periastron value, closer to solution yielding 0 for ea13
    new_x.insert(__ind + 1, inbetween_x)  # insert middle P value to calculate

    y_ = [new_y[__ind], new_y[__ind + 1]]  # results of eq13 given the P values
    # calculate the P value inbetween
    inbetween_solution = func(periastron=inbetween_x, **args)
    new_y.insert(__ind + 1, inbetween_solution)
    y_.insert(1, inbetween_solution)
    ind_of_sign_change_ = np.where(np.diff(np.sign(y_)))[0]
    new_ind = __ind + ind_of_sign_change_[0]

    return new_x, new_y, new_ind  # return x and y refined in relevant regions, as well as new index of sign change


def improve_solutions_midpoint(func, args, x, y, index_of_sign_change, iterations) -> float:
    """
    To increase precision.
    Recalculate each solution in :arg:`solutions` using the provided :arg:`func`.
    Achieves an improved solution be re-evaluating the provided :arg:`func` at a new
    :arg:`x`, inbetween two pre-existing values for :arg:`x` where the sign of :arg:`y` changes.
    Does this :arg:`iterations` times
    """
    index_of_sign_change_ = index_of_sign_change
    new_x = x
    new_y = y
    new_ind = index_of_sign_change_  # location in X and Y where eq13(P=X[ind]) equals Y=0
    for iteration in range(iterations):
        new_x, new_y, new_ind = midpoint_method(func=func, args=args, __x=new_x, __y=new_y, __ind=new_ind)
    updated_periastron = new_x[new_ind]
    return updated_periastron


def calc_periastron(_r, incl, _alpha, bh_mass, midpoint_iterations=100, plot_inbetween=False,
                    n=0, min_periastron=1., initial_guesses=20) -> float:
    """
        Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value
        This periastron can be converted to an impact parameter b, yielding the observer frame coordinates (b, alpha).
        Does this by generating range of periastron values, evaluating eq13 on this range and using a midpoint method
        to iteratively improve which periastron value solves equation 13.
        The considered initial periastron range must not be lower than min_periastron (i.e. the photon sphere),
        otherwise non-physical solutions will be found. These are interesting in their own right (the equation yields
        complex solutions within radii smaller than the photon sphere!), but are for now outside the scope of this project.
        Must be large enough to include solution, hence the dependency on the radius (the bigger the radius of the
        accretion disk where you want to find a solution, the bigger the periastron solution is, generally)

        Args:
            _r (float): radius on the accretion disk (BH frame)
            incl (float): inclination of the black hole
            _alpha: angle along the accretion disk (BH frame and observer frame)
            bh_mass (float): mass of the black hole
            midpoint_iterations (int): amount of midpoint iterations to do when searching a periastron value solving eq13
            plot_inbetween (bool): plot
        """

    # angle = (_alpha + n*np.pi) % (2 * np.pi)  # Assert the angle lies in [0, 2 pi]

    def get_plot(X, Y, solution, radius=_r):
        fig = plt.figure()
        plt.title("Eq13(P)\nr={}, a={}".format(radius, round(_alpha, 5)))
        plt.xlabel('P')
        plt.ylabel('Eq13(P)')
        plt.axhline(0, color='black')
        plt.plot(X, Y)
        plt.scatter(solution, 0, color='red')
        return plt

    # TODO: an x_range between [min - 2.*R] seems to suffice for isoradials < 30M, but this is guesstimated
    periastron_range = list(np.linspace(min_periastron, 2. * _r, initial_guesses))
    y_ = [eq13(P_value, _r, _alpha, bh_mass, incl, n) for P_value in periastron_range]  # values of eq13
    ind = np.where(np.diff(np.sign(y_)))[0]  # only one solution should exist
    periastron_solution = periastron_range[ind[0]] if len(ind) else None  # initial guesses for P

    if (periastron_solution is not None) and (not np.isnan(
            periastron_solution)):  # elliptic integral found a periastron solving equation 13
        args_eq13 = {"ir_radius": _r, "ir_angle": _alpha, "bh_mass": bh_mass, "incl": incl, "n": n}
        periastron_solution = \
            improve_solutions_midpoint(func=eq13, args=args_eq13,
                                       x=periastron_range, y=y_, index_of_sign_change=ind[0],
                                       iterations=midpoint_iterations)  # get better P values
    if plot_inbetween:
        get_plot(periastron_range, y_, periastron_solution).show()
    return periastron_solution


def calc_impact_parameter(_r, incl, _alpha, bh_mass, midpoint_iterations=100, plot_inbetween=False,
                          n=0, min_periastron=1., initial_guesses=20, use_ellipse=True) -> float:
    """
    Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value
    This periastron is then converted to an impact parameter b, yielding the observer frame coordinates (b, alpha).
    Does this by generating range of periastron values, evaluating eq13 on this range and using a midpoint method
    to iteratively improve which periastron value solves equation 13.
    The considered initial periastron range must not be lower than min_periastron (i.e. the photon sphere),
    otherwise non-physical solutions will be found. These are interesting in their own right (the equation yields
    complex solutions within radii smaller than the photon sphere!), but are for now outside the scope of this project.
    Must be large enough to include solution, hence the dependency on the radius (the bigger the radius of the
    accretion disk where you want to find a solution, the bigger the periastron solution is, generally)

    Args:
        _r (float): radius on the accretion disk (BH frame)
        incl (float): inclination of the black hole
        _alpha: angle along the accretion disk (BH frame and observer frame)
        bh_mass (float): mass of the black hole
        midpoint_iterations (int): amount of midpoint iterations to do when searching a periastron value solving eq13
        plot_inbetween (bool): plot
    """

    # angle = (_alpha + n*np.pi) % (2 * np.pi)  # Assert the angle lies in [0, 2 pi]

    periastron_solution = calc_periastron(_r, incl, _alpha, bh_mass, midpoint_iterations, plot_inbetween, n,
                                          min_periastron, initial_guesses)
    if periastron_solution is None or periastron_solution <= 2.*bh_mass:
        # No periastron was found, or a periastron was found, but it's non-physical
        # Assume this is because the image of the photon trajectory might have a periastron,
        # but it does not actually move towards this, but away from the black hole
        # these are generally photons at the front of the accretion disk: use the ellipse function
        # (the difference between the two goes to 0 as alpha approaches 0 or 2pi)
        return ellipse(_r, _alpha, incl)
    elif periastron_solution > 2.*bh_mass:
        b = calc_b_from_periastron(periastron_solution, bh_mass)
        return b
    else:
        # Should never happen
        # why was no P found?
        # fig = plt.figure()
        # plt.plot(x_, y_)
        # plt.show()
        raise ValueError(f"No solution was found for the periastron at (r, a) = ({_r}, {_alpha}) and incl={incl}")


def phi_inf(periastron, M):
    q = calc_q(periastron, M)
    ksq = (q - periastron + 6. * M) / (2. * q)
    z_inf = zeta_inf(periastron, M)
    phi = 2. * (np.sqrt(periastron / q)) * (ellipk(ksq) - ellipkinc(z_inf, ksq))
    return phi


def mu(periastron, bh_mass):
    return float(2 * phi_inf(periastron, bh_mass) - np.pi)


def ellipse(r, a, incl) -> float:
    """Equation of an ellipse, reusing the definition of cos_gamma.
    This equation can be used for calculations in the Newtonian limit (large P = b, small a)
    or to visualize the equatorial plane."""
    g = np.arccos(cos_gamma(a, incl))
    b_ = r * np.sin(g)
    return b_


def flux_intrinsic(r, acc, bh_mass):
    r_ = r / bh_mass
    log_arg = ((np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
              ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
    f = (3. * bh_mass * acc / (8 * np.pi)) * (1 / ((r_ - 3) * r ** 2.5)) * \
        (np.sqrt(r_) - np.sqrt(6) + 3 ** -.5 * np.log10(log_arg))
    return f


def flux_observed(r, acc, bh_mass, redshift_factor):
    flux_intr = flux_intrinsic(r, acc, bh_mass)
    return flux_intr / redshift_factor ** 4


def redshift_factor(radius, angle, incl, bh_mass, b_):
    """
    Calculate the gravitational redshift factor (1 + z), ignoring cosmological redshift.
    """
    # WARNING: the paper is absolutely incomprehensible here. Equation 18 for the redshift completely
    # leaves out important factors. It should be:
    # 1 + z = (1 - Ω*b*cos(η)) * (-g_tt -2Ω*g_tϕ - Ω²*g_ϕϕ)^(-1/2)
    # The expressions for the metric components, Ω and the final result of Equation 19 are correct though
    # TODO perhaps implement other metrics? e.g. Kerr, where g_tϕ != 0
    # gff = (radius * np.sin(incl) * np.sin(angle)) ** 2
    # gtt = - (1 - (2. * M) / radius)
    z_factor = (1. + np.sqrt(bh_mass / (radius ** 3)) * b_ * np.sin(incl) * np.sin(angle)) * \
               (1 - 3. * bh_mass / radius) ** -.5
    return z_factor


if __name__ == '__main__':
    M = 1
    solver_params = {'initial_guesses': 10,
                     'midpoint_iterations': 10,
                     'plot_inbetween': False,
                     'minP': 3.01 * M}
    # writeFramesEq13(5, solver_params=solver_params)
