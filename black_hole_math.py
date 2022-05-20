import numpy as np
import matplotlib.pyplot as plt
import mpmath
from typing import Tuple, Dict
from tqdm import tqdm

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


def calc_q(periastron: float, bh_mass: float) -> float:
    """Convert Periastron distance P to Q (easier to work with)"""
    q = mpmath.sqrt((periastron - 2. * bh_mass) * (periastron + 6. * bh_mass))
    # Q is complex if P < 2M = r_s
    return q


def calc_b_from_periastron(periastron: float, bh_mass: float, tol: float = 1e-5) -> float:
    """Get impact parameter b from Periastron distance P"""
    if abs(periastron) < tol:  # could physically never happen
        print("tolerance exceeded for calc_b_from_P(P_={}, M={}, tol={}".format(periastron, bh_mass, tol))
        return mpmath.sqrt(3 * periastron ** 2)
    # WARNING: the paper most definitely has a typo here. The fracture on the right hand side equals b², not b.
    # Just fill in u_2 in equation 3, and you'll see. Only this way do the limits P -> 3M and P >> M hold true,
    # as well as the value for b_c
    return mpmath.sqrt(periastron ** 3 / (periastron - 2. * bh_mass))  # the impact parameter


def k(periastron: float, bh_mass: float) -> float:
    """Calculate modulus of elliptic integral"""
    q = calc_q(periastron, bh_mass)
    if q < 10e-3:  # numerical stability
        return mpmath.sqrt(.5)
    else:
        # WARNING: Paper has an error here. There should be brackets around the numerator.
        return mpmath.sqrt((q - periastron + 6 * bh_mass) / (2 * q))  # the modulus of the ellipitic integral


def k2(periastron: float, bh_mass: float, tol: float = 1e-6):
    """Calculate the squared modulus of elliptic integral"""
    q = calc_q(periastron, bh_mass)
    # TODO: add inf / inf
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return (q - periastron + 6 * bh_mass) / (2 * q)  # the modulus of the ellipitic integral


def zeta_inf(periastron: float, bh_mass: float, tol: float = 1e-6) -> float:
    """Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)"""
    q = calc_q(periastron, bh_mass)  # Q variable, only call to function once
    arg = (q - periastron + 2 * bh_mass) / (q - periastron + 6 * bh_mass)
    z_inf = mpmath.asin(mpmath.sqrt(arg))
    return z_inf


def zeta_r(periastron: float, r: float, M: float) -> float:
    """Calculate the elliptic integral argument Zeta_r for a given value of P and r"""
    q = calc_q(periastron, M)
    a = (q - periastron + 2 * M + (4 * M * periastron) / r) / (q - periastron + (6 * M))
    s = mpmath.asin(mpmath.sqrt(a))
    return s


def cos_gamma(_a: float, incl: float, tol=10e-5) -> float:
    """Calculate the cos of the angle gamma"""
    if abs(incl) < tol:
        return 0
    else:
        return mpmath.cos(_a) / mpmath.sqrt(mpmath.cos(_a) ** 2 + mpmath.cot(incl) ** 2)  # real


def cos_alpha(phi: float, incl: float) -> float:
    """Returns cos(angle) alpha in observer frame given angles phi (black hole frame) and
    inclination (black hole frame)"""
    return mpmath.cos(phi) * mpmath.cos(incl) / mpmath.sqrt((1 - mpmath.sin(incl) ** 2 * mpmath.cos(phi) ** 2))


def alpha(phi: float, incl: float):
    """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)"""
    return mpmath.acos(cos_alpha(phi, incl))


def F(zeta: float, m):
    """Calculates the incomplete elliptic integral of argument zeta and mod m = k²
    Args:
        zeta: the argument of the elliptic integral
        m: the modulus of the elliptic integral. mpmath takes m=k² as modulus
    Returns:
        float: the value of the elliptic integral of argument zeta and modulus m=k²"""
    return mpmath.ellipf(zeta, m)  # takes k**2 as mod, not k


def K(m):
    """Calculates the complete elliptic integral of mod m=k²"""
    return mpmath.ellipf(np.pi / 2, m)


def filterP(periastron: np.ndarray, bh_mass: float, tol: float = 10e-3) -> []:
    """removes instances where P == 2*M
    returns indices where this was the case"""
    return [e for e in periastron if abs(e - 2. * bh_mass) > tol]


def eq13(periastron: float, _r: float, _a: float, bh_mass: float, incl: float, n: int = 0, tol=10e-6) -> float:
    """
    Relation between radius (where photon was emitted in accretion disk), a and P.
    P can be converted to b, yielding the polar coordinates (b, a) on the photographic plate

    This function get called almost everytime when you need to calculate some black hole property
    """
    z_inf= zeta_inf(periastron, bh_mass)
    q = calc_q(periastron, bh_mass)
    m_ = k2(periastron, bh_mass)  # modulus of the elliptic integrals. mpmath takes m = k² as argument.
    ellinf = F(z_inf
, m_)  # Elliptic integral F(z# # nf), k)
    g = mpmath.acos(cos_gamma(_a, incl))  # real

    # Calculate the argument of sn (mod is m = k², same as the original elliptic integral)
    # WARNING: paper has an error here: \sqrt(P / Q) should be in denominator, not numerator
    # There's no way that \gamma and \sqrt(P/Q) can end up on the same side of the division
    if n:  # higher order image
        ellK = K(m_)  # calculate complete elliptic integral of mod m = k²
        ellips_arg = (g - 2. * n * np.pi) / (2. * mpmath.sqrt(periastron / q)) - ellinf + 2. * ellK
    else:  # direct image
        ellips_arg = g / (2. * mpmath.sqrt(periastron / q)) + ellinf  # complex

    # sn is an Jacobi elliptic function: elliptic sine. ellipfun() takes 'sn'
    # as argument to specify "elliptic sine" and modulus m=k²
    sn = mpmath.ellipfun('sn', ellips_arg, m=m_)
    sn2 = sn * sn
    # sn2 = float(sn2.real)
    term1 = -(q - periastron + 2. * bh_mass) / (4. * bh_mass * periastron)
    term2 = ((q - periastron + 6. * bh_mass) / (4. * bh_mass * periastron)) * sn2

    return 1. - _r * (term1 + term2)  # solve this for zero


def write_frames_eq13(radius: float, solver_params: Dict, incl: float = 10., M: float = 1.,
                      angular_precision=100) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    fig.set_size_inches([5, 5])
    ax.set_xlabel('P')
    ax.set_ylabel('eq13(P, r, a)')

    for n in tqdm(range(angular_precision)):
        a = np.pi * n / angular_precision

        def eq13_P(P, radius_=radius, a_=a, M_=M, incl_=incl):
            return eq13(P, radius_, a_, M_, incl_)  # solve this equation for P

        s = calc_impact_parameter(radius, incl, a, M, **solver_params, use_ellipse=False)
        x = np.linspace(solver_params["minP"], 1.01 * radius, 100)
        x_range = filterP(x, M)
        y = [eq13_P(x_).real for x_ in x_range]
        x = [calc_b_from_periastron(p, M) for p in filterP(x, M)]

        ax.clear()
        ax.set_xlabel('P')
        ax.set_ylabel('eq13(P, r, a)')
        if s is not None:
            ax.scatter(s.real, 0, color='red', zorder=10)
        ax.plot(x, y)
        plt.axhline(0, color='black')
        plt.title("Equation 13, r = {}\na = {}".format(radius, round(a, 5)))
        fig.savefig('movie/eq13/frame{:03d}.png'.format(n))


def calc_impact_parameter(_r, incl, _alpha, bh_mass, midpoint_iterations=100, plot_inbetween=False,
                          n=0, min_periastron=1, initial_guesses=20, elliptic_integral_interval=None,
                          use_ellipse=True) -> float:
    """Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value"""

    def eq13_P(__P, __r, __alpha, __bh_mass, __incl, __n):
        s = eq13(__P, _r=__r, _a=__alpha, bh_mass=__bh_mass, incl=__incl, n=__n)  # solve this equation for P
        return s

    def midpoint_method_periastron(__x, __y, __ind, __radius, __angle, __bh_mass, __inclination, __n):
        new_x = __x
        new_y = __y

        x_ = [new_x[__ind], new_x[__ind + 1]]  # interval of P values
        inbetween_P = np.mean(x_)
        new_x.insert(__ind + 1, inbetween_P)  # insert middle P value to calculate

        y_ = [new_y[__ind], new_y[__ind + 1]]  # results of eq13 given the P values
        # calculate the P value inbetween
        inbetween_solution = eq13_P(__P=inbetween_P, __r=__radius, __alpha=__angle, __bh_mass=__bh_mass, __incl=__inclination,
                                    __n=__n)
        new_y.insert(__ind + 1, inbetween_solution)
        y_.insert(1, inbetween_solution)
        ind_of_sign_change_ = np.where(np.diff(np.sign(y_)))[0]  # TODO: how to deal with complex
        new_ind = __ind + ind_of_sign_change_[0]

        return new_x, new_y, new_ind  # return x and y refined in relevant regions, as well as new index of sign change

    def improve_periastron_solution(periastron, x, y, indices_of_sign_change, iterations, radius, angle, bh_mass, inclination, n_):
        """To increase precision.
        Searches again for a solution inbetween the interval where the sign changes
        Does this <iterations> times"""
        updated_periastron = periastron
        indices_of_sign_change_ = indices_of_sign_change
        new_x = x
        new_y = y
        for i in range(len(indices_of_sign_change_)):  # update each solution of P
            new_ind = indices_of_sign_change_[i]  # location in X and Y where eq13(P=X[ind]) equals Y=0
            for iteration in range(iterations):
                new_x, new_y, new_ind = midpoint_method_periastron(new_x, new_y, new_ind, radius, angle, bh_mass, inclination, n_)
            updated_periastron[i] = new_x[new_ind]
            indices_of_sign_change_ = [e + iterations for e in indices_of_sign_change_]
        return updated_periastron

    def get_plot(X, Y, solutions, radius=_r):
        fig = plt.figure()
        plt.title("Eq13(P)\nr={}, a={}".format(radius, round(_alpha, 5)))
        plt.xlabel('P')
        plt.ylabel('Eq13(P)')
        plt.axhline(0, color='black')
        plt.plot(X, Y)
        for periastron \
                in solutions:
            plt.scatter(periastron, 0, color='red')
        return plt

    if elliptic_integral_interval is None:
        elliptic_integral_interval = (70 * np.pi / 180,
                                      2 * np.pi - 70 * np.pi / 180)

    # TODO: an x_range until 1.1*R seems to suffice for isoradials < 30M, but this is guesstimated
    x_ = list(np.linspace(min_periastron, 2. * _r, initial_guesses))  # range of P values without P == 2*M
    x_ = filterP(x_, bh_mass)
    y_ = [eq13_P(P_value, _r, _alpha, bh_mass, incl, n) for P_value in x_]  # values of eq13
    y_ = [y.real if y.imag < 10e-6 else y for y in y_]  # drop tiny imaginary parts

    ind = np.where(np.diff(np.sign(y_)))[0]
    _P = [x_[i] for i in ind]  # initial guesses for P

    # If image is ghost image, or direct image with lots of curvature: calculate with elliptic integrals
    if len(_P):
        _P = improve_periastron_solution([P.real for P in _P], x_, y_, ind, midpoint_iterations, _r, _alpha, bh_mass, incl, n)  # get better P values
        if plot_inbetween:
            get_plot(x_, y_, _P).show()
        # TODO: how to correctly pick the right solution (there are generally 3 solutions, of which 2 complex)
        _P = max(_P)
        return float(calc_b_from_periastron(_P, bh_mass).real)
    elif use_ellipse:  # Front side of disk: calculate impact parameter with the ellipse formula
        return float(ellipse(_r, _alpha, incl))
    else:
        # Should never happen
        # why was no P found?
        # fig = plt.figure()
        # plt.plot(x_, y_)
        # plt.show()
        raise ValueError(f"No solution was found for the periastron at (r, a) = ({_r}, {_alpha}) and incl={incl}")
        return None  # TODO: implement


def phi_inf(periastron, M):
    q = calc_q(periastron, M)
    ksq = (q - periastron + 6. * M) / (2. * q)
    z_inf = zeta_inf(periastron, M)
    phi = 2. * (mpmath.sqrt(periastron / q)) * (mpmath.ellipk(ksq) - mpmath.ellipf(z_inf, ksq))
    return phi


def mu(periastron, bh_mass):
    return float(2 * phi_inf(periastron, bh_mass) - np.pi)


def ellipse(r, a, incl):
    """Equation of an ellipse, reusing the definition of cos_gamma.
    This equation can be used for calculations in the Newtonian limit (large P = b, small a)
    or to visualize the equatorial plane."""
    g = mpmath.acos(cos_gamma(a, incl))
    b_ = r * mpmath.sin(g)
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
    return flux_intr / redshift_factor**4


def redshift_factor(radius, angle, incl, M, b_):
    """Calculate the redshift factor (1 + z), ignoring cosmological redshift."""
    # TODO: the paper makes no sense here
    gff = (radius * mpmath.sin(incl) * mpmath.sin(angle)) ** 2
    gtt = - (1 - (2. * M) / radius)
    z_factor = (1. + np.sqrt(M / (radius ** 3)) * b_ * np.sin(incl) * np.sin(angle)) * \
               (1 - 3. * M / radius) ** -.5
    return z_factor


def find_a(b_, z, incl, M, r_):
    """Given a certain redshift value z (NOT redshift factor 1+z) and radius b on the observer plane, find the angle
    on the observer plane. Include contributions from the disk at radii r."""
    radius = np.linspace(3 * M, 100 * M, len(b_)) if not r_ else r_

    sin_angle = ((1. + z) * np.sqrt(1. - 3. * M / radius) - 1) / ((M / radius ** 3) ** .5 * b_ * np.sin(incl))
    return np.arcsin(sin_angle)


def getPFromB(b, M):
    # TODO: please don't ever use this
    num1 = 3**(2/3)*b**2
    num2 = 3**(1/3) * (mpmath.sqrt(81*b**4 * M**2 - 3*b**6) - 9 * b**2 * M)
    denom3 = mpmath.sqrt(81*b**4 * M**2 - 3*b**6) - 9*b**2 * M
    denom = 3*denom3**(1/3)
    s = (num1 + num2) / denom
    return s


if __name__ == '__main__':
    M = 1
    solver_params = {'initial_guesses': 10,
                     'midpoint_iterations': 10,
                     'plot_inbetween': False,
                     'minP': 3.1 * M}
    # writeFramesEq13(5, solver_params=solver_params)
