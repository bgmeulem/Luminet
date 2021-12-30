import numpy as np
import matplotlib.pyplot as plt
import mpmath
from typing import Tuple, Dict
from tqdm import tqdm

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


def Q(P_: float, M: float) -> float:
    """Convert Periastron distance P to Q (easier to work with)"""
    q = mpmath.sqrt((P_ - 2. * M) * (P_ + 6. * M))
    # Q is complex if P < 2M = r_s
    return q


def calc_b_from_P(P_: float, M: float, tol: float = 1e-5) -> float:
    """Get impact parameter b from Periastron distance P"""
    if P_ < tol:  # could physically never happen
        print("tolerance exceeded for calc_b_from_P(P_={}, M={}, tol={}".format(P_, M, tol))
        return mpmath.sqrt(3 * P_ ** 2)
    # WARNING: the paper most definitely has a typo here. The fracture on the right hand side equals b², not b.
    # Just fill in u_2 in equation 3, and you'll see. Only this way do the limits P -> 3M and P >> M hold true,
    # as well as the value for b_c
    return mpmath.sqrt(P_ ** 3 / (P_ - 2. * M))  # the impact parameter


def k(P: float, M: float) -> float:
    """Calculate modulus of elliptic integral"""
    Qvar = Q(P, M)
    if Qvar < 10e-3:  # numerical stability
        return mpmath.sqrt(.5)
    else:
        # WARNING: Paper has an error here. There should be brackets around the numerator.
        return mpmath.sqrt((Qvar - P + 6 * M) / (2 * Qvar))  # the modulus of the ellipitic integral


def k2(_P: float, M: float, tol: float = 1e-6):
    """Calculate the squared modulus of elliptic integral"""
    Qvar = Q(_P, M)
    if Qvar.real < tol:  # numerical stability
        print("tolerance exceeded in function k2(_P = {}, M={}, tol={})".format(_P, M, tol))
        return .5
    # TODO: add inf / inf
    else:
        # WARNING: Paper has an error here. There should be brackets around the numerator.
        return (Qvar - _P + 6 * M) / (2 * Qvar)  # the modulus of the ellipitic integral


def zeta_inf(_P: float, M: float, tol: float = 1e-6) -> float:
    """Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)"""
    Qvar = Q(_P, M)  # Q variable, only call to function once
    if (1 / Qvar).real < tol:
        print("tolerance exceeded in function zeta_inf(_P={}, M={}, tol={}".format(_P, M, tol))
        return 1.
    arg = (Qvar - _P + 2 * M) / (Qvar - _P + 6 * M)
    z_inf = mpmath.asin(mpmath.sqrt(arg))
    return z_inf


def zeta_r(_P: float, r: float, M: float) -> float:
    """Calculate the elliptic integral argument Zeta_r for a given value of P and r"""
    Qvar = Q(_P, M)
    a = (Qvar - _P + 2 * M + (4 * M * _P) / r) / (Qvar - _P + (6 * M))
    s = mpmath.asin(mpmath.sqrt(a))
    return s


def cos_gamma(_a: float, incl: float, tol=10e-5) -> float:
    """Calculate the cos of the angle gamma"""
    if abs(incl) < tol:
        return 0
    else:
        return mpmath.cos(_a) / mpmath.sqrt(mpmath.cos(_a) ** 2 + mpmath.cot(incl) ** 2)  # real


def cosAlpha(phi: float, incl: float) -> float:
    """Returns cos(angle) alpha in observer frame given angles phi (black hole frame) and
    inclination (black hole frame)"""
    return mpmath.cos(phi) * mpmath.cos(incl) / mpmath.sqrt((1 - mpmath.sin(incl) ** 2 * mpmath.cos(phi) ** 2))


def alpha(phi: float, incl: float):
    """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)"""
    return mpmath.acos(cosAlpha(phi, incl))


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


def filterP(P: np.ndarray, M: float, tol: float = 10e-4) -> []:
    """removes instances where P == 2*M
    returns indices where this was the case"""
    return [e for e in P if abs(e - 2 * M) > tol]


def eq13(_P: float, _r: float, _a: float, M: float, incl: float, n: int = 0) -> float:
    """Relation between radius (where photon was emitted in accretion disk), a and P.
    P can be converted to b, yielding the polar coordinates (b, a) on the photographic plate"""
    zinf = zeta_inf(_P, M)
    Qvar = Q(_P, M)
    m_ = k2(_P, M)  # modulus of the elliptic integrals. mpmath takes m = k² as argument.
    ellinf = F(zinf, m_)  # Elliptic integral F(zinf, k)
    g = mpmath.acos(cos_gamma(_a, incl))  # real

    # Calculate the argument of sn (mod is m = k², same as the original elliptic integral)
    # WARNING: paper has an error here: \sqrt(P / Q) should be in denominator, not numerator
    # There's no way that \gamma and \sqrt(P/Q) can end up on the same side of the division
    if n:  # higher order image
        ellK = K(m_)  # calculate complete elliptic integral of mod m = k²
        ellips_arg = (g - 2. * n * np.pi) / (2. * mpmath.sqrt(_P / Qvar)) - ellinf + 2. * ellK
    else:  # direct image
        ellips_arg = g / (2. * mpmath.sqrt(_P / Qvar)) + ellinf  # complex

    # sn is an Jacobi elliptic function: elliptic sine. ellipfun() takes 'sn'
    # as argument to specify "elliptic sine" and modulus m=k²
    sn = mpmath.ellipfun('sn', ellips_arg, m=m_)
    sn2 = sn * sn
    sn2 = sn2.real  # generally a negligible complex part
    # sn2 = float(sn2.real)
    term1 = -(Qvar - _P + 2. * M) / (4. * M * _P)
    term2 = ((Qvar - _P + 6. * M) / (4. * M * _P)) * sn2
    # TODO: log is easier to solve maybe? -> nope: intersection is more clear, but no change in precision
    # TODO: and takes about twice as long to calculate
    return float(1. - _r * (float(term1) + float(term2.real)))  # solve this for zero


def writeFramesEq13(radius: float, solver_params: Dict, incl: float = 10., M: float = 1.,
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

        s = calcP(radius, incl, a, M, **solver_params)
        x = np.linspace(0.2, 1.01 * radius, 100)
        x = filterP(x, M)
        y = [eq13_P(x_) for x_ in x]

        ax.clear()
        ax.set_xlabel('P')
        ax.set_ylabel('eq13(P, r, a)')
        if s is not None:
            for solution in list(s):
                ax.scatter(solution, 0, color='red', zorder=10)
        ax.plot(x, y)
        plt.axhline(0, color='black')
        plt.title("Equation 13, r = {}\na = {}".format(radius, round(a, 5)))
        fig.savefig('movie/eq13/frame{:03d}.png'.format(n))


def calcP(_r, incl, _alpha, M, midpoint_iterations=100, plot_inbetween=False,
          n=0, minP=2, initial_guesses=20):
    """Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value"""

    def eq13_P(__P, __r, __alpha, __M, __incl, __n):
        s = eq13(__P, _r=__r, _a=__alpha, M=__M, incl=__incl, n=__n)  # solve this equation for P
        return s

    def MidpointMethodP(__x, __y, __ind, __radius, __angle, __M, __inclination, __n):
        new_x = __x
        new_y = __y

        x_ = [new_x[__ind], new_x[__ind + 1]]  # interval of P values
        inbetween_P = np.mean(x_)
        new_x.insert(__ind + 1, inbetween_P)  # insert middle P value to calculate

        y_ = [new_y[__ind], new_y[__ind + 1]]  # results of eq13 given the P values
        # calculate the P value inbetween
        inbetween_solution = eq13_P(__P=inbetween_P, __r=__radius, __alpha=__angle, __M=__M, __incl=__inclination,
                                    __n=__n)
        new_y.insert(__ind + 1, inbetween_solution)
        y_.insert(1, inbetween_solution)
        ind_of_sign_change_ = np.where(np.diff(np.sign(y_)))[0]
        new_ind = __ind + ind_of_sign_change_[0]

        return new_x, new_y, new_ind  # return x and y refined in relevant regions, as well as new index of sign change

    def improveP(__P, x, y, indices_of_sign_change, iterations, radius, angle, M_, inclination, n_):
        """To increase precision.
        Searches again for a solution inbetween the interval where the sign changes
        Does this <iterations> times"""
        updated_P = __P
        indices_of_sign_change_ = indices_of_sign_change
        new_x = x
        new_y = y
        for i in range(len(indices_of_sign_change_)):  # update each solution of P
            new_ind = indices_of_sign_change_[i]  # location in X and Y where eq13(P=X[ind]) equals Y=0
            for iteration in range(iterations):
                new_x, new_y, new_ind = MidpointMethodP(new_x, new_y, new_ind, radius, angle, M_, inclination, n_)
            updated_P[i] = new_x[new_ind]
            indices_of_sign_change_ = [e + iterations for e in indices_of_sign_change_]
        return updated_P

    def getPlot(X, Y, solutions, radius=_r):
        fig = plt.figure()
        plt.title("Eq13(P)\nr={}, a={}".format(radius, round(_alpha, 5)))
        plt.xlabel('P')
        plt.ylabel('Eq13(P)')
        plt.axhline(0, color='black')
        plt.plot(X, Y)
        for P_ in solutions:
            plt.scatter(P_, 0, color='red')
        return plt

    # TODO: an x_range until 1.1*R seems to suffice for isoradials < 30M, but this is guesstimated
    x_ = list(np.linspace(minP, 1.1 * _r, initial_guesses))  # range of P values without P == 2*M
    y_ = [eq13_P(P_value, _r, _alpha, M, incl, n) for P_value in x_]  # values of eq13
    ind = np.where(np.diff(np.sign(y_)))[0]
    _P = [x_[i] for i in ind]  # initial guesses
    if any(_P):
        _P = improveP(_P, x_, y_, ind, midpoint_iterations, _r, _alpha, M, incl, n)  # get better P values
        if plot_inbetween:
            getPlot(x_, y_, _P).show()
        return _P
    else:
        # TODO: implement newtonian ellipse
        return None


def phi_inf(P, M):
    Qvar = Q(P, M)
    ksq = (Qvar - P + 6. * M) / (2. * Qvar)
    zinf = zeta_inf(P, M)
    phi = 2. * (np.sqrt(P / Qvar)) * (np.ellipk(ksq) - np.ellipf(zinf, ksq))
    return phi


def mu(P, M):
    return 2 * phi_inf(P, M) - np.pi


def ellipse(r, a, incl):
    """Equation of an ellipse, reusing the definition of cos_gamma.
    This equation can be used for calculations in the Newtonian limit (large P = b, small a)
    or to visualize the equatorial plane."""
    g = mpmath.acos(cos_gamma(a, incl))
    b_ = r * mpmath.sin(g)
    return b_


def flux_intrinsic(r, acc, M):
    r_ = r / M
    log_arg = ((np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
              ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
    f = (3. * M * acc / (8 * np.pi)) * (1 / ((r_ - 3) * r ** 2.5)) * \
        (np.sqrt(r_) - np.sqrt(6) + 3 ** -.5 * np.log10(log_arg))
    return f


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


if __name__ == '__main__':
    M = 1
    solver_params = {'initial_guesses': 10,
                     'midpoint_iterations': 7,
                     'plot_inbetween': False,
                     'minP': 3.1 * M}
    writeFramesEq13(30, solver_params=solver_params)
