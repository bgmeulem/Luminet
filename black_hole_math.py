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
    if abs(P_) < tol:  # could physically never happen
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
    # TODO: add inf / inf
    # WARNING: Paper has an error here. There should be brackets around the numerator.
    return (Qvar - _P + 6 * M) / (2 * Qvar)  # the modulus of the ellipitic integral


def zeta_inf(_P: float, M: float, tol: float = 1e-6) -> float:
    """Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)"""
    Qvar = Q(_P, M)  # Q variable, only call to function once
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


def filterP(P: np.ndarray, M: float, tol: float = 10e-3) -> []:
    """removes instances where P == 2*M
    returns indices where this was the case"""
    return [e for e in P if abs(e - 2. * M) > tol]


def eq13(_P: float, _r: float, _a: float, M: float, incl: float, n: int = 0, tol=10e-6) -> float:
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
    # sn2 = float(sn2.real)
    term1 = -(Qvar - _P + 2. * M) / (4. * M * _P)
    term2 = ((Qvar - _P + 6. * M) / (4. * M * _P)) * sn2

    s = 1. - _r * (term1 + term2)  # solve this for zero
    return s


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

        s = calcImpactParameter(radius, incl, a, M, **solver_params, use_ellipse=False)
        x = np.linspace(solver_params["minP"], 1.01 * radius, 100)
        x_range = filterP(x, M)
        y = [eq13_P(x_).real for x_ in x_range]
        x = [calc_b_from_P(p, M) for p in filterP(x, M)]

        ax.clear()
        ax.set_xlabel('P')
        ax.set_ylabel('eq13(P, r, a)')
        if s is not None:
            ax.scatter(s.real, 0, color='red', zorder=10)
        ax.plot(x, y)
        plt.axhline(0, color='black')
        plt.title("Equation 13, r = {}\na = {}".format(radius, round(a, 5)))
        fig.savefig('movie/eq13/frame{:03d}.png'.format(n))


def calcImpactParameter(_r, incl, _alpha, _M, midpoint_iterations=100, plot_inbetween=False,
                        n=0, minP=1, initial_guesses=20, elliptic_integral_interval=None, use_ellipse=True,
                        tol_critical_b=0.1):
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
        ind_of_sign_change_ = np.where(np.diff(np.sign(y_)))[0]  # TODO: how to deal with complex
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

    if elliptic_integral_interval is None:
        elliptic_integral_interval = (70 * np.pi / 180,
                                      2 * np.pi - 70 * np.pi / 180)

    # TODO: an x_range until 1.1*R seems to suffice for isoradials < 30M, but this is guesstimated
    x_ = list(np.linspace(minP, 2. * _r, initial_guesses))  # range of P values without P == 2*M
    x_ = filterP(x_, _M)
    y_ = [eq13_P(P_value, _r, _alpha, _M, incl, n) for P_value in x_]  # values of eq13
    y_ = [y.real if y.imag < 10e-6 else y for y in y_]  # drop tiny imaginary parts
    if any([y.imag >= 10e-6 for y in y_]):
        y1 = [float(y.real) for y in y_]
        y2 = [y.imag for y in y_]  # for testing
        y3 = [y.real + y.imag for y in y_]  # for testing
        # fig = plt.figure()
        # plt.plot(x_, y1)
        # plt.plot(x_, y2)
        # plt.show()
        y_ = y1
    ind = np.where(np.diff(np.sign(y_)))[0]
    _P = [x_[i] for i in ind]  # initial guesses for P
    if any([abs(p.real - 3*_M) < tol_critical_b for p in _P]):  # near critical limit
        return 5.19695*_M + 3.4823*_M*np.exp(-mu(_P[-1].real, _M))
    # If image is ghost image, or direct image with lots of curvature: calculate with elliptic integrals
    if len(_P) and \
            (elliptic_integral_interval[0] < _alpha < elliptic_integral_interval[1] or n or not use_ellipse):
        # Assume if no P is found, it's because it's in the region without periastron
        # TODO: check if this assumption is correct, throw error
        # Side or backside of disk: calculate impact parameter according to elliptic integrals
        _P = improveP([P.real for P in _P], x_, y_, ind, midpoint_iterations, _r, _alpha, _M, incl, n)  # get better P values

        if plot_inbetween:
            getPlot(x_, y_, _P).show()
        # TODO: how to correctly assume there is only one solution
        _P = max(_P)
        return float(calc_b_from_P(_P, _M).real)
    elif use_ellipse:  # Front side of disk: calculate impact parameter with the ellipse formula
        return float(ellipse(_r, _alpha, incl))
    else:
        # why was no P found?
        # fig = plt.figure()
        # plt.plot(x_, y_)
        # plt.show()
        return None  # TODO: implement


def phi_inf(P, M):
    Qvar = Q(P, M)
    ksq = (Qvar - P + 6. * M) / (2. * Qvar)
    zinf = zeta_inf(P, M)
    phi = 2. * (mpmath.sqrt(P / Qvar)) * (mpmath.ellipk(ksq) - mpmath.ellipf(zinf, ksq))
    return phi


def mu(P, M):
    return float(2 * phi_inf(P, M) - np.pi)


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


def flux_observed(r, acc, M, redshift_factor):
    flux_intr = flux_intrinsic(r, acc, M)
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
    num1 = 3**(2/3)*b**2
    num2 = 3**(1/3) * (mpmath.sqrt(81*b**4 * M**2 - 3*b**6) - 9 * b**2 * M)
    denom3 = mpmath.sqrt(81*b**4 * M**2 - 3*b**6) - 9*b**2 * M
    denom = 3*denom3**(1/3)
    s = (num1 + num2) / denom
    return s


def find_br(z, alpha, M, incl, n=0):
    """Does not work, function is too complicated to solve"""
    def b(__r, __z, __M, __incl, __alpha):
        """Find impact parameter b for a given z, M, inclination and alpha in function of r."""
        s = ((1 + __z) * mpmath.sqrt(1 - 3 * __M / __r)) * (__r * mpmath.sqrt(__r / __M) /
                                                            (mpmath.sin(__incl) * mpmath.sin(__alpha)))
        return s

    def P(r, __z, __M, __incl, __alpha):
        return getPFromB(b(r, __z, __M, __incl, __alpha), __M)

    def to_solve(r, __z, __M, __incl, __alpha, __n):
        _P = P(r, __z, __M, __incl, __alpha)
        zinf = zeta_inf(_P, __M)
        Qvar = Q(_P, __M)
        m_ = k2(_P, __M)  # modulus of the elliptic integrals. mpmath takes m = k² as argument.
        ellinf = F(zinf, m_)  # Elliptic integral F(zinf, k)
        g = mpmath.acos(cos_gamma(__alpha, __incl))  # real

        # Calculate the argument of sn (mod is m = k², same as the original elliptic integral)
        # WARNING: paper has an error here: \sqrt(P / Q) should be in denominator, not numerator
        # There's no way that \gamma and \sqrt(P/Q) can end up on the same side of the division
        if __n:  # higher order image
            ellK = K(m_)  # calculate complete elliptic integral of mod m = k²
            ellips_arg = (g - 2. * __n * np.pi) / (2. * mpmath.sqrt(_P / Qvar)) - ellinf + 2. * ellK
        else:  # direct image
            ellips_arg = g / (2. * mpmath.sqrt(_P / Qvar)) + ellinf  # complex

        # sn is an Jacobi elliptic function: elliptic sine. ellipfun() takes 'sn'
        # as argument to specify "elliptic sine" and modulus m=k²
        sn = mpmath.ellipfun('sn', ellips_arg, m=m_)
        sn2 = sn * sn
        # sn2 = float(sn2.real)
        term1 = -(Qvar - _P + 2. * __M) / (4. * __M * _P)
        term2 = ((Qvar - _P + 6. * __M) / (4. * __M * _P)) * sn2
        return -1 + r*(term1 + term2)

    X = np.linspace(3.01*M, 4*M, 1000)
    Y = [to_solve(x, z, M, incl, alpha, n) for x in X]

    Y_imag = [y.imag for y in Y]
    Y = [y.real for y in Y]
    dy = [(e.real-b.real) for b, e in zip(Y[:-1], Y[1:])]
    dx = 1/1000
    too_steep = [i for i in range(len(Y) - 1) if dy[i]/dx > 10]

    # zero_imag = np.where(np.diff(np.sign([y.imag for y in Y])))[0]
    # zero = [e for e in zero_real if e in zero_imag]
    plt.plot(X, Y_imag, alpha=.2)
    plt.plot(X, Y)
    plt.xlabel('radius')
    plt.ylabel('Yield 0 to find radius')
    plt.tight_layout()
    plt.ylim(-25, 25)
    plt.show()
    print("Radius = {}".format(X[zero_real]))


if __name__ == '__main__':
    M = 1
    solver_params = {'initial_guesses': 10,
                     'midpoint_iterations': 10,
                     'plot_inbetween': False,
                     'minP': 3.1 * M}
    # writeFramesEq13(5, solver_params=solver_params)
    find_br(-.05, alpha=np.pi/2, M=1, incl=70*np.pi/180)
