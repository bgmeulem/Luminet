import numpy as np
import matplotlib.pyplot as plt
import mpmath
from typing import Tuple, Dict
from tqdm import tqdm

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


def Q(P_: float, M: float) -> float:
    """Convert Periastron distance P to Q (easier to work with)"""
    q = abs(mpmath.sqrt((P_ - (2 * M)) * (P_ + (6 * M))))
    # Q is complex if P < 2M
    return q


def b(P_: float, M: float, tol: float = 1e-5) -> float:
    """Get impact parameter b from Periastron distance P"""
    if P_ < tol:
        return 3 * P_ ** 2
    return P_ ** 3 / (P_ - 2 * M)  # the impact parameter


def P(b_, M):
    num1 = b_
    num2 = (mpmath.sqrt(3) * mpmath.sqrt(27. * M ** 2 * b_ ** 2 - b_ ** 3)
            - 9. * M * b_) ** (1 / 3)  # generally complex
    denom1 = num2 * 3 ** (1 / 3)
    denom2 = 3 ** (2 / 3)
    s = (num1 / denom1 + num2 / denom2).__abs__()  # generally a negligible complex part
    return s


def getRoots(P: float, M: float) -> (float, float, float):
    Qvar = Q(P, M)
    u1 = -(Qvar - P + 2 * M) / (4 * M * P)
    u2 = 1 / P
    u3 = (Qvar + P - 2 * M) + (4 * M * P)
    return u1, u2, u3


def k(P: float, M: float) -> float:
    """Calculate modulus of elliptic integral"""
    Qvar = Q(P, M)
    if Qvar < 10e-3:  # numerical stability
        return np.sqrt(.5)
    else:
        return np.sqrt((Qvar - P + 6 * M / (2 * Qvar)))  # the modulus of the ellipitic integral


def zeta_inf(P: float, M: float) -> float:
    """Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)"""
    Qvar = Q(P, M)  # Q variable, only call to function once
    a = (Qvar - P + 2 * M) / (Qvar - P + 6 * M)
    z_inf = mpmath.asin(mpmath.sqrt(a))
    return z_inf


def zeta_r(P: float, r: float, M: float) -> float:
    """Calculate the elliptic integral argument Zeta_r for a given value of P and r"""
    Qvar = Q(P, M)
    a = (Qvar - P + 2 * M + (4 * M * P) / r) / (Qvar - P + (6 * M))
    s = mpmath.asin(mpmath.sqrt(a))
    return s


def cos_gamma(a: float, incl: float, tol=10e-5) -> float:
    """Calculate the cos of the angle gamma"""
    if abs(incl) < tol:
        return 0
    else:
        return mpmath.cos(a) / mpmath.sqrt(mpmath.cos(a) ** 2 + mpmath.cot(incl) ** 2)  # real


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
    """Calculates the complete elliptic integral of mod m=k**2"""
    return mpmath.ellipf(np.pi / 2, m)


def filterP(P: np.ndarray, M: float, tol: float = 10e-4) -> Tuple[np.ndarray, np.ndarray]:
    """removes instances where P == 2*M
    returns indices where this was the case"""
    to_return = P
    to_remove = np.where([np.abs(p - 2 * M) < tol for p in P])[0]
    if len(to_remove) > 1:
        del to_return[to_remove[0]:to_remove[-1]]
    elif len(to_remove) == 1:
        to_return = np.delete(to_return, to_remove[0])
    return list(to_return), to_remove


def eq13(P: float, r: float, a: float, M: float, incl: float, n: int = 0) -> float:
    """Relation between radius (where photon was emitted in accretion disk), a and P
    P can be converted to b, yielding the polar coordinates (b, a) on the photographic plate"""
    zinf = zeta_inf(P, M)
    Qvar = Q(P, M)
    m_ = k(P, M) ** 2

    # Elliptic integral F(zinf, k)
    ellinf = F(zinf, m_)
    g = mpmath.acos(cos_gamma(a, incl))  # real
    # argument of sn
    if n:
        ellK = K(m_)
        ellips_arg = (g / 2 - n * np.pi) * mpmath.sqrt(P / Qvar) - ellinf + 2 * ellK
    else:
        ellips_arg = (g / 2) * mpmath.sqrt(P / Qvar) + ellinf  # complex

    # sn is an Jacobi elliptic function: elliptic sine. Takes 'sn'
    # as argument to specify "elliptic sine" and modulus m=k**2
    sn = mpmath.ellipfun('sn', ellips_arg, m=m_)
    term1 = -(Qvar - P + 2 * M) / (4 * M * P)
    term2 = ((Qvar - P + 6 * M) / (4 * M * P)) * (sn ** 2).real
    return -1 + r * (term1 + term2)  # should yield zero


def writeFramesEq13(radius: float, solver_params: Dict, incl: float = 10., M: float = 1.) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    fig.set_size_inches([5, 5])
    ax.set_xlabel('P')
    ax.set_ylabel('eq13(P, r, a)')

    root_precision = solver_params['root_precision']
    for n in tqdm(range(root_precision)):
        a = np.pi * n / root_precision

        def eq13_P(P, M_=M):
            return eq13(P, radius, a, M_, incl=incl)  # solve this equation for P

        s = findP(radius, incl, a, M, **solver_params)
        x = np.linspace(0.2, 1.01 * radius, root_precision)
        x, _ = filterP(x, M)
        y = [float(eq13_P(x_)) for x_ in x]

        ax.clear()
        ax.set_xlabel('P')
        ax.set_ylabel('eq13(P, r, a)')
        for solution in list(s):
            ax.scatter(solution, 0, color='red', zorder=10)
        ax.plot(x, y)
        plt.axhline(0, color='black')
        plt.title("Equation 13, r = {}\na = {}".format(radius, round(a, 5)))
        fig.savefig('movie/frame{:03d}.png'.format(n))


def findP(r, incl, alpha, M, midpoint_iterations=100, plot_inbetween=False,
          n=0, minP=2, initial_guesses=20):
    """Given a value for r (BH frame) and alpha (BH/observer frame), calculate the corresponding periastron value"""

    def eq13_P(P_, r_, alpha_, M_, incl_, n_):
        return eq13(P_, r_, alpha_, M_, incl_, n_)  # solve this equation for P

    def MidpointMethodP(x, y, ind, radius, angle, M_, inclination, n_):
        # for each P point, look around the point within expansion radius and find crossing more accurately
        # for each P point, look around the point within expansion radius and find crossing more accurately
        new_x = x
        new_y = y

        x_ = [x[ind], x[ind + 1]]  # interval of P values
        inbetween_P = np.mean(x_)
        new_x.insert(ind + 1, inbetween_P)  # insert middle P value to calculate

        y_ = [y[ind], y[ind + 1]]  # results of eq13 given the P values
        # calculate the P value inbetween
        inbetween_solution = eq13_P(P_=inbetween_P, r_=radius, alpha_=angle, M_=M_, incl_=inclination, n_=n_)
        new_y.insert(ind + 1, inbetween_solution)
        y_.insert(1, inbetween_solution)
        new_ind = ind + np.where(np.diff(np.sign(y_)))[0][0]

        return new_x, new_y, new_ind  # return x and y refined in relevant regions, as well as new index of sign change

    def improveP(P_, x, y, indices_of_signchange, iterations, radius, angle, M_, inclination, n_):
        """To increase precision.
        Only considers the function within the expansion radius around the solutions in P_.
        Searches again for a solution with a precision root_precision"""
        updated_P = P_
        indices_of_signchange_ = indices_of_signchange
        for i in range(len(indices_of_signchange_)):  # update each solution of P
            new_x = x
            new_y = y
            new_ind = indices_of_signchange[i]
            for iteration in range(iterations):
                new_x, new_y, new_ind = MidpointMethodP(new_x, new_y, new_ind, radius, angle, M_, inclination, n_)
            updated_P[i] = new_x[new_ind]
        return updated_P

    def getPlot(X, Y, solutions, radius=r):
        fig = plt.figure()
        plt.title("Eq13(P)\nr={}, a={}".format(radius, round(alpha, 5)))
        plt.xlabel('P')
        plt.ylabel('Eq13(P)')
        plt.axhline(0, color='black')
        plt.plot(X, Y)
        for P_ in solutions:
            plt.scatter(P_, 0, color='red')
        return plt

    x, _ = filterP(np.linspace(minP, 1.01 * r, initial_guesses), M)  # range of P values without P == 2*M
    y = [eq13_P(float(x_), r, alpha, M, incl, n) for x_ in x]  # values of eq13
    ind = np.where(np.diff(np.sign(y)))[0]
    P = [x[i] for i in ind]  # initial guesses
    if any(P):
        P = improveP(P, x, y, ind, midpoint_iterations, r, alpha, M, incl, n)  # get better P values
        if plot_inbetween:
            getPlot(x, y, P).show()
        return P
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


def makeXN(arr, x=1):
    """Expands list of radii to always contain x solutions
    If there is only one solution, simply duplicate them"""
    to_return = []
    for e in arr:
        expanded_element = e
        for n in range(x - len(e)):  # difference between current length and desired length of sub-array
            expanded_element.append(e[-1])
        to_return.append(expanded_element)
    return to_return


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
    return (1. + np.sqrt((M / radius ** 3)) * b_ * np.sin(incl) * np.sin(angle)) / np.sqrt((1. - 3. * M / radius))


def find_a(b_, z, incl, M, r_):
    """Given a certain redshift z (NOT 1+z) and radius b on the observer plaque, find the angle
    on the observer plaque. Include contributions from the disk at radii r."""
    radius = np.linspace(3 * M, 100 * M, len(b_)) if not r_ else r_

    sin_angle = ((1. + z) * np.sqrt(1. - 3. * M / radius) - 1) / ((M / radius ** 3) ** .5 * b_ * np.sin(incl))
    return np.arcsin(sin_angle)


if __name__ == '__main__':
    blist = []
    alist = []
    for r in [6, 10, 20, 30, 50]:
        b = np.linspace(3, 20)
        a = find_a(b, z=0.1, incl=10, M=1, r_=r)
        blist.append(b)
        alist.append(a)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    for b, a in zip(blist, alist):
        ax.plot(a, b)
    plt.show()
