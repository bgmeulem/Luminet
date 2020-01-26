import numpy as np
import matplotlib.pyplot as plt
import mpmath
import pylab as pl
from tqdm import tqdm
inclination = 5  # inclination above equatorial plane (disk)
t = np.pi/2 - inclination*2*np.pi/360
M = float(mpmath.sqrt(3))


def Q(P):
    q = abs(mpmath.sqrt((P - (2*M))*(P + (6*M))))
    # Q is complex if P < 2M
    return q


def zeta_inf(P):
    Qvar = Q(P)  # Q variable, only call to function once
    a = (Qvar - P + 2*M) / (Qvar - P + 6*M)
    return mpmath.asin(mpmath.sqrt(a))


def zeta_r(P, r):
    Qvar = Q(P)
    a = (Qvar - P + 2*M + (4*M*P)/r) / (Qvar - P + (6*M))
    s = mpmath.asin(mpmath.sqrt(a))
    return s


def F(zeta, m):
    # ellipf = F(theta, m), where m=k**2
    result = mpmath.ellipf(zeta, m)  # takes k**2 as mod, not k
    return result


def eq13(r, a, P):
    zinf = zeta_inf(P)
    Qvar = Q(P)
    m = (Qvar - P + 6*M)/(2*Qvar)  # this is already k**2

    # Elliptic integral F(zinf, k)
    ellinf = F(zinf, m)

    # angle gamma (equation 10)
    if a < np.pi/2:
        a = 2*np.pi-a  # Plots neater?
    g = mpmath.acos(mpmath.cos(a)/mpmath.sqrt(mpmath.cos(a)**2 + mpmath.cot(t)**2))
    # argument of sn
    ellipsarg = (g/2)*mpmath.sqrt(abs(P/Qvar)) + ellinf

    # sn is an Jacobi elliptic function: elliptic sine. Takes 'sn'
    # as argument to specify "elliptic sine" and modulus m=k**2
    sn = mpmath.ellipfun('sn', ellipsarg, m=m)  # q = k**2
    term1 = -(Qvar-P+2*M)/(4*M*P)
    term2 = ((Qvar-P+6*M)/(4*M*P))*sn.real**2
    return -1 + r*(term1 + term2)  # should yield zero


def findP(r, a):

    def eq13_P(P):
        return eq13(r=r, a=a, P=P)

    # Newtonian limit, P >> M
    newt = r * (1 + mpmath.tan(t) ** 2 * mpmath.cos(a) ** 2) ** (-.5)

    # initial values, quite sensitive
    if a < .5 * np.pi or a > (3 / 2) * np.pi:  # lower half of image
        initial = 2 * r + 3 * M
    else:
        initial = 3 * M

    # calculate P
    try:
        P = mpmath.findroot(eq13_P, initial)
    except:  # Newtonian limit, P- > inf, Q -> inf
        P = newt
    return P


def b(P):
    # in general complex?
    return (P**3)/(P-2*M)


def phi_inf(P):
    Qvar = Q(P)
    ksq = Qvar - P + 6*M/(2*Q)
    zinf = zeta_inf(P)
    phi = 2*(mpmath.sqrt(P/Qvar))*(mpmath.ellipk(ksq) - mpmath.ellipf(zinf, ksq))
    return phi


def mu(P):
    return 2*phi_inf(P) - np.pi


def isoradials(r, precision):
    # For fixed r, loop over all a, find all corresponding P
    isoradials = {}
    for radius in r:
        radii = []
        anglelist = []
        # apparent singularity at a = 0+2Npi?
        for a in tqdm(np.linspace(0, 2*np.pi, precision), position=0, desc='angle', leave=True):
            P = findP(radius, a)
            radii.append(float(P))
            anglelist.append(a)
        isoradials[radius] = [anglelist, radii]
    return isoradials


def plotIsoradials(r=[x*M for x in (6, 10, 20, 30)], precision=100, save=False):
    isoradiallist = isoradials(r=r, precision=precision)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location("S")
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    radii = [e for e in isoradiallist]
    ax.set_ylim([0, max(isoradiallist[max(radii)][1]) + 20])

    def ellipse(r, a, t=t):
        gamma = mpmath.acos(mpmath.cos(a)/mpmath.sqrt(mpmath.cos(a)**2 + mpmath.cot(t)**2))
        return r*mpmath.sin(gamma)

    # Draw circle of maximum radius around plot
    # ax.set_rgrids([max(r)], labels=[int(max(r)/M)], fontsize=20, color='grey')
    ax.set_rgrids([], labels=[], fontsize=20, color='grey')
    # unset theta grid. Might be useful to graphically determine where it shifts to newtonian limit
    ax.set_thetagrids(angles=[])

    titletext = str("isoradial " + r"P($\alpha$)" + "\n" +
                    r"$\theta_0$ = {}°".format(inclination) + "\n"
                    "metric: Schwarzschild \n" +
                    "radii: {} M".format([6, 10, 30, 60]) + "\n")
    plt.figtext(0.03, 0, titletext, fontfamily='book antiqua', color='white', fontsize=21,
                fontstyle='italic')

    # Plot Newtonian ellipses
    for element in isoradiallist:
        data = isoradiallist[element]
        plt.polar(data[0], [ellipse(element, p) for p in data[0]], color="grey", zorder=1)

    # plot horizon
    circle = pl.Circle((0, 0), 3 * M, transform=ax.transData._b, color="red")
    ax.add_artist(circle)


    # Plot isoradials
    for element in isoradiallist:
        data = isoradiallist[element]
        plt.polar(data[0], data[1], color='white')
    plt.show()

    if save==True:
        fig.savefig("Isoradials(t={}).png".format(inclination), dpi=300, facecolor='black')


def findR(z, a):

    def P(r):
        return findP(r, a=a)

    def tosolve(r):
        P_ = P(r)
        b_ = b(P_)
        return -1 - z + ((1 - 3 * M / r) ** (-.5) * (1 + (M/r**3)**.5 * b_ * mpmath.sin(t) * mpmath.sin(a)))

    s = mpmath.findroot(tosolve, 100)
    return s


# TODO: might make more sense to loop over a first and then look for the r of certain z's?
def isoredshifts(z, precision):
    radii = []
    angles = []
    s = {}
    for redshift in z:
        for a in tqdm(np.linspace(0, np.pi, precision)):
            try:
                radius = findR(redshift, a)
                print(float(radius.real))
                # TODO: doesn't seem to find any sensible roots
            except ValueError:
                continue
            radii.append(float(radius.real))
            angles.append(a)
        s[redshift] = [angles, radii]
    return s


def plotIsoredshifts(z=(round(n*0.05, 2) for n in range(1, 4, 1)), precision=20, save=False):
    isoredshiftlist = isoredshifts(z=z, precision=precision)
    print(isoredshiftlist)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location("S")
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    radii = [e for e in isoredshiftlist]
    ax.set_ylim([0, max(isoredshiftlist[max(radii)][1]) + 20])
    print(max(isoredshiftlist[max(radii)][1]) + 10)
    ax.set_rgrids([], labels=[], color='grey')
    ax.set_thetagrids(angles=[])
    titletext = str("isoredshift " + r"z($\alpha$)" + "\n" +
                    r"$\theta_0$ = {}°".format(inclination) + "\n"
                                                              "metric: Schwarzschild \n" +
                    "redshifts: {}".format(z) + "\n")
    plt.figtext(0.03, 0, titletext, fontfamily='book antiqua', color='white', fontsize=21,
                fontstyle='italic')

    for element in isoredshiftlist:
        print(isoredshiftlist[element][1])
        data = isoredshiftlist[element]
        plt.polar(data[0], data[1], color='white')
    plt.show()

    if save == True:
        fig.savefig("Isoradials(t={}).png".format(inclination), dpi=300, facecolor='black')



# plotIsoradials(precision=1000, save=False)
plotIsoredshifts(z = [0.5], precision=10, save=False)


