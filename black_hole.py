import matplotlib.colors as clrs
import matplotlib.cm as cm
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np

from black_hole_math import *

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


class BlackHole:
    def __init__(self, mass=1, inclination=10, acc=10e-8):
        """Initialise black hole with mass and accretion rate
        Set viewer inclination above equatorial plane
        """
        self.t = inclination * np.pi / 180
        self.M = mass
        self.acc = acc  # accretion rate
        self.angular_properties = {'start_angle': 0,
                                   'angular_precision': 60}
        self.solver_params = {'initial_guesses': 20,
                              'midpoint_iterations': 6,
                              'plot_inbetween': False,
                              'minP': 3.1 * self.M}
        self.plot_params = {'save_plot': True,
                            'plot_ellipse': False,
                            'key': "",
                            'face_color': 'black',
                            'line_color': 'white',
                            'text_color': 'white',
                            'alpha': 1.,
                            'show_grid': False,
                            'legend': False,
                            'title': "Isoradials for M = {}".format(self.M)}
        self.isoradials = {}
        self.isoredshifts = {}

    def setInclination(self, incl):
        self.t = incl * np.pi / 180

    def calcIsoRedshifts(self, minR, maxR, r_precision=100, midpoint_steps=10,
                         redshifts=[-.15, -.1, -.05, 0., .05, .1, .15, .20, .25, .5]):
        def calcIsoRedshift(redshift, dirty_isoradials, midpoint_steps):
            """Calculates the isoredshift for a single redshift value"""
            solutions = []
            t = tqdm(dirty_isoradials, position=1, leave=False)
            for ir in dirty_isoradials:
                t.set_description("Calculating isoredshift {} at R={:.2f}".format(redshift, ir.radius))
                t.update()
                angles_, radii_ = ir.calcRedshiftLocation(redshift, midpoint_steps)
                for s in range(len(angles_)):
                    solutions.append([angles_[s], radii_[s]])
                # else:
                #     print("No solution found at z={}, R={}".format(redshift, ir.radius))
            return solutions

        def getDirtyIsoradials(minR, maxR, r_precision, angular_precision=20):
            isoradials = []  # for initial guesses
            for radius in np.linspace(minR, maxR, r_precision):  # calculate the initial guesses
                isoradial = Isoradial(radius, self.t, self.M)
                isoradial.angular_properties = {'start_angle': 0,
                                                'end_angle': np.pi,
                                                'angular_precision': angular_precision,
                                                'mirror': True}
                isoradial.calculateCoordinates()
                isoradial.calcRedshiftFactors()
                isoradials.append(isoradial)
            return isoradials

        dirty_isoradials = getDirtyIsoradials(minR, maxR, r_precision)
        isoredshifts = {}
        t = tqdm(redshifts, desc="Calculating redshift", position=0)
        for redshift in t:
            t.set_description("Calculating redshift {}".format(redshift))
            dirty_isoradials_copy = dirty_isoradials  # to mutate
            coordinates = calcIsoRedshift(redshift, dirty_isoradials_copy, midpoint_steps=midpoint_steps)
            isoredshifts[redshift] = coordinates
        self.isoredshifts = isoredshifts
        return isoredshifts

    def plotIsoradials(self, direct_r: [], ghost_r: [], y_lim=None):
        """Given an array of radii for the direct image and/or ghost image, plots the corresponding
        isoradials.
        Calculates the isoradials according to self.root_params
        Plots the isoradials according to self.plot_params"""

        def plotEllipse(r_, ax):
            ax_ = ax
            a = np.linspace(-np.pi, np.pi, 2 * self.plot_params['angular_precision'])
            scale = 1. / mpmath.acos(cos_gamma(0, self.t))
            ell = [ellipse(r_ * scale, a_, self.t) for a_ in a]
            ax_.plot(a, ell)
            return ax_

        fig = plt.figure(figsize=(10, 10))
        ax_ = fig.add_subplot(111, projection='polar')
        ax_.set_theta_zero_location("S")
        fig.patch.set_facecolor(self.plot_params['face_color'])
        ax_.set_facecolor(self.plot_params['face_color'])
        if self.plot_params['show_grid']:
            ax_.grid(color='grey')
            ax_.tick_params(which='both', labelcolor=self.plot_params['text_color'],
                            labelsize=15)
        else:
            ax_.grid()
            ax_.spines['polar'].set_visible(False)

        color_range = (-1, 1)

        # plot ghost images
        self.plot_params['line_color'] = 'grey'  # default value in case of no redshift
        self.plot_params['alpha'] = .5
        for radius in tqdm(sorted(ghost_r), desc='Ghost Image', position=1, leave=False):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=1)
            isoradial.solver_params = self.solver_params
            isoradial.calculate()
            ax_ = isoradial.plot(ax_, self.plot_params, colornorm=color_range)

        # plot direct images
        self.plot_params['line_color'] = 'white'  # default value in case of no redshift
        self.plot_params['alpha'] = 1.
        for i, radius in enumerate(tqdm(sorted(direct_r), desc='Direct Image', position=1, leave=False)):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=0)
            isoradial.solver_params = self.solver_params
            isoradial.calculate()
            ax_ = isoradial.plot(ax_, self.plot_params, colornorm=color_range)

        if self.plot_params['plot_ellipse']:  # plot ellipse
            for radius in direct_r:
                ax_ = plotEllipse(radius, ax_)

        ax_.autoscale_view(scalex=False)
        ylim = y_lim if y_lim != () else [0, ax_.get_ylim()]
        ax_.set_ylim(ylim)  # assure the radial axis of a polar plot makes sense and starts at 0
        plt.title(self.plot_params['title'], color=self.plot_params['text_color'])
        plt.show()
        if self.plot_params['save_plot']:
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            fig.savefig(name, dpi=300, facecolor=self.plot_params['face_color'])

    def writeFrames(self, direct_r=[6, 10, 20, 30], ghost_r=[6, 10, 20, 30], start=0, end=180, stepsize=5,
                    y_lim=(0, 130)):
        # For an isoradial of R = 30*M, the maximum impact parameter is about 123.7
        steps = np.linspace(start, end, 1 + (end - start) // stepsize)
        for a in tqdm(steps, position=0, desc='Writing frames'):
            self.setInclination(a)
            bh.plot_params['title'] = 'inclination = {:03}°'.format(int(a))
            bh.plotIsoradials(direct_r, ghost_r, y_lim=y_lim)

    def plotIsoRedshifts(self, minR, maxR, r_precision, midpoint_steps=5,
                         redshifts=[-.5, -.35, -.15, 0., .15, .25, .5, .75, 1.]):
        isoredshifts = bh.calcIsoRedshifts(minR, maxR, r_precision=r_precision, midpoint_steps=midpoint_steps,
                                           redshifts=redshifts)
        print(isoredshifts)
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.set_theta_zero_location("S")
        # ax.set_ylim([0, 100])

        i = 0
        for z in isoredshifts:
            color = colors[i % 6]
            i += 1
            coordinates = isoredshifts[z]
            co1 = []
            co2 = []
            for co in coordinates:
                if np.pi / 2 < co[0] <= 1.5 * np.pi:
                    co1.append(co)
                else:
                    co2.append(co)
            if len(co1):
                ax.plot(np.array(co1)[:, 0], np.array(co1)[:, 1], color=color, label=z)
            if len(co2):
                ax.plot(np.array(co2)[:, 0], np.array(co2)[:, 1], color=color)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.suptitle("Isoredshift lines for M={}".format(self.M))
        plt.show()


class Isoradial:
    # TODO: fix. Black hole *has* isoradials -> no inheritance, but attribute
    # TODO: now, isoradials initializes a black hole with default parameters (incline of 10)
    def __init__(self, R, incl, mass, order=0):
        self.M = mass  # mass of the black hole containing this isoradial
        self.t = incl  # inclination of observer's plane
        self.radius = R
        self.order = order
        self.angular_properties = {'start_angle': 0.,
                                   'end_angle': np.pi,
                                   'angular_precision': 60,
                                   'mirror': True}
        self.solver_params = {'initial_guesses': 20,
                              'midpoint_iterations': 5,
                              'plot_inbetween': False,
                              'minP': 3.1 * self.M}
        self.plot_params = {'save_plot': True,
                            'plot_ellipse': False,
                            'key': "",
                            'face_color': 'black',
                            'line_color': 'white',
                            'text_color': 'white',
                            'alpha': 1.,
                            'show_grid': False,
                            'legend': False,
                            'title': "Isoradials for R = {}".format(R)}
        self.radii_b = []
        self.angles = []
        self.periastrons = []
        self.redshift_factors = []

    def calculateCoordinates(self):
        """Calculates the angles (alpha) and radii (b) of the photons emitted at radius self.radius as they would appear
        on the observer's photographic plate. Also saves the corresponding values for the impact parameters (P).

        Args:

        Returns:
            tuple: Tuple containing the angles (alpha) and radii (b) for the image on the observer's photographic plate
        """

        start_angle = self.angular_properties['start_angle']
        end_angle = self.angular_properties['end_angle']
        angular_precision = self.angular_properties['angular_precision']

        angles = []
        impact_parameters = []
        periastrons = []
        for alpha_ in tqdm(np.linspace(start_angle, end_angle, angular_precision),
                           desc='Calculating isoradial R = {}'.format(self.radius), position=2, leave=False):
            b_, P_ = self.calc_bP(alpha_)
            if P_ is not None:
                periastrons.append(P_)
                angles.append(alpha_)
                impact_parameters.append(b_)
        if self.order > 0:
            angles = [a_ + np.pi for a_ in angles]

        if self.angular_properties['mirror']:  # by default True. Halves computation time for calculating full isoradial
            # add second half of image (left half if 0° is set at South)
            angles += [2 * np.pi - a_ for a_ in angles[::-1]]
            impact_parameters += impact_parameters[::-1]

        # flip image if necessary
        if self.t > np.pi / 2:
            angles = [a_ + np.pi for a_ in angles]
        self.angles = angles
        self.radii_b = impact_parameters
        self.periastrons = periastrons
        return angles, impact_parameters

    def calcRedshiftFactors(self):
        """Calculates the redshift factor (1 + z) over the line of the isoradial"""
        redshift_factors = [redshift_factor(radius=self.radius, angle=angle, incl=self.t, M=self.M, b_=b_)
                            for b_, angle in zip(self.radii_b, self.angles)]
        self.redshift_factors = redshift_factors
        return redshift_factors

    def calculate(self):
        self.calculateCoordinates()
        self.calcRedshiftFactors()

    def findAngle(self, z) -> [int]:
        """Returns angle at which the isoradial redshift equals some value z
        Args:
            z: The redshift value z. Do not confuse with redshift factor 1 + z"""
        indices = np.where(np.diff(np.sign([redshift - z - 1 for redshift in self.redshift_factors])))[0]
        return [self.angles[i] for i in indices if len(indices)]

    def get_b_from_angle(self, angle: float):
        signs = np.sign([a_ - angle for a_ in self.angles])
        diff = np.diff(signs)
        indices = np.where(diff)[0]  # locations where sign switches from -1 to 0 or from 0 to 1
        return self.radii_b[indices[0]] if len(indices) else None

    def calc_bP(self, alpha):
        P_ = calcP(r=self.radius, incl=self.t, alpha=alpha, M=self.M, n=self.order, **self.solver_params)
        if P_:
            P_ = P_[-1] if len(P_) else None  # pick biggest one
            b_ = float(b(P_, self.M))
            return b_, P_
        else:
            return None, None

    def plot(self, ax_=None, plot_params=None, show=False, colornorm=(0, 1)):
        def make_segments(x, y):
            """
            Create list of line segments from x and y coordinates, in the correct format
            for LineCollection: an array of the form numlines x (points per line) x 2 (x
            and y) array
            """

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            return segments

        def colorline(
                _ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(*colornorm),
                linewidth=3):
            """
            http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
            http://matplotlib.org/examples/pylab_examples/multicolored_line.html
            Plot a colored line with coordinates x and y
            Optionally specify colors in the array z
            Optionally specify a colormap, a norm function and a line width
            """

            # Default colors equally spaced on [0,1]:
            if z is None:
                z = np.linspace(0.0, 1.0, len(x))

            # Special case if a single number:
            if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
                z = np.array([z])

            z = np.asarray(z)

            segments = make_segments(x, y)
            lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                                      linewidth=linewidth, alpha=self.plot_params['alpha'])
            lc.set_array(z)
            _ax.add_collection(lc)
            mx = max(segments[:][:, 1].flatten())
            _ax.set_ylim((0, mx))
            return _ax

        if not ax_:
            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')
        else:
            ax = ax_

        if not plot_params:
            plot_params = self.plot_params

        # Plot isoradial
        if len(self.redshift_factors):
            ax = colorline(ax, self.angles, self.radii_b, z=[e - 1 for e in self.redshift_factors],
                           cmap=cm.get_cmap('RdBu'))
        else:
            ax.plot(self.angles, self.radii_b, color=plot_params['line_color'],
                    alpha=plot_params['alpha'])
        if self.plot_params['legend']:
            plt.legend(prop={'size': 16})
        if show:
            ax.autoscale_view(scalex=False)
            ax.set_ylim([0, ax.get_ylim()[1] * 1.1])
            plt.show()
        return ax

    def calcBetween(self, ind):
        """
        Calculates the impact parameter, redshift factor and observer photographic plate coordinates at the
        isoradial location between place ind and ind + 1

        Args:
            ind: the index denoting the location at which the middle point should be calculated. The impact parameter,
            redshift factor, b (observer plane) and alpha (observer/BH coordinate system) will be calculated on the
            isoradial between location ind and ind + 1

        Returns:
            None: Nothing. Updates the isoradial.
        """
        mid_angle = .5 * (self.angles[ind] + self.angles[ind + 1])
        b_, P_ = self.calc_bP(mid_angle)
        z_ = redshift_factor(self.radius, mid_angle, self.t, self.M, b_)
        self.radii_b.insert(ind + 1, b_)
        self.angles.insert(ind + 1, mid_angle)
        self.impact_parameters.insert(ind + 1, P_)
        self.redshift_factors.insert(ind + 1, z_)

    def calcRedshiftLocation(self, redshift, midpoint_steps=10):
        """Calculates which location on the isoradial has some redhsift value (not redhisft factor!)"""
        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        initial_guess_indices = np.where(np.diff(np.sign(diff)))[0]

        angle_solutions = []
        b_solutions = []
        for s in range(len(initial_guess_indices)):  # generally, two solutions exists on a single isoradial
            new_ind = initial_guess_indices[s]  # initialize the initial guess.
            for _ in range(midpoint_steps):
                self.calcBetween(new_ind)  # insert more accurate solution
                diff = [redshift + 1 - z_ for z_ in self.redshift_factors[new_ind:new_ind + 3]]  # calc new interval
                start = np.where(np.diff(np.sign(diff)))[0]  # returns index where the sign changes
                new_ind += start[0]  # index of new redshift solution in refined isoradial
            # append average values of final interval
            angle_solutions.append(.5 * (self.angles[new_ind] + self.angles[new_ind + 1]))
            b_solutions.append(.5 * (self.radii_b[new_ind] + self.radii_b[new_ind + 1]))
            # update the initial guess indices, as the indexing has changed due to inserted solutions
            initial_guess_indices = [e + midpoint_steps for e in initial_guess_indices]
        return angle_solutions, b_solutions

    def plotRedshift(self):
        fig = plt.figure()
        plt.plot(ir.angles, [z - 1 for z in self.redshift_factors])
        plt.title("Redshift values for isoradial\nR={} | M = {}".format(20, M))
        plt.xlim([0, 2 * np.pi])
        plt.show()


if __name__ == '__main__':
    M = 1.
    bh = BlackHole(inclination=80, mass=M)
    # bh.writeFrames(direct_r=[6, 10, 20, 30], ghost_r=[6, 10, 20, 30], start=0, end=180, stepsize=5,
    #                y_lim=(0, 130))
    bh.plotIsoradials([30], [30], y_lim=(0, 130))
    ir = Isoradial(R=30 * M, incl=80 * np.pi / 180, mass=M, order=0)
    ir.calculate()
    ir.plotRedshift()

    bh.plotIsoRedshifts(minR=5, maxR=80, r_precision=20, midpoint_steps=5,
                        redshifts=[-.5, -.35, -.15, 0., .15, .25, .5, .75, 1.])
