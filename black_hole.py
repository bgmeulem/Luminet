import os.path
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.image as img
from collections import OrderedDict
from black_hole_math import *

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


class BlackHole:
    def __init__(self, mass=1, inclination=80, acc=10e-8):
        """Initialise black hole with mass and accretion rate
        Set viewer inclination above equatorial plane
        """
        self.t = inclination * np.pi / 180
        self.M = mass
        self.acc = acc  # accretion rate
        self.angular_properties = \
            {'start_angle': 0,
             'end_angle': np.pi,
             'angular_precision': 60,
             'mirror': True}
        self.solver_params = \
            {'initial_guesses': 20,
             'midpoint_iterations': 6,
             'plot_inbetween': False,
             'minP': 3.1 * self.M,
             'elliptic_integral_interval': (70 * np.pi / 180,
                                            2 * np.pi - 70 * np.pi / 180),
             'use_ellipse': True}
        self.plot_params = \
            {'save_plot': False,
             'plot_ellipse': False,
             'plot_core': True,
             'redshift': True,
             'linestyle': '-',
             'key': "",
             'face_color': 'black',
             'line_color': 'white',
             'text_color': 'white',
             'alpha': 1.,
             'show_grid': False,
             'legend': False,
             'orig_background': False,
             'title': "Isoradials for M = {}".format(self.M)}
        self.isoradials = {}
        self.isoredshifts = {}

    def core(self, _alpha, radius):
        if _alpha < 0:
            _alpha += 2 * np.pi
        if np.pi / 2 < _alpha <= 3 * np.pi / 2:
            return polarToCartesian([radius], [_alpha])
        else:
            r = ellipse(radius, _alpha, self.t)
            return polarToCartesian([r], [_alpha])

    def plotCore(self, _ax, c='red'):
        # plot black hole
        x = []
        y = []
        self.solver_params["minP"] = -1
        for a in np.linspace(0, 2 * np.pi, 2 * self.angular_properties["angular_precision"]):
            # TODO: how did Luminet calculate the isoradial at 2M? P is complex, how to calculate b?
            b = 2 * M if np.pi / 2 < a < 3 * np.pi / 2 else ellipse(2 * M, a, self.t)
            x_, y_ = polarToCartesian([b], [a])
            x.append(x_)
            y.append(y_)
        self.solver_params["minP"] = 3.1 * self.M
        _ax.plot(x, y, color=c, zorder=0)
        # plot critical value of b
        x_, y_ = polarToCartesian([5.2] * 2 * self.angular_properties["angular_precision"],
                                  np.linspace(-np.pi, np.pi, 2 * self.angular_properties["angular_precision"]))
        _ax.fill(x_, y_, facecolor="none", edgecolor='white', zorder=0, hatch='\\\\\\\\', alpha=.5, linewidth=.5)
        # plot black hole itself

        x_, y_ = polarToCartesian([2 * self.M] * 2 * self.angular_properties["angular_precision"],
                                  np.linspace(-np.pi, np.pi, 2 * self.angular_properties["angular_precision"]))
        _ax.fill(x_, y_, facecolor='none', zorder=0, edgecolor='white', hatch='////')
        return _ax

    def plotApparentInnerEdge(self, _ax, c='red'):
        # plot black hole
        x = []
        y = []
        self.solver_params["minP"] = -1
        for a in np.linspace(0, 2 * np.pi, 2 * self.angular_properties["angular_precision"]):
            b = 3 * np.sqrt(3) * self.M if np.pi / 2 < a < 3 * np.pi / 2 else \
                min(ellipse(3 * np.sqrt(3) * self.M, a, self.t), 3 * np.sqrt(3) * self.M)
            x_, y_ = polarToCartesian([b], [a])
            x.append(x_)
            y.append(y_)
        self.solver_params["minP"] = 3.1 * self.M
        _ax.plot(x, y, color=c, zorder=0, linestyle='--')
        return _ax

    def __getFigure(self):
        _fig = plt.figure(figsize=(10, 10))
        _ax = _fig.add_subplot(111)
        plt.axis('off')  # command for hiding the axis.
        _fig.patch.set_facecolor(self.plot_params['face_color'])
        _ax.set_facecolor(self.plot_params['face_color'])
        if self.plot_params['show_grid']:
            _ax.grid(color='grey')
            _ax.tick_params(which='both', labelcolor=self.plot_params['text_color'],
                            labelsize=15)
        else:
            _ax.grid()
        return _fig, _ax

    def setInclination(self, incl):
        self.t = incl * np.pi / 180

    def calcIsoRedshift(self, _redshift, _dirty_isoradials, cartesian=False):
        """Calculates the isoredshift for a single redshift value"""
        solutions = OrderedDict()
        _maxR = None
        t = tqdm(_dirty_isoradials, position=1, leave=False)
        for ir in _dirty_isoradials:
            t.set_description("Calculating isoredshift {} at R={:.2f}".format(_redshift, ir.radius))
            t.update()
            # Use the same solver params from the black hole to calculate the redshift location on the isoradial
            a, r = ir.calcRedshiftLocation(_redshift, self.solver_params['midpoint_iterations'],
                                           cartesian=cartesian)
            solutions[ir.radius] = [a, r]
        # initialise an Isoredshift with the coordinates calculated above instance and return
        return Isoredshift(inclination=self.t, redshift=_redshift, bh_mass=self.M, ir_solver_params=self.solver_params,
                           coordinates=solutions)

    def calcIsoRedshifts(self, minR=6, maxR=60, r_precision=10, redshifts=None):
        if redshifts is None:
            redshifts = [-.15, -.1, -.05, 0., .05, .1, .15, .20, .25, .5]

        def getDirtyIsoradials(_minR, _maxR, _r_precision, angular_precision=10):
            # an array of quick and dirty isoradials for the initial guesses of redshifts
            isoradials = []  # for initial guesses
            for radius in np.linspace(_minR, _maxR, _r_precision):  # calculate the initial guesses
                isoradial = Isoradial(radius, self.t, self.M,
                                      angular_properties={'start_angle': 0,
                                                          'end_angle': np.pi,
                                                          'angular_precision': angular_precision,
                                                          'mirror': True})

                isoradials.append(isoradial)
            return isoradials

        dirty_isoradials = getDirtyIsoradials(minR, maxR, r_precision)
        isoredshifts = []
        t = tqdm(redshifts, desc="Calculating redshift", position=0)
        for redshift in t:
            t.set_description("Calculating redshift {}".format(redshift))
            dirty_isoradials_copy = dirty_isoradials  # to mutate while finding the isoredshift
            iz = self.calcIsoRedshift(redshift, dirty_isoradials_copy, cartesian=False)
            isoredshifts.append(iz)
        self.isoredshifts = isoredshifts
        return isoredshifts

    def plotIsoradials(self, direct_r: [], ghost_r: [], ax_lim=None, show=False):
        """Given an array of radii for the direct image and/or ghost image, plots the corresponding
        isoradials.
        Calculates the isoradials according to self.root_params
        Plots the isoradials according to self.plot_params"""

        def plotEllipse(__r, __ax, incl):
            ax_ = __ax
            a = np.linspace(-np.pi, np.pi, 2 * self.angular_properties['angular_precision'])
            ell = [ellipse(__r, a_, incl) for a_ in a]
            x, y = polarToCartesian(ell, a)
            ax_.plot(x, y, color='red', zorder=-1)
            return ax_

        _fig, _ax = self.__getFigure()
        color_range = (-1, 1)
        progress_bar = tqdm(range(len(direct_r) + len(ghost_r)), position=0, leave=False)

        # plot background
        if self.plot_params['orig_background']:
            image = img.imread('bh_background.png')
            scale = (940 / 30 * 2. * M)  # 940 px by 940 px, and 2M ~ 30px
            _ax.imshow(image, extent=(-scale / 2, scale / 2, -scale / 2, scale / 2))
        else:
            _ax.set_facecolor('black')

        # plot ghost images
        progress_bar.set_description("Ghost images")
        self.plot_params['alpha'] = .5
        for radius in sorted(ghost_r):
            progress_bar.update(1)
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=1,
                                  _solver_params=self.solver_params, plot_params=self.plot_params,
                                  angular_properties=self.angular_properties)
            plt_, _ax = isoradial.plot(_ax, self.plot_params, colornorm=color_range)

        # plot direct images
        progress_bar.set_description("Direct images")
        self.plot_params['alpha'] = 1.
        for radius in sorted(direct_r):
            progress_bar.update(1)
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=0,
                                  _solver_params=self.solver_params, plot_params=self.plot_params,
                                  angular_properties=self.angular_properties)
            plt_, _ax = isoradial.plot(_ax, self.plot_params, colornorm=color_range)

        if self.plot_params['plot_ellipse']:  # plot ellipse
            progress_bar.write("Plotting ellipses")
            for radius in direct_r:
                _ax = plotEllipse(radius, _ax, self.t)

        if self.plot_params['plot_core']:
            progress_bar.write("Plotting center of black hole")
            _ax = self.plotApparentInnerEdge(_ax, 'red')

        if ax_lim:
            _ax.set_ylim(ax_lim)
            _ax.set_xlim(ax_lim)
        plt.title(self.plot_params['title'], color=self.plot_params['text_color'])
        if show:
            plt.show()
        if self.plot_params['save_plot']:
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            _fig.savefig(name, dpi=300, facecolor=self.plot_params['face_color'])
        return _fig, _ax

    def writeFrames(self, direct_r=None, ghost_r=None, start=0, end=180, step_size=5,
                    ax_lim=(-35, 35)):
        if ghost_r is None:
            ghost_r = [6, 10, 20, 30]
        if direct_r is None:
            direct_r = [6, 10, 20, 30]
        steps = np.linspace(start, end, 1 + (end - start) // step_size)
        for a in tqdm(steps, position=0, desc='Writing frames'):
            self.setInclination(a)
            bh.plot_params['title'] = 'inclination = {:03}°'.format(int(a))
            fig_, ax_ = bh.plotIsoradials(direct_r, ghost_r, ax_lim=ax_lim)
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            fig_.savefig('movie/' + name, dpi=300, facecolor=self.plot_params['face_color'])
            plt.close()  # to not destroy your RAM

    def plotIsoRedshifts(self, minR=6, maxR=40, r_precision=15, redshifts=None):
        if redshifts is None:
            redshifts = [-.5, -.35, -.15, 0., .15, .25, .5, .75, 1.]
        isoredshifts = bh.calcIsoRedshifts(minR, maxR, r_precision=r_precision, redshifts=redshifts)
        _fig, _ax = self.__getFigure()  # make new figure
        color_map = plt.get_cmap('RdBu_r')
        norm = mpl.colors.Normalize(-1, 1)

        for irz in isoredshifts:
            color = cm.ScalarMappable(norm=norm, cmap=color_map).to_rgba(irz.redshift)
            plt.plot(irz.y, [-e for e in irz.x], color=color)  # TODO: hack to correctly orient plot
            irz.improveTipOnce()
            plt.plot(irz.y, [-e for e in irz.x], color=color)  # TODO: hack to correctly orient plot
            tries = 0
            while len(irz.ir_radii_w_co) < 10 and tries < 10:  # TODO: make this an Isoredshift method
                irz.improveBetweenSolutionsOnce()
                tries += 1

            plt.plot(irz.y, [-e for e in irz.x], color=color)  # TODO: hack to correctly orient plot
        # Shrink current axis by 20%
        # mx = 1.1 * max([max(isoredshift.redshift) for isoredshift in isoredshifts])
        # ax.set_xlim((-mx, mx))
        # ax.set_ylim((-mx, mx))
        plt.suptitle("Isoredshift lines for M={}".format(self.M))
        plt.show()

    def samplePoints(self, minR=None, maxR=None, N=1000, f='points.csv', f2='points_secondary.csv'):
        # TODO save time here by calculating when secondary image will be visible given some inclination
        df = pd.read_csv(f, index_col=0) if os.path.exists('./{}'.format(f)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'z_factor', 'flux_o'])
        df2 = pd.read_csv(f, index_col=0) if os.path.exists('./{}'.format(f2)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'z_factor', 'flux_o'])

        minR_ = minR if minR else self.M * 3.01
        maxR_ = maxR if maxR else self.M * 60
        t = tqdm(range(N))
        for _ in t:
            t.update(1)
            r = minR_ + maxR_ * np.sqrt(np.random.random())  # uniformly sampling a circle's surface
            theta = np.random.random() * 2 * np.pi
            b_ = calcImpactParameter(r, incl=self.t, _alpha=theta, _M=self.M, **self.solver_params)
            b_2 = calcImpactParameter(r, incl=self.t, _alpha=theta, _M=self.M, **self.solver_params, n=1)
            if b_ is not None:
                x = b_ * np.cos(theta)
                y = b_ * np.sin(theta)
                redshift_factor_ = redshift_factor(r, theta, self.t, self.M, b_)
                F_o = flux_observed(r, self.acc, self.M, redshift_factor_)
                df = df.append(pd.DataFrame({'X': x, 'Y': y, 'impact_parameter': b_, 'z_factor': redshift_factor_, 'flux_o': F_o}, index=[0]))
            if b_2 is not None:
                theta += np.pi  # TODO: fix dirty manual flip for ghost image
                x = b_2 * np.cos(theta)
                y = b_2 * np.sin(theta)
                redshift_factor_2 = redshift_factor(r, theta, self.t, self.M, b_2)
                F_o2 = flux_observed(r, self.acc, self.M, redshift_factor_2)
                df2 = df2.append(pd.DataFrame({'X': x, 'Y': y, 'impact_parameter': b_2, 'z_factor': redshift_factor_2, 'flux_o': F_o2}, index=[0]))
        df.to_csv(f)
        df2.to_csv(f2)

    def plotPoints(self, f='points.csv', f2='points_secondary.csv', powerscale=.7):
        """
        Plot the points written out by samplePoints()
        :param f: filename of points (direct image)
        :param f2: filename of points (secondary image)
        :param powerscale: powerscale to apply to flux. No powerscale = 1. Anything lower than 1 will make the
        dim points pop out more.
        :return:
        """
        # _fig, _ax = self.plotIsoradials([6, 20], [6, 20])  # for testing
        _fig, _ax = self.__getFigure()
        points1 = pd.read_csv(f)
        points2 = pd.read_csv(f2)
        points2 = points2[points2['X'] < 0]  # dirty hack to fix overlapping images where there shouldn't be
        # TODO: how to properly show ghost ring?
        max_flux = max(max(points1['flux_o']), max(points2['flux_o']))
        min_flux = 0
        for i, points in enumerate([points1, points2]):
            points = points[(points['flux_o'] > 0)]
            fluxes = [(abs(fl + min_flux) / (max_flux + min_flux))**powerscale for fl in points['flux_o']]
            color = [cm.ScalarMappable(cmap="Greys_r", norm=plt.Normalize(0, 1)).to_rgba(flux) for flux in fluxes]
            _l = plt.scatter(points['X'], points['Y'], alpha=1, color=color, zorder=i)
            _l.set_sizes([7] * len(points))
        _ax.set_xlim((-40, 40))
        _ax.set_ylim((-40, 40))

        plt.savefig('SampledPoints_incl={}.png'.format(self.t), dpi=300)
        plt.show()


class Isoradial:
    def __init__(self, R, incl, mass, order=0, _solver_params=None, plot_params=None, angular_properties=None):
        self.M = mass  # mass of the black hole containing this isoradial
        self.t = incl  # inclination of observer's plane
        self.radius = R
        self.order = order
        self.angular_properties = angular_properties if angular_properties else \
            {'start_angle': 0.,
             'end_angle': np.pi,
             'angular_precision': 60,
             'mirror': True}  # default values
        self.solver_params = _solver_params if _solver_params else \
            {'initial_guesses': 20,
             'midpoint_iterations': 5,
             'plot_inbetween': False,
             'minP': 3.1 * self.M}  # default values
        self.plot_params = plot_params if plot_params else \
            {'save_plot': True,
             'plot_ellipse': False,
             'redshift': False,
             'linestyle': '-',
             'key': "",
             'face_color': 'black',
             'line_color': 'white',
             'text_color': 'white',
             'alpha': 1.,
             'show_grid': False,
             'orig_background': False,
             'legend': False,
             'title': "Isoradials for R = {}".format(R)}  # default values
        self.radii_b = []
        self.angles = []
        self.X = []
        self.Y = []
        self.redshift_factors = []

        self.calculate()

    def calculateCoordinates(self, _tqdm=False):
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
        t = np.linspace(start_angle, end_angle, angular_precision)
        if _tqdm:
            t = tqdm(t, desc='Calculating isoradial R = {}'.format(self.radius), position=2, leave=False)
        for alpha_ in t:
            b_ = calcImpactParameter(self.radius, self.t, alpha_, self.M, n=self.order, **self.solver_params)
            if b_:
                angles.append(alpha_)
                impact_parameters.append(b_)
        if self.order > 0:  # TODO: fix dirty manual flip for ghost images
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
        self.X, self.Y = polarToCartesian(self.radii_b, self.angles)
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

    def plot(self, _ax=None, plot_params=None, show=False, colornorm=(0, 1)):
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
                __ax, __x, __y, z=None, cmap=plt.get_cmap('RdBu_r'), norm=plt.Normalize(*colornorm),
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
                z = np.linspace(0.0, 1.0, len(__x))

            # Special case if a single number:
            if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
                z = np.array([z])

            z = np.asarray(z)

            segments = make_segments(__x, __y)
            lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                                      linewidth=linewidth, alpha=self.plot_params['alpha'])
            lc.set_array(z)
            __ax.add_collection(lc)
            # mx = max(segments[:][:, 1].flatten())
            # _ax.set_ylim((0, mx))
            return __ax

        if not _ax:
            ir_fig = plt.figure(figsize=(5, 5))
            ir_ax = ir_fig.add_subplot()
        else:
            ir_ax = _ax

        if not plot_params:
            plot_params = self.plot_params

        # Plot isoradial
        if self.plot_params['redshift']:
            ir_ax = colorline(ir_ax, self.X, self.Y, z=[e - 1 for e in self.redshift_factors],
                              cmap=cm.get_cmap('RdBu_r'))  # red-blue colormap reversed to match redshift
        else:
            ir_ax.plot(self.X, self.Y, color=plot_params['line_color'],
                       alpha=plot_params['alpha'], linestyle=self.plot_params['linestyle'])
        if self.plot_params['legend']:
            plt.legend(prop={'size': 16})
        if len(self.X) and len(self.Y):
            mx = np.max([np.max(self.X), np.max(self.Y)])
            mx *= 1.1
            ir_ax.set_xlim([-mx, mx])
            ir_ax.set_ylim([-mx, mx])
        if show:
            # ax.autoscale_view(scalex=False)
            # ax.set_ylim([0, ax.get_ylim()[1] * 1.1])
            plt.show()
        return plt, ir_ax

    def calcBetween(self, ind):
        """
        Calculates the impact parameter and redshift factor at the
        isoradial location between place ind and ind + 1

        Args:
            ind: the index denoting the location at which the middle point should be calculated. The impact parameter,
            redshift factor, b (observer plane) and alpha (observer/BH coordinate system) will be calculated on the
            isoradial between location ind and ind + 1

        Returns:
            None: Nothing. Updates the isoradial.
        """
        mid_angle = .5 * (self.angles[ind] + self.angles[ind + 1])
        b_ = calcImpactParameter(self.radius, self.t, mid_angle, self.M, **self.solver_params)
        z_ = redshift_factor(self.radius, mid_angle, self.t, self.M, b_)
        self.radii_b.insert(ind + 1, b_)
        self.angles.insert(ind + 1, mid_angle)
        self.redshift_factors.insert(ind + 1, z_)

    def calcRedshiftLocation(self, redshift, midpoint_steps=10, cartesian=False):
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
        if cartesian:
            return polarToCartesian(b_solutions, angle_solutions)
        return angle_solutions, b_solutions

    def plotRedshift(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.angles, [z - 1 for z in self.redshift_factors])
        plt.title("Redshift values for isoradial\nR={} | M = {}".format(20, M))
        ax.set_xlim([0, 2 * np.pi])
        plt.show()


class Isoredshift:
    # TODO: isoredshift should ideally be initialised from the coordinates closest to the black hole.
    def __init__(self, inclination, redshift, bh_mass, ir_solver_params=None, coordinates: {} = None):
        self.coordinates = coordinates if coordinates is not None else {}
        self.t = inclination
        self.M = bh_mass
        self.t = inclination
        self.redshift = redshift
        self.co = self.angles, self.radii = self.__extractCo(coordinates)
        self.isoradial_solver_params = ir_solver_params if ir_solver_params else \
            {'initial_guesses': 20,
             'midpoint_iterations': 5,
             'plot_inbetween': False,
             'minP': 3.1 * self.M}
        self.x, self.y = polarToCartesian(self.radii, self.angles, rotation=0)
        self.orderCoordinates()

    def __extractCo(self, solutions):
        a = []
        r = []
        for key, val in solutions.items():
            if len(val[0]) > 0:  # at least one solution was found
                angles, radii = val
                [a.append(angle) for angle in angles]
                [r.append(radius) for radius in radii]
        self.co = self.angles, self.radii = a, r
        return a, r

    def splitCoOnSolutions(self):
        keys_w_s = []
        keys_wo_s = []
        for key in self.coordinates:
            if len(self.coordinates[key][0]) == 0:
                keys_wo_s.append(key)
            else:
                keys_w_s.append(key)
        dict_w_s = {key: self.coordinates[key] for key in keys_w_s}
        dict_wo_s = {key: self.coordinates[key] for key in keys_wo_s}
        return dict_w_s, dict_wo_s

    def calcCoreCoordinates(self):
        """Calculates the coordinates of the redshift on the closest possible isoradial: 6*M (2*R_s)"""
        ir = Isoradial(6 * self.M, self.t, self.M, order=0, **self.isoradial_solver_params)
        co = ir.calcRedshiftLocation(self.redshift)
        return co

    def __calcCoordinates(self, dirty_isoradials: [Isoradial], midpoint_steps):
        """Calculates the isoredshift for a single redshift value"""
        _a = []
        _radius = []
        t = tqdm(dirty_isoradials, position=1, leave=False)
        for ir in dirty_isoradials:
            t.set_description("Calculating isoredshift {} at R={:.2f}".format(self.redshift, ir.radius))
            t.update()
            a, radius = ir.calcRedshiftLocation(self.redshift, midpoint_steps, cartesian=False)
            _a.append(a)
            _radius.append(radius)
        self.a = _a
        self.radius = _radius
        return _a, _radius

    def orderCoordinates(self, plot_inbetween=False):
        angles, radii = self.co
        x, y = polarToCartesian(radii, angles)
        cart = [[x_, y_] for x_, y_ in zip(x, y)]
        cx, cy = np.mean(x, axis=0), np.mean(y, axis=0)

        s = sorted(cart, key=lambda p: getAngleAround([cx, cy], p))
        x, y = np.asarray(s).T

        if plot_inbetween:
            # use this to get a visual overview of what happens when ordering the isoradial points using
            # getAngleAround()
            fig, ax = plt.subplots()
            for i, p in enumerate(s):
                plt.plot(*p, 'bo')
                plt.text(x[i] * (1 + 0.01), y[i] * (1 + 0.01), i, fontsize=12)
            plt.scatter(cx, cy)
            plt.plot([0, cx], [0, cy])
            plt.show()

        self.x, self.y = x, y
        self.co = np.asarray([cartesianToPolar(x_, y_) for x_, y_ in zip(x, y)]).T
        self.angles, self.radii = [list(e) for e in self.co]  # TODO: clean this

        return s

    def calcBetween(self, radius, begin_angle=0, end_angle=np.pi, angular_precision=3, mirror=False):
        ir = Isoradial(R=radius, incl=self.t, mass=self.M,
                       angular_properties={'start_angle': begin_angle,
                                           'end_angle': end_angle,
                                           'angular_precision': angular_precision,
                                           'mirror': mirror})
        a, r = ir.calcRedshiftLocation(self.redshift, self.isoradial_solver_params['midpoint_iterations'],
                                       cartesian=False)

        if len(a):
            for angle, radius in zip(a, r):
                self.angles.append(angle)
                self.radii.append(radius)
        self.co = self.angles, self.radii
        self.orderCoordinates(plot_inbetween=True)  # TODO
        self.x, self.y = polarToCartesian(self.radii, self.angles, rotation=0)
        return a, r

    def improveBetweenSolutionsOnce(self, angular_precision_dirty_ir=10):
        """
        Calculates the redshift on the isoredshift line between the already known redshifts
        Does so by calculating the entire isoradial (with low precision) inbetween the radii corresponding
        to these redshift solutions and calculating the redshifts on this isoradial
        # TODO: accuracy can be improved by using previous solutions as initial guesses
        # TODO: This implementation probably needs a way of checking if the line is connected or not
        # TODO: to properly assess the possible co interval
        # TODO: right now: the entire isoradial is calculated, which is no better than BlackHole.calculateIsoredshifts()
        # TODO: but with higher R precision

        # TODO: split in improve entire redshift and improve tip only
        :param angular_precision_dirty_ir: the angular precision at which to calculate the dirty isoradials
        :return: 0 if success
        """
        ir_radii_w_co = sorted(self.ir_radii_w_co)
        if self.maxR is not None:
            ir_radii_w_co.append(self.maxR)  # no co found at this R, maybe inbetween maxR and previous R?

        # assert self.maxR is not None, "Max radius not defined or calculated. Aborting."
        for b, e in zip(ir_radii_w_co[:-1], ir_radii_w_co[1:]):  # isoradial intervals
            r_inbtw = .5*(b+e)
            # TODO: calc isoradial inbetween.
            ir = Isoradial(R=r_inbtw, incl=self.t, mass=self.M,
                           angular_properties={'start_angle': 0,
                                               'end_angle': np.pi,
                                               'angular_precision': angular_precision_dirty_ir,
                                               'mirror': True})
            a, r = ir.calcRedshiftLocation(self.redshift, self.isoradial_solver_params['midpoint_iterations'],
                                           cartesian=False)

            if len(a) == 0:
                if self.maxR is None or ir.radius < self.maxR:
                    # no coordinates found for this isoradial.
                    self.maxR = r_inbtw
            else:
                self.ir_radii_w_co.append(r_inbtw)
                for angle, radius in zip(a, r):
                    self.angles.append(angle)
                    self.radii.append(radius)
        self.co = self.angles, self.radii
        self.orderCoordinates(plot_inbetween=False)
        self.x, self.y = polarToCartesian(self.radii, self.angles, rotation=0)
        self.ir_radii_w_co = sorted(self.ir_radii_w_co)
        return self

    def improveTipOnce(self):
        """
        # TODO: this works! finds one more isoradial that previously did not have solution.
        # TODO: The other IR simply have no redshift -> fix with improveBetweenSOlutionsOnce()

        :param angular_precision_dirty_ir: the angular precision at which to calculate the dirty isoradials
        :return: 0 if success
        """

        r_w_s, r_wo_s = self.splitCoOnSolutions()
        angle_interval, last_radii = self.coordinates[max(r_w_s.keys())]
        for ir_radius_wo_co in r_wo_s:  # isoradials who did not contain this redshift
            # calculate again, but this time ensure there is a point closer to the tip, and it's not cut off
            # due to low precision
            begin_angle, end_angle = angle_interval
            if end_angle - begin_angle > np.pi:  # in case the angle is around 0 and 2pi
                begin_angle, end_angle = end_angle, begin_angle  # this works, apparently
            a, r = self.calcBetween(ir_radius_wo_co, begin_angle, end_angle, angular_precision=7, mirror=True)
            if len(a) > 0:
                self.coordinates[ir_radius_wo_co] = [a, r]  # update this radius with new coordinates
        self.co = self.angles, self.radii = self.__extractCo(self.coordinates)  # update coordinates

def polarToCartesian(radii, angles, rotation=0):
    x = []
    y = []
    for R, th in zip(radii, angles):
        x.append(R * np.cos(th + rotation))
        y.append(R * np.sin(th + rotation))
    return x, y


def cartesianToPolar(x, y):
    R = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    return th, R


def getAngleAround(p1, p2):
    """
    Calculates the angle of p2 around p1

    :param m_point: the middle of the two redshift points on the innermost isoradial.
    :param split_zero_index: if p[split_zero_index] < 0, return 2pi - angle
    :param p1: coordinate 1 in format [x, y]
    :param p2:  coordinate 2 in format [x, y]
    :return: angle in radians
    """
    cx, cy = p1

    p2_ = np.subtract(p2, p1)
    angle_center, _ = cartesianToPolar(cx, cy)
    # rotate p2_ counter clockwise until the vector to the isoradial center is aligned with negative y-axis
    theta = np.pi - angle_center
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    p2_ = np.dot(rot, p2_)
    angle_target, _ = cartesianToPolar(p2[0], p2[1])
    angle_target_around_center, R_target_around_center = cartesianToPolar(p2_[0], p2_[1])
    if angle_target_around_center < 0:
        angle_target_around_center = 2 * np.pi + angle_target_around_center
    if angle_target_around_center > np.pi and p2[1] < 0 and cx > 0:
        # dirty check to see if angle is in between the angle defined by origin - isoradial center - p2
        # cause then the angle will be too big
        angle_target_around_center -= 2 * np.pi

    return angle_target_around_center


if __name__ == '__main__':
    M = 1.
    bh = BlackHole(inclination=80, mass=M)
    # bh.writeFrames(direct_r=[6, 10, 20, 30], ghost_r=[6, 10, 20, 30], start=90, end=180, stepsize=5, ax_lim=(-35, 35))
    bh.solver_params = {'initial_guesses': 12,
                        'midpoint_iterations': 11,  # 5 is fine for direct image. Up until 10 for ghost image of large R
                        'plot_inbetween': False,
                        'minP': 3.01,
                        'elliptic_integral_interval': (0, 2 * np.pi),  # which section should be calculated with an elliptic integral
                        'use_ellipse': True}
    bh.plot_params['plot_core'] = False
    bh.angular_properties['angular_precision'] = 100
    # fig, ax = bh.plotIsoradials([6, 10, 20, 30], [6, 10, 30, 1000], ax_lim=(-35, 35))
    bh.samplePoints(N=200, minR=20, maxR=40)  # maxR of 40 suffices for direct image, but needs up until 60 to hide ghost image because of lazy programming to properly deal with this
    bh.plotPoints()
    # fig, ax = bh.plotIsoRedshifts(redshifts=[-.2])
    # bh.angular_properties['start_angle'] = np.pi/2
    # fig, ax = bh.plotIsoradials([6, 10, 20, 30], [], show=False)
    # x, y = [2 * np.cos(th) for th in np.linspace(0, 2 * np.pi, 100)], \
    #        [2 * np.sin(th) for th in np.linspace(0, 2 * np.pi, 100)]
    # ax.plot(x, y, color='red')
    # plt.show()

    # bh.plotIsoRedshifts(minR=3.*M, maxR=60*M, r_precision=10, midpoint_steps=5,
    #                     redshifts=[-.15, -.1, -.05, 0., 0.05, 0.1, .15, .25, .5, .75])
