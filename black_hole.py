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
            {'plot_isoredshifts_inbetween': False,
             'save_plot': False,
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
            return polar_to_cartesian_lists([radius], [_alpha])
        else:
            r = ellipse(radius, _alpha, self.t)
            return polar_to_cartesian_lists([r], [_alpha])

    def plot_core(self, _ax, c='red'):
        # plot black hole
        x = []
        y = []
        self.solver_params["minP"] = -1
        for a in np.linspace(0, 2 * np.pi, 2 * self.angular_properties["angular_precision"]):
            # TODO: how did Luminet calculate the isoradial at 2M? P is complex, how to calculate b?
            b = 2 * M if np.pi / 2 < a < 3 * np.pi / 2 else ellipse(2 * M, a, self.t)
            x_, y_ = polar_to_cartesian_lists([b], [a])
            x.append(x_)
            y.append(y_)
        self.solver_params["minP"] = 3.1 * self.M
        _ax.plot(x, y, color=c, zorder=0)
        # plot critical value of b
        x_, y_ = polar_to_cartesian_lists([5.2] * 2 * self.angular_properties["angular_precision"],
                                          np.linspace(-np.pi, np.pi, 2 * self.angular_properties["angular_precision"]))
        _ax.fill(x_, y_, facecolor="none", edgecolor='white', zorder=0, hatch='\\\\\\\\', alpha=.5, linewidth=.5)
        # plot black hole itself

        x_, y_ = polar_to_cartesian_lists([2 * self.M] * 2 * self.angular_properties["angular_precision"],
                                          np.linspace(-np.pi, np.pi, 2 * self.angular_properties["angular_precision"]))
        _ax.fill(x_, y_, facecolor='none', zorder=0, edgecolor='white', hatch='////')
        return _ax

    def get_apparent_inner_edge(self, a):
        return 3 * np.sqrt(3) * self.M if np.pi / 2 < a < 3 * np.pi / 2 else \
            min(ellipse(3 * np.sqrt(3) * self.M, a, self.t), 3 * np.sqrt(3) * self.M)  # imact parameter

    def plot_apparent_inner_edge(self, _ax, c='red'):
        # plot black hole
        x = []
        y = []
        self.solver_params["minP"] = -1
        for a in np.linspace(0, 2 * np.pi, 2 * self.angular_properties["angular_precision"]):
            b = self.get_apparent_inner_edge(a)
            rot = -np.pi / 2 if self.t < np.pi / 2 else np.pi / 2
            x_, y_ = polar_to_cartesian_lists([b], [a], rotation=rot)
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
            a, r = ir.calc_redshift_location(_redshift, self.solver_params['midpoint_iterations'],
                                             cartesian=cartesian)
            solutions[ir.radius] = [a, r]
        # initialise an Isoredshift with the coordinates calculated above instance and return
        return Isoredshift(inclination=self.t, redshift=_redshift, bh_mass=self.M, ir_solver_params=self.solver_params,
                           coordinates=solutions)

    def calcIsoRedshifts(self, minR=6, maxR=60, r_precision=10, redshifts=None):
        if redshifts is None:
            redshifts = [-.15, 0., .1, .20, .5]

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
            iz.improve(plot_inbetween=self.plot_params['plot_isoredshifts_inbetween'])
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
            x, y = polar_to_cartesian_lists(ell, a)
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
            _ax = self.plot_apparent_inner_edge(_ax, 'red')

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

        # Shrink current axis by 20%
        # mx = 1.1 * max([max(isoredshift.redshift) for isoredshift in isoredshifts])
        # ax.set_xlim((-mx, mx))
        # ax.set_ylim((-mx, mx))
        plt.suptitle("Isoredshift lines for M={}".format(self.M))
        plt.show()

    def samplePoints(self, minR=None, maxR=None, N=1000, f='points.csv', f2='points_secondary.csv'):
        """
        Samples points on the accretion disk. This sampling is not done uniformly, but a bias is added towards the
        center of the accretion disk, as the observed flux is exponentially bigger here and this needs the most
        precision.
        Both the direct and ghost image for each point is calculated. It's coordinates (polar and cartesian),
        redshift and
        :param minR:
        :param maxR:
        :param N:
        :param f:
        :param f2:
        :return:
        """
        df = pd.read_csv(f, index_col=0) if os.path.exists('./{}'.format(f)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'angle', 'z_factor', 'flux_o'])
        df2 = pd.read_csv(f, index_col=0) if os.path.exists('./{}'.format(f2)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'angle', 'z_factor', 'flux_o'])

        minR_ = minR if minR else self.M * 3.01
        maxR_ = maxR if maxR else self.M * 60
        t = tqdm(range(N), desc="Sampling points for direct and ghost image")
        for _ in t:
            t.update(1)
            # r = minR_ + maxR_ * np.sqrt(np.random.random())  # uniformly sampling a circle's surface
            r = minR_ + maxR_ * np.random.random()  # bias towards center (where the interesting stuff is)
            theta = np.random.random() * 2 * np.pi
            b_ = calc_impact_parameter(r, incl=self.t, _alpha=theta, bh_mass=self.M, **self.solver_params)
            b_2 = calc_impact_parameter(r, incl=self.t, _alpha=theta, bh_mass=self.M, **self.solver_params, n=1)
            if b_ is not None:
                x, y = polar_to_cartesian_lists([b_], [theta], rotation=-np.pi / 2)
                redshift_factor_ = redshift_factor(r, theta, self.t, self.M, b_)
                F_o = flux_observed(r, self.acc, self.M, redshift_factor_)
                df = pd.concat([df,
                                pd.DataFrame.from_dict({'X': x, 'Y': y, 'impact_parameter': b_,
                                                        'angle': (theta + 3 * np.pi / 2) % (2 * np.pi),
                                                        'z_factor': redshift_factor_, 'flux_o': F_o})])
            if b_2 is not None:
                theta = (theta + np.pi) % (2 * np.pi)  # TODO: fix dirty manual flip for ghost image
                x, y = polar_to_cartesian_lists([b_2], [theta], rotation=-np.pi / 2)
                redshift_factor_2 = redshift_factor(r, theta, self.t, self.M, b_2)
                F_o2 = flux_observed(r, self.acc, self.M, redshift_factor_2)
                df2 = pd.concat([df2,
                                 pd.DataFrame.from_dict({'X': x, 'Y': y, 'impact_parameter': b_2,
                                                         'angle': (theta + 3 * np.pi / 2) % (2 * np.pi),
                                                         'z_factor': redshift_factor_2, 'flux_o': F_o2})
                                 ])
        df.to_csv(f)
        df2.to_csv(f2)

    def plotPoints(self, f='points.csv', f2='points_secondary.csv', powerscale=.7, levels=100):
        """
        Plot the points written out by samplePoints()
        :param f: filename of points (direct image)
        :param f2: filename of points (secondary image)
        :param powerscale: powerscale to apply to flux. No powerscale = 1. Anything lower than 1 will make the
        dim points pop out more.
        :return:
        """

        def plotDirectImage(_ax, points, _min_flux, _max_flux, _powerscale):
            # direct image
            fluxes = [(abs(fl + _min_flux) / (_max_flux + _min_flux)) ** _powerscale for fl in points['flux_o']]
            _ax.tricontourf(points['X'], points['Y'], fluxes, cmap='Greys_r', norm=plt.Normalize(0, 1),
                            levels=levels,
                            nchunk=2)
            br = blackRing(self)
            _ax.fill_between(br.X, br.Y, color='black')  # to fill Delauney triangulation artefacts with black
            return _ax

        def plotGhostImage(_ax, points, _levels, _min_flux, _max_flux, _powerscale):
            # ghost image
            cross_angle = np.pi / 40  # about where the ghost image dips under the accretion disk
            points = points.loc[(points['angle'] < np.pi + cross_angle) |
                                (points['angle'] > 2 * np.pi - cross_angle)]
            points.sort_values(by=['flux_o'], ascending=False)
            N_chunks = _levels // 20
            for level in range(N_chunks):
                points_chunk = points[level * len(points) // N_chunks: (level + 1) * len(points) // N_chunks]
                fluxes = [(abs(fl + _min_flux) / (_max_flux + _min_flux)) ** _powerscale for fl in
                          points_chunk['flux_o']]
                color = [cm.ScalarMappable(cmap="Greys_r", norm=plt.Normalize(0, 1)).to_rgba(flux)
                         for flux in fluxes]
                # make sure brightest points are on top
                _ax.scatter(points_chunk['X'], points_chunk['Y'], color=color, zorder=_levels - level + 1,
                            s=.5)
            return _ax

        def blackRing(_bh):
            ir = Isoradial(radius=6 * self.M, incl=self.t, order=0, _solver_params=_bh.solver_params, bh_mass=_bh.M)
            ir.radii_b = [.99 * b for b in ir.radii_b]
            ir.X, ir.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)
            return ir

        _fig, _ax = self.__getFigure()
        points1 = pd.read_csv(f)
        # points1 = addBlackRing(self, points1)
        points2 = pd.read_csv(f2)
        max_flux = max(max(points1['flux_o']), max(points2['flux_o']))
        min_flux = 0

        _ax = plotDirectImage(_ax, points1, min_flux, max_flux, powerscale)
        _ax = plotGhostImage(_ax, points2, levels, min_flux, max_flux, powerscale)

        _ax.set_xlim((-40, 40))
        _ax.set_ylim((-40, 40))

        plt.savefig('SampledPoints_incl={}.png'.format(self.t), dpi=300, facecolor='black')
        plt.show()

    def plot_isoredshifts_from_points(self, f='points.csv', levels=500):
        def blackRing(_bh):
            ir = Isoradial(radius=6 * self.M, incl=self.t, order=0, _solver_params=_bh.solver_params, bh_mass=_bh.M)
            ir.radii_b = [.99 * b for b in ir.radii_b]
            ir.X, ir.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)
            return ir
        _fig, _ax = self.__getFigure()
        points = pd.read_csv(f)
        br = blackRing(self)
        color_map = plt.get_cmap('RdBu_r')
        norm = mpl.colors.Normalize(-1, 1)

        # points1 = addBlackRing(self, points1)
        mn, mx = min(points['X']), max(points['X'])
        levels_ = [-.2, -.15, -.1, -0.05, 0., .05, .1, .15, .2, .25, .5, .75]
        _ax.tricontour(points['X'], points['Y'], [e for e in points['z_factor']], cmap=color_map,
                       norm=plt.Normalize(0, 2),
                       levels=[e + 1 for e in levels_],
                       nchunk=2,
                       linewidths=1)
        _ax.fill_between(br.X, br.Y, color='black', zorder=2)
        _ax.set_xlim((mn, mx))
        _ax.set_ylim((mn, mx))
        plt.show()
        _fig.savefig(f"Plots/Isoredshifts_incl={180*self.t/np.pi}.svg", facecolor='black', dpi=300)


class Isoradial:
    def __init__(self, radius, incl, bh_mass, order=0, _solver_params=None, plot_params=None, angular_properties=None):
        self.M = bh_mass  # mass of the black hole containing this isoradial
        self.t = incl  # inclination of observer's plane
        self.radius = radius
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
             'min_periastron': 3.1 * self.M}  # default values
        self.find_redshift_params = {
            'force_redshift_solution': False,  # force finding a redshift solution on the isoradial
            'max_force_iter': 5  # only make this amount of iterations when forcing finding a solution
        }
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
             'title': "Isoradials for R = {}".format(radius)}  # default values
        self.radii_b = []
        self.angles = []
        self.X = []
        self.Y = []
        self.redshift_factors = []

        self.calculate()

    def calculate_coordinates(self, _tqdm=False):
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
            b_ = calc_impact_parameter(self.radius, self.t, alpha_, self.M, n=self.order, **self.solver_params)
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
        self.X, self.Y = polar_to_cartesian_lists(self.radii_b, self.angles, rotation=-np.pi / 2)
        return angles, impact_parameters

    def calc_redshift_factors(self):
        """Calculates the redshift factor (1 + z) over the line of the isoradial"""
        redshift_factors = [redshift_factor(radius=self.radius, angle=angle, incl=self.t, M=self.M, b_=b_)
                            for b_, angle in zip(self.radii_b, self.angles)]
        self.redshift_factors = redshift_factors
        return redshift_factors

    def calculate(self):
        self.calculate_coordinates()
        self.calc_redshift_factors()

    def find_angle(self, z) -> [int]:
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

    def calc_between(self, ind):
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
        b_ = calc_impact_parameter(self.radius, self.t, mid_angle, self.M, **self.solver_params)
        z_ = redshift_factor(self.radius, mid_angle, self.t, self.M, b_)
        self.radii_b.insert(ind + 1, b_)
        self.angles.insert(ind + 1, mid_angle)
        self.redshift_factors.insert(ind + 1, z_)

    def force_intersection(self, redshift):
        # TODO: improve this method, currently does not seem to work
        """
        If you know a redshift should exist on the isoradial, use this function to calculate the isoradial until
        it finds it. Useful for when the redshift you're looking for equals (or is close to) the maximum
        redshift along some isoradial line.

        Only works if the redshift can be found within the isoradial begin and end angle.
        """

        if len(self.angles) == 2:
            self.calc_between(0)
        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        cross = np.where(np.diff(np.sign(diff)))[0]
        if len(cross):
            return diff  # intersection is found

        it = 0
        while len(cross) == 0 and it < self.find_redshift_params['max_force_iter']:
            # calc derivatives
            delta = [e - b for b, e in zip(self.redshift_factors[:-1], self.redshift_factors[1:])]
            # where does the redshift go back up/down before it reaches the redshift we want to find
            initial_guess_indices = np.where(np.diff(np.sign(delta)))[0]
            new_ind = initial_guess_indices[0]  # initialize the initial guess.
            self.calc_between(new_ind)  # insert more accurate solution
            diff = [redshift + 1 - z_ for z_ in self.redshift_factors]  # calc new interval
            cross = np.where(np.diff(np.sign(diff)))[0]
            it += 1
        return diff

    def calc_redshift_location(self, redshift, midpoint_steps=10, cartesian=False):
        """
        Calculates which location on the isoradial has some redhsift value (not redhisft factor)
        Doest this by means of a midpoint method, with :param midpoint_steps: steps. The isoradial is thus
        calculated at locations closer and closer to where the desired redshift is.
        It does not matter all that much how high the isoradial resolution is, since :param midpoint_steps: is
        much more important to find an accurate location.
        """

        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        if self.find_redshift_params['force_redshift_solution']:
            diff = self.force_intersection(redshift)
        initial_guess_indices = np.where(np.diff(np.sign(diff)))[0]

        angle_solutions = []
        b_solutions = []
        if len(initial_guess_indices):
            for s in range(len(initial_guess_indices)):  # generally, two solutions exists on a single isoradial
                new_ind = initial_guess_indices[s]  # initialize the initial guess.
                for _ in range(midpoint_steps):
                    self.calc_between(new_ind)  # insert more accurate solution
                    diff = [redshift + 1 - z_ for z_ in self.redshift_factors[new_ind:new_ind + 3]]  # calc new interval
                    start = np.where(np.diff(np.sign(diff)))[0]  # returns index where the sign changes
                    new_ind += start[0]  # index of new redshift solution in refined isoradial
                # append average values of final interval
                angle_solutions.append(.5 * (self.angles[new_ind] + self.angles[new_ind + 1]))
                b_solutions.append(.5 * (self.radii_b[new_ind] + self.radii_b[new_ind + 1]))
                # update the initial guess indices, as the indexing has changed due to inserted solutions
                initial_guess_indices = [e + midpoint_steps for e in initial_guess_indices]
            if cartesian:
                return polar_to_cartesian_lists(b_solutions, angle_solutions)
        return angle_solutions, b_solutions

    def plot_redshift(self, fig=None, ax=None, show=True):
        """
        Plots the redshift values along the isoradial line in function of the angle<
        """
        fig_ = fig if fig else plt.figure()
        ax_ = ax if ax else fig_.add_subplot()
        ax_.plot(self.angles, [z - 1 for z in self.redshift_factors])
        plt.title("Redshift values for isoradial\nR={} | M = {}".format(20, M))
        ax_.set_xlim([0, 2 * np.pi])
        if show:
            plt.show()


class Isoredshift:
    # TODO: isoredshift should be initialised from either some coordinates (implemented) or
    #  without (iterative procedure: calc co at R=6M, expand isoradials until stopping criterion )
    def __init__(self, inclination, redshift, bh_mass, ir_solver_params=None, coordinates: {} = None):
        # Parent black hole parameters
        self.t = inclination
        self.M = bh_mass
        self.t = inclination
        self.redshift = redshift

        # Parent isoradial(s) solver parameters: recycled here.
        # TODO: currently same as Isoradial out of laziness, but these might require different solver params
        self.isoradial_solver_params = ir_solver_params if ir_solver_params else \
            {'initial_guesses': 20,
             'midpoint_iterations': 5,
             'plot_inbetween': False,
             'min_periastron': 3.1 * self.M}

        # Isoredshift attributes
        self.radii_w_coordinates_dict = coordinates if coordinates is not None else {}
        self.coordinates_with_radii_dict = self.__init_co_to_radii_dict()
        self.ir_radii_w_co = [key for key, val in self.radii_w_coordinates_dict.items() if
                              len(val[0]) > 0]  # list of R that have solution
        self.co = self.angles, self.radii = self.__extract_co_from_solutions_dict(coordinates)
        self.maxR = max(self.radii)
        self.x, self.y = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)
        self.order_coordinates()

    def __update(self):
        self.ir_radii_w_co = [key for key, val in self.radii_w_coordinates_dict.items() if
                              len(val[0]) > 0]  # list of R that have solution
        self.co = self.angles, self.radii = self.__extract_co_from_solutions_dict(self.radii_w_coordinates_dict)
        self.x, self.y = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)
        self.order_coordinates()

    def __add_solution(self, angle, radius_b, radius_ir):
        """
        Updates all attributes to contain newly found solution
        :return:
        """
        if radius_ir in self.radii_w_coordinates_dict:  # radius is already considered
            if len(self.radii_w_coordinates_dict[radius_ir][0]):  # radius already has a solution
                self.radii_w_coordinates_dict[radius_ir][0].append(angle)
                self.radii_w_coordinates_dict[radius_ir][1].append(radius_b)
            else:
                self.radii_w_coordinates_dict[radius_ir] = [[angle], [radius_b]]
        else:
            self.radii_w_coordinates_dict[radius_ir] = [[angle], [radius_b]]
        self.coordinates_with_radii_dict[(angle, radius_b)] = radius_ir
        self.__update()

    def __add_solutions(self, angles, radii_b, radius_ir):
        for angle, radius_b in zip(angles, radii_b):
            self.__add_solution(angle, radius_b, radius_ir)

    def __init_co_to_radii_dict(self):
        to_return = {}
        for radius, co in self.radii_w_coordinates_dict.items():
            if len(co[0]):  # if radius has solution
                co1, co2 = [tuple(e) for e in np.array(co).T]  # TODO do these need to be lists actually?
                to_return[co1] = radius
                to_return[co2] = radius
        return to_return

    def __extract_co_from_solutions_dict(self, solutions):
        # TODO: might be deprecated on latest addition of __initCoWithRadiiDict (these keys are just the co anyways)
        a = []
        r = []
        for key, val in solutions.items():
            if len(val[0]) > 0:  # at least one solution was found
                angles, radii = val
                [a.append(angle) for angle in angles]
                [r.append(radius) for radius in radii]
        self.co = self.angles, self.radii = a, r
        return a, r

    def split_co_on_solutions(self):
        """
        Iterates the dictionary of coordinates that looks like {r_0: [[angle1, angle2], [b_1, b_2]],
        r_1: [[...], [...]]}
        Checks if each key (radius corresponding to an isoradial) has solutions for the isoredshift or not.
        Splits the original dict in two: one with solutions and one without solutions

        :returns: two dictionaries: one with solutions and one without.
        """
        keys_w_s = []
        keys_wo_s = []
        for key in self.radii_w_coordinates_dict:
            if len(self.radii_w_coordinates_dict[key][0]) == 0:
                keys_wo_s.append(key)
            else:
                keys_w_s.append(key)
        dict_w_s = {key: self.radii_w_coordinates_dict[key] for key in keys_w_s}
        dict_wo_s = {key: self.radii_w_coordinates_dict[key] for key in keys_wo_s}
        return dict_w_s, dict_wo_s

    def calc_core_coordinates(self):
        """Calculates the coordinates of the redshift on the closest possible isoradial: 6*M (= 2*R_s)"""
        ir = Isoradial(6 * self.M, self.t, self.M, order=0, **self.isoradial_solver_params)
        co = ir.calc_redshift_location(self.redshift)
        return co

    def calc_coordinates(self, dirty_isoradials: [Isoradial], midpoint_steps):
        """Calculates the isoredshift for a single redshift value"""
        # TODO: is this deprecated? Aren't isoredshifts *initialised* from coordinates to begin with?
        _a = []
        _radius = []
        t = tqdm(dirty_isoradials, position=1, leave=False)
        for ir in dirty_isoradials:
            t.set_description("Calculating isoredshift {} at R={:.2f}".format(self.redshift, ir.radius))
            t.update()
            a, radius = ir.calc_redshift_location(self.redshift, midpoint_steps, cartesian=False)
            _a.append(a)
            _radius.append(radius)
        self.a = _a
        self.radius = _radius
        return _a, _radius

    def order_coordinates(self, plot_inbetween=False, plot_title=""):
        angles, radii = self.co
        co = [(a, r) for a, r in zip(angles, radii)]
        x, y = polar_to_cartesian_lists(radii, angles)
        cx, cy = np.mean(x, axis=0), np.mean(y, axis=0)

        sorted_co = sorted(co, key=lambda polar_point: get_angle_around(
            [cx, cy], polar_to_cartesian_single(polar_point[0], polar_point[1]))
                           )

        if plot_inbetween:
            # use this to get a visual overview of what happens when ordering the isoradial points using
            # getAngleAround() as a key
            fig, ax = plt.subplots()
            for i, p in enumerate(sorted_co):
                plt.plot(*polar_to_cartesian_single(*p), 'bo')
                plt.text(x[i] * (1 + 0.01), y[i] * (1 + 0.01), i, fontsize=12)
            plt.scatter(cx, cy)
            plt.plot([0, cx], [0, cy])
            plt.title(plot_title)
            plt.show()

        self.co = self.angles, self.radii = [e[0] for e in sorted_co], [e[1] for e in sorted_co]
        self.x, self.y = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)

    def calc_between_angles(self, radius, begin_angle=0, end_angle=np.pi, angular_precision=2, mirror=False,
                            plot_inbetween=False, title='', force_solution=False):
        ir = Isoradial(radius=radius, incl=self.t, bh_mass=self.M,
                       angular_properties={'start_angle': begin_angle,
                                           'end_angle': end_angle,
                                           'angular_precision': angular_precision,
                                           'mirror': mirror})
        ir.find_redshift_params['force_redshift_solution'] = force_solution
        a, r = ir.calc_redshift_location(self.redshift, self.isoradial_solver_params['midpoint_iterations'],
                                         cartesian=False)
        if plot_inbetween:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.axhline(self.redshift)
            fig.suptitle(title)
            ir.plot_redshift(fig, ax, show=False)
            fig.savefig('Plots/{}.png'.format(title))
        return a, r

    def improve_between_all_solutions_once(self):
        """
        Calculates the redshift on the isoredshift line between the already known redshifts
        Does so by calculating the entire isoradial (with low precision) inbetween the radii corresponding
        to these redshift solutions and calculating the redshifts on this isoradial
        """

        self.order_coordinates()  # TODO: is this necessary or already done before? currently depends on further implementation
        co = [(angle, radius_b) for angle, radius_b in zip(*self.co)]
        i = 0
        for b, e in zip(co[:-1], co[1:]):
            r_inbetw = .5 * (self.coordinates_with_radii_dict[b] + self.coordinates_with_radii_dict[e])
            begin_angle, end_angle = b[0], e[0]
            if end_angle - begin_angle > np.pi:
                begin_angle, end_angle = end_angle, begin_angle
            # calc location of redshift, guaranteed to exist between the angles begin_angle and end_angle
            # NOT guaranteed to exist at r_inbetw (isoradial radius, not impact parameter):
            #   1. If coordinates aren't split on a jump (either on the black hole or if they never meet at inf)
            #   2. If we're trying to find one at a tip -> should be covered by other methods though.
            a, r = self.calc_between_angles(r_inbetw, begin_angle - .1, end_angle + .1, plot_inbetween=False,
                                            title='between p{} and p{}'.format(i, i + 1), force_solution=True)
            i += 1
            if len(a):
                self.__add_solutions(a, r, r_inbetw)

    def improve_closest_isoradial_wo_solutions(self, angular_precision=3):
        """
        Recalculates the first (closest) isoradial that did not find a solution with more angular precision.
        Isoradial is recalculated withing the angular interval of the two last (furthest) solutions.
        This is done to guarantee that the lack of solutions is not due to lack of angular precision.

        :param angular_precision: the angular precision at which to recalculate the isoradial

        :return: 0 if success
        """

        r_w_s, r_wo_s = self.split_co_on_solutions()
        angle_interval, last_radii = self.radii_w_coordinates_dict[max(r_w_s.keys())]
        assert len(angle_interval) > 1, "1 or less angles found for corresponding isoradial R={}".format(max(r_w_s))
        closest_r_wo_s = min(r_wo_s.keys())
        begin_angle, end_angle = angle_interval
        if end_angle - begin_angle > np.pi:  # in case the angle is around 0 and 2pi
            begin_angle, end_angle = end_angle, begin_angle  # this works, apparently
        # calculate solutions and add them to the class attributes if they exist
        a, r = self.calc_between_angles(closest_r_wo_s, begin_angle, end_angle, angular_precision=angular_precision,
                                        mirror=True)
        if len(a):
            self.__add_solutions(a, r, closest_r_wo_s)
        return a, r

    def retry_ir_wo_solutions(self, angular_precision=5, plot_inbetween=False):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s.keys()) > 0:
            a, r = self.improve_closest_isoradial_wo_solutions()  # re-calculate isoradials where no solutions were found
            self.order_coordinates(plot_inbetween=plot_inbetween, plot_title="improving tip angular")
            while len(a) > 0 and len(r_wo_s.keys()) > 0:
                a, r = self.improve_closest_isoradial_wo_solutions()  # re-calculate isoradials where no solutions were found
                r_w_s, r_wo_s = self.split_co_on_solutions()
                self.order_coordinates(plot_inbetween=plot_inbetween, plot_title="improving tip angular")

    def calc_closest_ir_wo_solutions_between(self, angular_precision=5, angular_margin=.6):
        """
        Given two isoradials (one with solutions and one without), calculates a new isoradial inbetween the two.
        Either a solution is found, or the location of the tip of the isoredshift is more closed in>
        """
        r_w_s, r_wo_s = self.split_co_on_solutions()
        angle_interval, last_radii = self.radii_w_coordinates_dict[max(r_w_s.keys())]  # one isoradial: two angles/radii
        if len(r_wo_s.keys()) > 0:  # assert there are radii considered without solutions
            first_r_wo_s = min(r_wo_s.keys())
            last_r_w_s = max(r_w_s.keys())
            inbetween_r = .5 * (first_r_wo_s + last_r_w_s)
            begin_angle, end_angle = angle_interval
            if end_angle - begin_angle > np.pi:  # in case the angle is around 0 and 2pi
                begin_angle, end_angle = end_angle, begin_angle  # this works, apparently
            a, r = self.calc_between_angles(inbetween_r, begin_angle - angular_margin, end_angle + angular_margin,
                                            angular_precision=angular_precision, mirror=False)
            if len(a):
                self.__add_solutions(a, r, inbetween_r)
            else:
                self.radii_w_coordinates_dict[inbetween_r] = [[], []]

    def improve_tip(self, iterations=6, plot_inbetween=False):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s.keys()) > 0:
            for _ in range(iterations):
                self.calc_closest_ir_wo_solutions_between(angular_precision=5)
                self.order_coordinates(plot_inbetween=plot_inbetween, plot_title="Improving tip iteratively")

    def improve(self, times_inbetween=2, times_tip=6,
                plot_inbetween=False):  # TODO: figure out good number for @param times
        # Improve the initial dirty isoradials
        self.retry_ir_wo_solutions(plot_inbetween=False)
        self.improve_tip(iterations=times_tip, plot_inbetween=plot_inbetween)
        for n in range(times_inbetween):
            # TODO: why doesn't he calculate at the tip region?
            # TODO: maybe due to the fact that the angular interval is too strict? let's see what angular margin does
            self.improve_between_all_solutions_once()  # TODO is this the correct value
            self.order_coordinates(plot_inbetween=plot_inbetween, plot_title="calculating inbetween")

    def splitCoOnJump(self):
        """
        Re-adjusts the order of the coordinate dictionary so that they are:
        1. sorted
        2. the first element is at the photosphere, and the last element is back at the photosphere

        This function assumes that, if no solutions for a isoredshift are found for some radius, it is
        because it does not exist (and not because the solver hasn't found it yet).
        This is generally True after having called :func:improve().
        :return:
        """

        self.order_coordinates()
        self.__update()
        split = 0
        dist = []
        for i in range(len(self.co) - 1):
            p1, p2 = zip(self.co[i], self.co[i + 1])
            angles, radii = np.asarray(p1, p2).T
            print("angles: ", angles)
            p1_c, p2_c = polar_to_cartesian_lists(radii, angles)
            print("cartesian: {}".format(p1_c))
            print("cartesian 2: {}".format(p2_c))
            center = np.mean([p1, p2], axis=0)
            print("center: ", center)
            if center < 6 * self.M:
                split = i

        self.co = self.co[split:] + self.co[:split]
        print("coordinates: {}".format(self.co))

    def plot(self, ax, norm, color_map):
        color = cm.ScalarMappable(norm=norm, cmap=color_map).to_rgba(self.redshift)
        plt.plot(self.y, [-e for e in self.x], color=color)  # TODO: hack to correctly orient plot
        self.improve_closest_isoradial_wo_solutions()
        plt.plot(self.y, [-e for e in self.x], color=color)  # TODO: hack to correctly orient plot
        tries = 0
        while len(self.ir_radii_w_co) < 10 and tries < 10:
            self.improve_between_all_solutions_once()
            tries += 1

        plt.plot(self.y, [-e for e in self.x], color=color)  # TODO: hack to correctly orient plot


def polar_to_cartesian_lists(radii, angles, rotation=0):
    x = []
    y = []
    for R, th in zip(radii, angles):
        x.append(R * np.cos(th + rotation))
        y.append(R * np.sin(th + rotation))
    return x, y


def polar_to_cartesian_single(th, R, rotation=0):
    x = R * np.cos(th + rotation)
    y = R * np.sin(th + rotation)
    return x, y


def cartesian_to_polar(x, y):
    R = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    th = th if th > 0 else th + 2 * np.pi
    return th, R


def get_angle_around(p1, p2):
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
    angle_center, _ = cartesian_to_polar(cx, cy)
    # rotate p2_ counter clockwise until the vector to the isoradial center is aligned with negative y-axis
    theta = np.pi - angle_center
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    p2_ = np.dot(rot, p2_)
    angle_target, _ = cartesian_to_polar(p2[0], p2[1])
    angle_target_around_center, R_target_around_center = cartesian_to_polar(p2_[0], p2_[1])
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
    # bh.writeFrames(direct_r=[6, 10, 20, 30], ghost_r=[6, 10, 20, 30], start=0, end=180, step_size=5, ax_lim=(-35, 35))
    bh.solver_params = {'initial_guesses': 12,
                        'midpoint_iterations': 11,  # 5 is fine for direct image. Up until 10 for ghost image of large R
                        'plot_inbetween': False,
                        'min_periastron': 3.01,
                        'elliptic_integral_interval': (0, 2 * np.pi),
                        # which section should be calculated with an elliptic integral
                        'use_ellipse': True}
    bh.plot_params['plot_core'] = False
    bh.plot_params['plot_isoredshifts_inbetween'] = False
    bh.angular_properties['angular_precision'] = 100
    # bh.samplePoints(N=10000, maxR=200)
    #bh.plotIsoRedshifts(redshifts=[-.15], r_precision=20)
    bh.plot_isoredshifts_from_points()
    # TODO: check why Isoredshift.calcInbetween doesn't always find a solution
    # TODO: split on jump
    # fig, ax = bh.plotIsoradials([6, 10, 20, 30], [6, 10, 30, 1000], ax_lim=(-35, 35))
    # bh.samplePoints(N=5000, minR=6, maxR=40)
    # bh.plotPoints(powerscale=1)
