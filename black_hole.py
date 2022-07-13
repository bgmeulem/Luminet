import os.path
import pandas as pd
import matplotlib.cm as cm
import matplotlib.collections as mcoll
import matplotlib.image as img
from collections import OrderedDict
from black_hole_math import *
import configparser

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors


class BlackHole:
    def __init__(self, mass=1., inclination=80, acc=10e-8):
        """Initialise black hole with mass and accretion rate
        Set viewer inclination above equatorial plane
        """
        self.t = inclination * np.pi / 180
        self.M = mass
        self.acc = acc  # accretion rate
        self.critical_b = 3 * np.sqrt(3) * self.M
        self.settings = {}  # All settings: see below
        self.plot_params = {}
        self.ir_parameters = {}
        self.angular_properties = {}
        self.iz_solver_params = {}
        self.solver_params = {}
        self.__read_parameters()

        self.disk_outer_edge = 50. * self.M
        self.disk_inner_edge = 6. * self.M
        self.disk_apparent_outer_edge = self.calc_apparent_outer_disk_edge()  # outer edge after curving through spacetime
        self.disk_apparent_inner_edge = self.calc_apparent_inner_disk_edge()  # inner edge after curving through spacetime

        self.isoradials = {}
        self.isoredshifts = {}

    def __read_parameters(self):
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('parameters.ini')
        for i, section in enumerate(config.sections()):
            self.settings[section] = {key: eval(val) for key, val in config[section].items()}
        self.plot_params = self.settings["plot_params"]
        self.ir_parameters["isoradial_angular_parameters"] = self.angular_properties = self.settings[
            "isoradial_angular_parameters"]
        self.ir_parameters["isoradial_solver_parameters"] = self.solver_params = self.settings["solver_parameters"]
        self.iz_solver_params = self.settings["isoredshift_solver_parameters"]

    def apparent_inner_edge(self, cartesian=True, scale=.99):
        """
        The apparent inner edge of the black hole (not the apparent inner disk edge). THis takes eventual ghost images
        into account
        """
        b = []
        a = []
        for a_ in np.linspace(0, 2 * np.pi, self.angular_properties["angular_precision"]):
            a.append(a_)
            if np.pi / 2 < a_ < 3 * np.pi / 2:
                b.append(self.critical_b * scale)
            else:
                b_ = min(self.critical_b, self.disk_apparent_inner_edge.get_b_from_angle(a_))
                b.append(b_ * scale)
        if not cartesian:
            return b, a
        else:
            return polar_to_cartesian_lists(b, a, rotation=-np.pi / 2)

    def plot_photon_sphere(self, _ax, c='red'):
        # plot black hole itself
        x, y = self.apparent_inner_edge(cartesian=True)
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

    def calc_apparent_inner_disk_edge(self):
        ir = Isoradial(radius=self.disk_inner_edge, incl=self.t, order=0,
                       params=self.ir_parameters, bh_mass=self.M)
        ir.radii_b = [.99 * b for b in ir.radii_b]  # scale slightly down?  # TODO
        ir.X, ir.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)
        return ir

    def calc_apparent_outer_disk_edge(self):
        ir = Isoradial(radius=self.disk_outer_edge, incl=self.t, order=0,
                       params=self.ir_parameters, bh_mass=self.M)
        ir.X, ir.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)
        return ir

    def get_apparent_outer_edge_radius(self, angle, rotation=0):
        return self.disk_apparent_outer_edge.get_b_from_angle(angle + rotation)

    def get_apparent_inner_edge_radius(self, angle, rotation=0):
        return self.disk_apparent_inner_edge.get_b_from_angle(angle + rotation)

    def plot_apparent_inner_edge(self, _ax, linestyle='--'):
        # plot black hole (photon sphere)
        # TODO why don't I use the Isoradial class for this?
        x = []
        y = []
        impact_parameters = []
        angles = np.linspace(0, 2 * np.pi, 2 * self.angular_properties["angular_precision"])
        for a in angles:
            b = self.get_apparent_inner_edge_radius(a)
            impact_parameters.append(b)
            rot = -np.pi / 2 if self.t < np.pi / 2 else np.pi / 2
            x_, y_ = polar_to_cartesian_lists([b], [a], rotation=rot)
            x.append(x_)
            y.append(y_)

        _ax.plot(x, y, zorder=0, linestyle=linestyle, linewidth=2. * self.plot_params["linewidth"])
        return _ax

    def get_figure(self):
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
        _ax.set_ylim(self.plot_params["ax_lim"])
        _ax.set_xlim(self.plot_params["ax_lim"])
        return _fig, _ax

    def calc_isoredshifts(self, redshifts=None):
        if redshifts is None:
            redshifts = [-.15, 0., .1, .20, .5]

        def get_dirty_isoradials(__bh):
            # an array of quick and dirty isoradials for the initial guesses of redshifts
            isoradials = []  # for initial guesses
            for radius in np.linspace(__bh.disk_inner_edge, __bh.disk_outer_edge,
                                      __bh.iz_solver_params["initial_radial_precision"]):
                isoradial = Isoradial(radius, __bh.t, __bh.M,
                                      params=__bh.ir_parameters)
                isoradials.append(isoradial)
            return isoradials

        dirty_isoradials = get_dirty_isoradials(self)
        t = tqdm(redshifts, desc="Calculating redshift", position=0)
        for redshift in t:
            t.set_description("Calculating redshift {}".format(redshift))
            dirty_ir_copy = dirty_isoradials.copy()
            # spawn an isoredshift instance and calc coordinates based on dirty isoradials
            iz = Isoredshift(inclination=self.t, redshift=redshift, bh_mass=self.M,
                             solver_parameters=self.iz_solver_params, from_isoradials=dirty_ir_copy)
            # iteratively improve coordinates and closing tip of isoredshift
            iz.improve()
            self.isoredshifts[redshift] = iz
        return self.isoredshifts

    def add_isoradial(self, isoradial, radius, order):
        """
        Add isoradial to dict of isoradials. Each key is a radius corresponding to
        some set of isoradials. Each value is again a dict, with as keys the order
        of the isoradial (usually just 0 for direct and 1 for ghost image)
        """
        if radius in self.isoradials.keys():
            self.isoradials[radius][order] = isoradial
        else:
            self.isoradials[radius] = {order: isoradial}

    def calc_isoradials(self, direct_r: [], ghost_r: []):
        progress_bar = tqdm(range(len(direct_r) + len(ghost_r)), position=0, leave=False)
        # calc ghost images
        progress_bar.set_description("Ghost images")
        self.plot_params['alpha'] = .5
        for radius in sorted(ghost_r):
            progress_bar.update(1)
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=1,
                                  params=self.ir_parameters, plot_params=self.plot_params)
            self.add_isoradial(isoradial, radius, 1)

        # calc direct images
        progress_bar.set_description("Direct images")
        self.plot_params['alpha'] = 1.
        for radius in sorted(direct_r):
            progress_bar.update(1)
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = Isoradial(radius, self.t, self.M, order=0,
                                  params=self.ir_parameters, plot_params=self.plot_params)
            self.add_isoradial(isoradial, radius, 0)

    def plot_isoradials(self, direct_r: [], ghost_r: [], show=False):
        """Given an array of radii for the direct image and/or ghost image, plots the corresponding
        isoradials.
        Calculates the isoradials according to self.root_params
        Plots the isoradials according to self.plot_params"""

        def plot_ellipse(__r, __ax, incl):
            ax_ = __ax
            a = np.linspace(-np.pi, np.pi, 2 * self.angular_properties['angular_precision'])
            ell = [ellipse(__r, a_, incl) for a_ in a]
            x, y = polar_to_cartesian_lists(ell, a)
            ax_.plot(x, y, color='red', zorder=-1)
            return ax_

        self.calc_isoradials(direct_r, ghost_r)
        _fig, _ax = self.get_figure()
        color_range = (-1, 1)

        # plot background
        if self.plot_params['orig_background']:
            image = img.imread('bh_background.png')
            scale = (940 / 30 * 2. * M)  # 940 px by 940 px, and 2M ~ 30px
            _ax.imshow(image, extent=(-scale / 2, scale / 2, -scale / 2, scale / 2))
        else:
            _ax.set_facecolor('black')

        # plot ghost images
        self.plot_params['alpha'] = .5
        for radius in sorted(ghost_r):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = self.isoradials[radius][1]
            plt_, _ax = isoradial.plot(_ax, self.plot_params, colornorm=color_range)

        # plot direct images
        self.plot_params['alpha'] = 1.
        for radius in sorted(direct_r):
            self.plot_params['key'] = 'R = {}'.format(radius)
            isoradial = self.isoradials[radius][0]
            plt_, _ax = isoradial.plot(_ax, self.plot_params, colornorm=color_range)

        if self.plot_params['plot_ellipse']:  # plot ellipse
            for radius in direct_r:
                _ax = plot_ellipse(radius, _ax, self.t)
        if self.plot_params['plot_core']:
            _ax = self.plot_apparent_inner_edge(_ax, 'red')

        plt.title(f"Isoradials for M={self.M}", color=self.plot_params['text_color'])
        if show:
            plt.show()
        if self.plot_params['save_plot']:
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            _fig.savefig(name, dpi=300, facecolor=self.plot_params['face_color'])
        return _fig, _ax

    def write_frames(self, func, direct_r=None, ghost_r=None, step_size=5):
        """
        Given some function that produces  fig and ax, this method sets increasing values for the inclination,
        plots said function and write it out as a frame.
        """
        if ghost_r is None:
            ghost_r = [6, 10, 20, 30]
        if direct_r is None:
            direct_r = [6, 10, 20, 30]
        steps = np.linspace(0, 180, 1 + (0 - 180) // step_size)
        for a in tqdm(steps, position=0, desc='Writing frames'):
            self.t = a
            bh.plot_params['title'] = 'inclination = {:03}°'.format(int(a))
            fig_, ax_ = func(direct_r, ghost_r, ax_lim=self.plot_params["ax_lim"])
            name = self.plot_params['title'].replace(' ', '_')
            name = name.replace('°', '')
            fig_.savefig('movie/' + name, dpi=300, facecolor=self.plot_params['face_color'])
            plt.close()  # to not destroy your RAM

    def plot_isoredshifts(self, redshifts=None, plot_core=False):
        if redshifts is None:
            redshifts = [-.2, -.15, 0., .15, .25, .5, .75, 1.]
        _fig, _ax = self.get_figure()  # make new figure

        bh.calc_isoredshifts(redshifts=redshifts).values()

        for redshift, irz in self.isoredshifts.items():
            r_w_s, r_wo_s = irz.split_co_on_solutions()
            if len(r_w_s.keys()):
                split_index = irz.split_co_on_jump()
                if split_index is not None:
                    plt.plot(irz.y[:split_index], [-e for e in irz.x][:split_index],
                             linewidth=self.plot_params["linewidth"])
                    plt.plot(irz.y[split_index + 1:], [-e for e in irz.x][split_index + 1:],
                             linewidth=self.plot_params["linewidth"])
                else:
                    plt.plot(irz.y, [-e for e in irz.x],
                             linewidth=self.plot_params["linewidth"])  # todo: why do i need to flip x

        if plot_core:
            _ax = self.plot_apparent_inner_edge(_ax, linestyle='-')
        plt.suptitle("Isoredshift lines for M={}".format(self.M))
        plt.show()
        return _fig, _ax

    def sample_points(self, n_points=1000, f=None, f2=None):
        """
        # TODO: sample separately for direct and ghost image?
        Samples points on the accretion disk. This sampling is not done uniformly, but a bias is added towards the
        center of the accretion disk, as the observed flux is exponentially bigger here and this needs the most
        precision.
        Both the direct and ghost image for each point is calculated. It's coordinates (polar and cartesian),
        redshift and
        :param min_radius:
        :param max_radius:
        :param n_points: Amount of points to sample. 10k takes about 6 minutes and gives ok precision mostly
        :param f:
        :param f2:
        :return:
        """
        if f is None:
            f = f"Points/points_incl={int(self.t * 180 / np.pi)}.csv"
        if f2 is None:
            f2 = f"Points/points_secondary_incl={int(self.t * 180 / np.pi)}.csv"
        df = pd.read_csv(f, index_col=0) if os.path.exists('./{}'.format(f)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'angle', 'z_factor', 'flux_o'])
        df2 = pd.read_csv(f2, index_col=0) if os.path.exists('./{}'.format(f2)) else \
            pd.DataFrame(columns=['X', 'Y', 'impact_parameter', 'angle', 'z_factor', 'flux_o'])

        min_radius_ = self.disk_inner_edge
        max_radius_ = self.disk_outer_edge
        t = tqdm(range(n_points), desc="Sampling points for direct and ghost image")
        for _ in t:
            t.update(1)
            # r = minR_ + maxR_ * np.sqrt(np.random.random())  # uniformly sampling a circle's surface
            theta = np.random.random() * 2 * np.pi
            r = min_radius_ + max_radius_ * np.random.random()  # bias towards center (where the interesting stuff is)
            b_ = calc_impact_parameter(r, incl=self.t, _alpha=theta, bh_mass=self.M, **self.solver_params)
            b_2 = calc_impact_parameter(r, incl=self.t, _alpha=theta, bh_mass=self.M, **self.solver_params, n=1)
            if b_ is not None:
                x, y = polar_to_cartesian_lists([b_], [theta], rotation=-np.pi / 2)
                redshift_factor_ = redshift_factor(r, theta, self.t, self.M, b_)
                f_o = flux_observed(r, self.acc, self.M, redshift_factor_)
                df = pd.concat([df,
                                pd.DataFrame.from_dict({'X': x, 'Y': y, 'impact_parameter': b_,
                                                        'angle': theta,
                                                        'z_factor': redshift_factor_, 'flux_o': f_o})])
            if b_2 is not None:
                x, y = polar_to_cartesian_lists([b_2], [theta], rotation=-np.pi / 2)
                redshift_factor_2 = redshift_factor(r, theta, self.t, self.M, b_2)
                F_o2 = flux_observed(r, self.acc, self.M, redshift_factor_2)
                df2 = pd.concat([df2,
                                 pd.DataFrame.from_dict({'X': x, 'Y': y, 'impact_parameter': b_2,
                                                         'angle': theta,
                                                         'z_factor': redshift_factor_2, 'flux_o': F_o2})
                                 ])
        df.to_csv(f)
        df2.to_csv(f2)

    def plot_points(self, power_scale=.9, levels=100):
        """
        Plot the points written out by samplePoints()
        :param levels: amount of levels in matplotlib contour plot
        :param power_scale: powers_cale to apply to flux. No power_scale = 1. Anything lower than 1 will make the
        dim points pop out more.
        :return:
        """

        def plot_direct_image(_ax, points, _levels, _min_flux, _max_flux, _power_scale):
            # direct image
            points.sort_values(by="angle", inplace=True)
            points_ = points.iloc[[b_ <= self.get_apparent_outer_edge_radius(a_) for b_, a_ in
                                   zip(points["impact_parameter"], points["angle"])]]
            fluxes = [(abs(fl + _min_flux) / (_max_flux + _min_flux)) ** _power_scale for fl in points_['flux_o']]
            _ax.tricontourf(points_['X'], points_['Y'], fluxes, cmap='Greys_r',
                            levels=_levels, norm=plt.Normalize(0, 1), nchunk=2
                            )
            br = self.calc_apparent_inner_disk_edge()
            _ax.fill_between(br.X, br.Y, color='black', zorder=1)  # to fill Delauney triangulation artefacts with black
            return _ax

        def plot_ghost_image(_ax, points, _levels, _min_flux, _max_flux, _power_scale):
            # ghost image
            points_inner = points.iloc[[b_ < self.get_apparent_inner_edge_radius(a_ + np.pi) for b_, a_ in
                                        zip(points["impact_parameter"], points["angle"])]]
            points_outer = points.iloc[[b_ > self.get_apparent_outer_edge_radius(a_ + np.pi) for b_, a_ in
                                        zip(points["impact_parameter"], points["angle"])]]
            # _ax.plot(self.disk_apparent_inner_edge.X, self.disk_apparent_inner_edge.Y)
            for i, points_ in enumerate([points_inner, points_outer]):
                points_.sort_values(by=['flux_o'], ascending=False)
                fluxes = [(abs(fl + _min_flux) / (_max_flux + _min_flux)) ** _power_scale for fl in
                          points_['flux_o']]
                _ax.tricontourf(points_['X'], [-e for e in points_['Y']], fluxes, cmap='Greys_r',
                                norm=plt.Normalize(0, 1), levels=_levels, nchunk=2, zorder=1 - i
                                )
            x, y = self.apparent_inner_edge(cartesian=True)
            _ax.fill_between(x, y, color='black', zorder=1)  # to fill Delauney triangulation artefacts with black
            x, y = self.calc_apparent_outer_disk_edge().cartesian_co
            _ax.fill_between(x, y, color='black', zorder=0)  # to fill Delauney triangulation artefacts with black
            return _ax

        _fig, _ax = self.get_figure()
        if self.plot_params["plot_disk_edges"]:
            _ax.plot(self.disk_apparent_outer_edge.X, self.disk_apparent_outer_edge.Y, zorder=4)
            _ax.plot(self.disk_apparent_inner_edge.X, self.disk_apparent_inner_edge.Y, zorder=4)

        points1 = pd.read_csv(f"Points/points_incl={round(self.t * 180 / np.pi)}.csv")
        points2 = pd.read_csv(f"Points/points_secondary_incl={round(self.t * 180 / np.pi)}.csv")
        max_flux = max(max(points1['flux_o']), max(points2['flux_o']))
        min_flux = 0

        _ax = plot_direct_image(_ax, points1, levels, min_flux, max_flux, power_scale)
        _ax = plot_ghost_image(_ax, points2, levels, min_flux, max_flux, power_scale)

        _ax.set_xlim((-40, 40))
        _ax.set_ylim((-40, 40))

        plt.savefig('SampledPoints_incl={}.png'.format(self.t), dpi=300, facecolor='black')
        plt.show()
        return _fig, _ax

    def plot_isoredshifts_from_points(self, levels=None, extension="png"):
        # TODO add ghost image

        if levels is None:
            levels = [-.2, -.15, -.1, -0.05, 0., .05, .1, .15, .2, .25, .5, .75]

        _fig, _ax = self.get_figure()
        points = pd.read_csv(f"points_incl={int(round(self.t * 180 / np.pi))}.csv")
        br = self.calc_apparent_inner_disk_edge()
        color_map = plt.get_cmap('RdBu_r')

        # points1 = addBlackRing(self, points1)
        levels_ = [-.2, -.15, -.1, -0.05, 0., .05, .1, .15, .2, .25, .5, .75]
        _ax.tricontour(points['X'], points['Y'] if self.t <= np.pi / 2 else [-e for e in points['Y']],
                       # TODO why do I have to flip it myself
                       [e for e in points['z_factor']], cmap=color_map,
                       norm=plt.Normalize(0, 2),
                       levels=[e + 1 for e in levels_],
                       nchunk=2,
                       linewidths=2)
        _ax.fill_between(br.X, br.Y, color='black', zorder=2)
        plt.show()
        _fig.savefig(f"Plots/Isoredshifts_incl={str(int(180 * self.t / np.pi)).zfill(3)}.{extension}",
                     facecolor='black', dpi=300)
        return _fig, _ax


class Isoradial:
    def __init__(self, radius, incl, bh_mass, order=0, params=None, plot_params=None, angular_properties=None):
        self.M = bh_mass  # mass of the black hole containing this isoradial
        self.t = incl  # inclination of observer's plane
        self.radius = radius
        self.order = order
        self.params = params if params is not None else {}
        self.angular_properties = params["isoradial_angular_parameters"] if params \
            else angular_properties if angular_properties \
            else {}
        self.solver_params = params["isoradial_solver_parameters"] if params else self.__read_default_solver_params()
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
        self.cartesian_co = self.X, self.Y = [], []
        self.redshift_factors = []

        self.calculate()

    def __read_default_solver_params(self):
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read('parameters.ini')
        return {key: eval(val) for key, val in config["solver_parameters"].items()}

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
            if b_ is not None:
                angles.append(alpha_)
                impact_parameters.append(b_)
        if self.order > 0:  # TODO: fix dirty manual flip for ghost images
            angles = [a_ + np.pi for a_ in angles]

        # flip image if necessary
        if self.t > np.pi / 2:
            angles = [(a_ + np.pi) % (2 * np.pi) for a_ in angles]
        if self.angular_properties['mirror']:  # by default True. Halves computation time for calculating full isoradial
            # add second half of image (left half if 0° is set at South)
            angles += [(2 * np.pi - a_) % (2 * np.pi) for a_ in angles[::-1]]
            impact_parameters += impact_parameters[::-1]
        self.angles = angles
        self.radii_b = impact_parameters
        self.X, self.Y = polar_to_cartesian_lists(self.radii_b, self.angles, rotation=-np.pi / 2)
        self.cartesian_co = self.X, self.Y
        return angles, impact_parameters

    def calc_redshift_factors(self):
        """Calculates the redshift factor (1 + z) over the line of the isoradial"""
        redshift_factors = [redshift_factor(radius=self.radius, angle=angle, incl=self.t, bh_mass=self.M, b_=b_)
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
        # TODO: this method only works if angles augment from index 0 to end
        # if image is flipped, then the mod operator makes it so they jump back to 0 about halfway
        # yielding a fake intersection
        d = [abs(a_ % (2 * np.pi) - angle % (2 * np.pi)) for a_ in self.angles]
        mn = min(d)
        res = [i for i, val in enumerate(d) if val == mn]
        return self.radii_b[res[0]] if len(res) else None

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
        isoradial angle between place ind and ind + 1

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
            # plt.plot(self.angles, [redshift + 1 - z_ for z_ in self.redshift_factors])
            # plt.axvline(0)
            # plt.show()
        return diff

    def calc_redshift_location_on_ir(self, redshift, cartesian=False):
        """
        Calculates which location on the isoradial has some redshift value (not redshift factor)
        Doest this by means of a midpoint method, with midpoint_steps steps (defined in parameters.ini).
        The (b, alpha, z) coordinates of the isoradial are calculated closer and closer to the desired z.
        It does not matter all that much how high the isoradial resolution is, since midpoint_steps is
        much more important to find an accurate location.
        """

        diff = [redshift + 1 - z_ for z_ in self.redshift_factors]
        # if self.find_redshift_params['force_redshift_solution']:
        #     pass  # TODO, force_intersection does not always seem to work
        #     diff = self.force_intersection(redshift)
        initial_guess_indices = np.where(np.diff(np.sign(diff)))[0]

        angle_solutions = []
        b_solutions = []
        if len(initial_guess_indices):
            for s in range(len(initial_guess_indices)):  # generally, two solutions exists on a single isoradial
                new_ind = initial_guess_indices[s]  # initialize the initial guess.
                for _ in range(self.solver_params["midpoint_iterations"]):
                    self.calc_between(new_ind)  # insert more accurate solution
                    diff_ = [redshift + 1 - z_ for z_ in
                             self.redshift_factors[new_ind:new_ind + 3]]  # calc new interval
                    start = np.where(np.diff(np.sign(diff_)))[0]  # returns index where the sign changes
                    new_ind += start[0]  # index of new redshift solution in refined isoradial
                # append average values of final interval
                angle_solutions.append(.5 * (self.angles[new_ind] + self.angles[new_ind + 1]))
                b_solutions.append(.5 * (self.radii_b[new_ind] + self.radii_b[new_ind + 1]))
                # update the initial guess indices, as the indexing has changed due to inserted solutions
                initial_guess_indices = [e + self.solver_params["midpoint_iterations"] for e in initial_guess_indices]
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
    def __init__(self, inclination, redshift, bh_mass, solver_parameters=None, from_isoradials=None):
        # Parent black hole parameters
        if from_isoradials is None:
            from_isoradials = {}

        self.t = inclination
        self.M = bh_mass
        self.t = inclination
        self.redshift = redshift

        # Parent isoradial(s) solver parameters: recycled here.
        # TODO: currently same as Isoradial out of laziness, but these might require different solver params
        self.solver_params = solver_parameters if solver_parameters else \
            {'initial_guesses': 20,
             'midpoint_iterations': 10,
             'plot_inbetween': False,
             'min_periastron': 3.001 * self.M,
             'retry_angular_precision': 30,
             "retry_tip": 15}

        # Isoredshift attributes
        self.radii_w_coordinates_dict = {}
        if from_isoradials is not None:
            self.calc_from_isoradials(from_isoradials)
        else:
            pass  # TODO: initialise from photon sphere isoradial?
        self.coordinates_with_radii_dict = self.__init_co_to_radii_dict()
        self.ir_radii_w_co = [key for key, val in self.radii_w_coordinates_dict.items() if
                              len(val[0]) > 0]  # list of R that have solution
        self.co = self.angles, self.radii = self.__extract_co_from_solutions_dict()
        self.max_radius = max(self.radii) if len(self.radii) else 0
        self.x, self.y = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)
        self.order_coordinates()

    def __update(self):
        self.ir_radii_w_co = [key for key, val in self.radii_w_coordinates_dict.items() if
                              len(val[0]) > 0]  # list of R that have solution
        self.co = self.angles, self.radii = self.__extract_co_from_solutions_dict()
        self.x, self.y = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)
        self.order_coordinates()

    def __add_solutions(self, angles, impact_parameters, radius_ir):
        def __add_solution(__iz: Isoredshift, __angle, __radius_b, __radius_ir):
            """
            Updates all attributes to contain newly found solution
            :return:
            """
            if __radius_ir in __iz.radii_w_coordinates_dict:  # radius is already considered
                if len(__iz.radii_w_coordinates_dict[__radius_ir][0]):  # radius already has a solution
                    __iz.radii_w_coordinates_dict[__radius_ir][0].append(__angle)
                    __iz.radii_w_coordinates_dict[__radius_ir][1].append(__radius_b)
                else:
                    __iz.radii_w_coordinates_dict[__radius_ir] = [[__angle], [__radius_b]]
            else:
                __iz.radii_w_coordinates_dict[__radius_ir] = [[__angle], [__radius_b]]
            __iz.coordinates_with_radii_dict[(__angle, __radius_b)] = __radius_ir
            __iz.__update()

        for angle, impact_parameter in zip(angles, impact_parameters):
            __add_solution(self, angle, impact_parameter, radius_ir)

    def __init_co_to_radii_dict(self):
        to_return = {}
        for radius, co in self.radii_w_coordinates_dict.items():
            if len(co[0]):  # if radius has solution
                coordinates = [tuple(e) for e in np.array(co).T]  # TODO do these need to be lists actually?
                for co_ in coordinates:  # either one or two solutions
                    to_return[co_] = radius
        return to_return

    def __extract_co_from_solutions_dict(self):
        a = []
        r = []
        for key, val in self.radii_w_coordinates_dict.items():
            if len(val[0]) > 0:  # at least one solution was found
                angles, radii = val
                [a.append(angle) for angle in angles]
                [r.append(radius) for radius in radii]
        self.co = self.angles, self.radii = a, r
        return a, r

    def calc_from_isoradials(self, isoradials, cartesian=False):
        """
        Calculates the isoredshift for a single redshift value, based on a couple of isoradials calculated
        at low precision
        """
        solutions = OrderedDict()
        _max_radius = None
        for ir in isoradials:
            # Use the same solver params from the black hole to calculate the redshift location on the isoradial
            a, r = ir.calc_redshift_location_on_ir(self.redshift, cartesian=cartesian)
            solutions[ir.radius] = [a, r]
        self.radii_w_coordinates_dict = solutions
        self.__update()

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
        ir = Isoradial(6. * self.M, self.t, self.M, order=0, **self.solver_params)
        co = ir.calc_redshift_location_on_ir(self.redshift)
        return co

    def order_coordinates(self, plot_title="", plot_inbetween=False):
        angles, radii = self.co
        co = [(a, r) for a, r in zip(angles, radii)]
        x, y = polar_to_cartesian_lists(radii, angles)
        cx, cy = np.mean(x, axis=0), np.mean(y, axis=0)
        order_around = [.3 * cx, .8 * cy]

        sorted_co = sorted(
            co, key=lambda polar_point: get_angle_around(
                order_around, polar_to_cartesian_single(polar_point[0], polar_point[1]))
        )

        if plot_inbetween:
            # use this to get a visual overview of what happens when ordering the isoradial points using
            # getAngleAround() as a key
            fig, ax = plt.subplots()
            for i, p in enumerate(sorted_co):
                plt.plot(*polar_to_cartesian_single(*p), 'bo')
                plt.text(x[i] * (1 + 0.01), y[i] * (1 + 0.01), i, fontsize=12)
            plt.plot(*np.array([polar_to_cartesian_single(*p) for p in sorted_co]).T)
            plt.scatter(*order_around)
            plt.plot([0, order_around[0]], [0, order_around[1]])
            plt.title(plot_title)
            plt.show()
            plt.close('all')

        self.co = self.angles, self.radii = [e[0] for e in sorted_co], [e[1] for e in sorted_co]
        self.x, self.y = polar_to_cartesian_lists(self.radii, self.angles, rotation=0)

    def calc_redshift_on_ir_between_angles(self, radius, begin_angle=0, end_angle=np.pi, angular_precision=3,
                                           mirror=False, plot_inbetween=False, title='', force_solution=False):
        ir = Isoradial(radius=radius, incl=self.t, bh_mass=self.M,
                       angular_properties={'start_angle': begin_angle,
                                           'end_angle': end_angle,
                                           'angular_precision': angular_precision,
                                           'mirror': mirror})
        ir.find_redshift_params['force_redshift_solution'] = force_solution
        a, r = ir.calc_redshift_location_on_ir(self.redshift, cartesian=False)
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
            a, r = self.calc_redshift_on_ir_between_angles(r_inbetw, begin_angle - .1, end_angle + .1,
                                                           plot_inbetween=False,
                                                           title='between p{} and p{}'.format(i, i + 1),
                                                           force_solution=True)
            i += 1
            if len(a):
                self.__add_solutions(a, r, r_inbetw)

    def recalc_redshift_on_closest_isoradial_wo_z(self):
        """
        Recalculates the first (closest) isoradial that did not find a solution with more angular precision.
        Isoradial is recalculated withing the angular interval of the two last (furthest) solutions.
        This is done to guarantee that the lack of solutions is not due to lack of angular precision.

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
        a, b = self.calc_redshift_on_ir_between_angles(closest_r_wo_s, begin_angle, end_angle,
                                                       angular_precision=
                                                       self.solver_params['retry_angular_precision'],
                                                       mirror=False)
        if len(a):
            self.__add_solutions(a, b, closest_r_wo_s)
        return a, b

    def recalc_isoradials_wo_redshift_solutions(self, plot_inbetween=False):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s.keys()) > 0 and len(r_w_so) > 0:
            a, r = self.recalc_redshift_on_closest_isoradial_wo_z()  # re-calculate isoradials where no solutions were found
            self.order_coordinates(plot_title="improving tip angular")
            r_w_so, r_wo_s = self.split_co_on_solutions()
            while len(a) > 0 and len(r_wo_s.keys()) > 0:
                a, r = self.recalc_redshift_on_closest_isoradial_wo_z()  # re-calculate isoradials where no solutions were found
                r_w_s, r_wo_s = self.split_co_on_solutions()
                self.order_coordinates(plot_inbetween=plot_inbetween, plot_title="improving tip angular")

    def calc_ir_before_closest_ir_wo_z(self, angular_margin=.3):
        """
        Given two isoradials (one with solutions and one without), calculates a new isoradial inbetween the two.
        Either a solution is found, or the location of the tip of the isoredshift is more closed in.
        """
        r_w_s, r_wo_s = self.split_co_on_solutions()
        angle_interval, last_radii = self.radii_w_coordinates_dict[max(r_w_s.keys())]  # one isoradial: two angles/radii
        if len(r_wo_s.keys()) > 0 and len(r_w_s) > 0:  # assert there are radii considered without solutions
            first_r_wo_s = min(r_wo_s.keys())
            last_r_w_s = max(r_w_s.keys())
            inbetween_r = .5 * (first_r_wo_s + last_r_w_s)
            begin_angle, end_angle = angle_interval
            if end_angle - begin_angle > np.pi:  # in case the angle is around 0 and 2pi
                begin_angle, end_angle = end_angle, begin_angle  # this works, apparently
            a, r = self.calc_redshift_on_ir_between_angles(inbetween_r, begin_angle - angular_margin,
                                                           end_angle + angular_margin,
                                                           angular_precision=
                                                           self.solver_params['retry_angular_precision'],
                                                           mirror=False)
            if len(a):
                self.__add_solutions(a, r, inbetween_r)
            else:
                self.radii_w_coordinates_dict[inbetween_r] = [[], []]

    def improve_tip(self, iterations=6):
        r_w_so, r_wo_s = self.split_co_on_solutions()
        if len(r_wo_s.keys()) > 0:
            for it in range(iterations):
                self.calc_ir_before_closest_ir_wo_z()
                self.order_coordinates(plot_title=f"Improving tip iteration {it}",
                                       plot_inbetween=self.solver_params["plot_inbetween"])

    def improve(self):
        """
        Given an isoredshift calculated from just a couple coordinates, improves the solutions by:
        1. recalculating isoradials that did not contain the wanted redshift with more precision
        2. calculating isoradials inbetween the largest isoradial that had the wanted redshift and
        the closest that did not.
        """
        r_w_s, r_wo_s = self.split_co_on_solutions()
        if len(r_w_s):  # at least one solution is found
            self.recalc_isoradials_wo_redshift_solutions(plot_inbetween=False)
            self.improve_tip(iterations=self.solver_params["retry_tip"])
            for n in range(self.solver_params["times_inbetween"]):
                self.improve_between_all_solutions_once()
                self.order_coordinates(plot_title="calculating inbetween",
                                       plot_inbetween=self.solver_params["plot_inbetween"])

    def split_co_on_jump(self, threshold=2):
        """
        Returns the index where the difference in isoredshift coordinate values is significantly bigger than the median
        distance. This is used to avoid the plotter to connect two points that should not be connected.
        A jump between two coordinates co1 and co2 is when the isoredshift line does not connect within
        the considered frame, but either does not connect (purely radial isoredshift lines), or connects very far
        from the black hole
        """

        def dist(__x, __y):
            x1, x2 = __x
            y1, y2 = __y
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return d

        self.order_coordinates()
        self.__update()
        x, y = polar_to_cartesian_lists(self.radii, self.angles)
        _dist = [dist((x1, x2), (y1, y2)) for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:])]
        mx2, mx = sorted(_dist)[-2:]
        if mx > threshold * mx2:
            split_ind = np.where(_dist == mx)[0][0]
            if not abs(np.diff(np.sign(self.x[split_ind:split_ind + 2]))) > 0:
                # not really a jump, just an artefact of varying point density along the isoredshift line
                split_ind = None
        else:
            split_ind = None
        return split_ind

    def plot(self, norm, color_map):
        color = cm.ScalarMappable(norm=norm, cmap=color_map).to_rgba(self.redshift)
        plt.plot(self.y, [-e for e in self.x], color=color)  # TODO: hack to correctly orient plot
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


def polar_to_cartesian_single(th, radius, rotation=0):
    x = radius * np.cos(th + rotation)
    y = radius * np.sin(th + rotation)
    return x, y


def cartesian_to_polar(x, y):
    R = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    th = th if th > 0 else th + 2 * np.pi
    return th, R


def get_angle_around(p1, p2):
    """
    Calculates the angle of p2 around p1

    :param p1: coordinate 1 in format [x, y]
    :param p2:  coordinate 2 in format [x, y]
    :return: angle in radians
    """
    cx, cy = p1

    p2_ = np.subtract(p2, p1)
    angle_center, _ = cartesian_to_polar(cx, cy)
    # rotate p2_ counter-clockwise until the vector to the isoradial center is aligned with negative y-axis
    theta = np.pi - angle_center if angle_center > np.pi else angle_center
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    p2_ = np.dot(rot, p2_)
    angle_target, _ = cartesian_to_polar(p2[0], p2[1])
    angle_target_around_center, _ = cartesian_to_polar(p2_[0], p2_[1])

    return angle_target_around_center


if __name__ == '__main__':
    M = 1.
    incl = 76
    bh = BlackHole(inclination=incl, mass=M)
    bh.disk_outer_edge = 100 * M
    bh.iz_solver_params['radial_precision'] = 30
    bh.angular_properties["angular_precision"] = 200
    # bh.sample_points(n_points=10000)
    # bh.calc_isoradials([10, 20], [])
    # bh.plot_isoradials([10, 20], [10, 20], show=True)
    fig, ax = bh.plot_isoredshifts(redshifts=[-.2, -.15, -.1, -.05, 0., .05, .1, .15, .25, .5, .75],
                                   plot_core=True)
    fig.savefig(f"isoredshifts_{incl}_2.svg", facecolor="#F0EAD6")
