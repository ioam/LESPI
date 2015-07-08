import copy
import colorsys

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

import imagen

from holoviews.plotting.mpl import Plot
from holoviews.ipython.widgets import ProgressBar
from holoviews import Histogram

from featuremapper.analysis import *



class CooccurenceHistogram(param.ParameterizedFunction):

    collapse_distance = param.Boolean(default=True)

    distance_bins = param.List(default=[0, 1, 2, 3, 4])

    num_dtheta = param.Number(default=12, doc="""
        Number of orientation bins.""")

    num_phi = param.Number(default=12, doc="""
        Number of azimuth bins.""")

    num_samples = param.Number(default=100, doc="""
        Number of random non-zero samples to take (won't return
        desired number if max_attempts is exceeded.)""")

    max_attempts = param.Number(default=5000, doc="""
        Maximum number of non-zero attempts to make for each
        connection field.""")

    max_overlap = param.Number(default=0.5, doc="""
        Maximum Afferent Overlap.""")

    sampling_seed = param.Number(default=42)



    def _initialize_bins(self, p):
        self.distance_bins = p.distance_bins
        self.phi_bins = np.linspace(-np.pi/2, np.pi/2, p.num_phi+1)\
                        + np.pi/p.num_phi/2
        self.theta_bins = np.linspace(-np.pi/2, np.pi/2, p.num_dtheta+1)\
                          + np.pi/p.num_dtheta/2


    def _magnitude(self, x):
        return np.sqrt(np.dot(x, x))


    def _find_overlap(self, cf1, cf2):
        product = np.dot(cf1.flat, cf2.flat)
        magnitude_product = self._magnitude(cf1.flat) * self._magnitude(cf2.flat)
        return product/magnitude_product


    def _distance_phi(self, coord1, coord2):
        distance = self._magnitude(coord1-coord2)
        angle = np.arctan2(coord1[1] - coord2[1], coord1[0] - coord2[0])
        return distance, angle


    def __call__(self, lateral_grid, on_grid, off_grid, or_pref, or_sel, xpref, ypref, **params):
        p = param.ParamOverrides(self, params)
        self._initialize_bins(p)

        self.progress_bar = ProgressBar()

        v_hist = None
        num_projs = float(len(lateral_grid.keys()))

        for idx, proj in enumerate(lateral_grid.items()):
            coord, proj = proj
            lat_weights = proj.last
            unit_theta = or_pref.last[coord]
            unit_sel = or_sel.last[coord]
            unit_pos = np.array([xpref.last[coord], ypref.last[coord]])
            unit_on = on_grid[coord].last.situated.data
            unit_off = off_grid[coord].last.situated.data
            l, b, r, t = lat_weights.bounds.lbrt()
            xd, yd = lat_weights.xdensity, lat_weights.ydensity
            half_x_width = r-l/xd
            half_y_width = t-b/yd

            samples = 0
            attempts = 0
            distances = np.array(0); phis = np.array(0); thetas = np.array(0); weights = np.array(0)
            sampled_coords = []
            while samples < p.num_samples and attempts < 2000:
                attempts += 1
                sample_coord = (np.random.uniform(l, r-half_x_width), np.random.uniform(b+half_y_width, t))
                lat_weight = lat_weights[sample_coord]
                if (lat_weight == 0) or (sample_coord in sampled_coords):
                    continue
                sampled_coords.append(sample_coord)

                # Calculate overlap coefficient
                if p.max_overlap == 1.:
                    overlap_coeff = 0
                else:
                    sample_on = on_grid[sample_coord].last.situated.data
                    sample_off = off_grid[sample_coord].last.situated.data
                    cfs = [(unit_on, sample_on), (unit_off, sample_off)]
                    overlap_coeff = np.mean([self._find_overlap(cf1, cf2) for cf1, cf2 in cfs])
                if overlap_coeff < p.max_overlap:
                    samples += 1
                    sample_theta = or_pref.last[sample_coord]
                    sample_pos = np.array([xpref.last[sample_coord], ypref.last[sample_coord]])

                    X = np.array([unit_pos[0], sample_pos[0]])
                    Y = np.array([unit_pos[1], sample_pos[1]])
                    Theta = np.array([unit_theta, sample_theta])

                    dx = X[:, np.newaxis] - X[np.newaxis, :]
                    dy = Y[:, np.newaxis] - Y[np.newaxis, :]
                    d = np.sqrt(dx**2 + dy**2)

                    theta = Theta[:, np.newaxis] - Theta[np.newaxis, :]
                    phi = np.arctan(dy, dx) - np.pi/2 - Theta[np.newaxis, :]
                    phi -= theta/2

                    weight = lat_weight * or_sel.last[sample_coord] * unit_sel

                    weights = np.append(weights, [0, weight, weight, 0])
                    phis = np.append(phis, phi.ravel())
                    thetas = np.append(thetas, theta.ravel())
                    distances = np.append(distances, d.ravel())

            phis = ((phis + np.pi/2 - np.pi/self.num_phi/2.) %
                    np.pi) - np.pi/2 + np.pi/self.num_phi/2.
            thetas = ((thetas + np.pi/2 - np.pi/self.num_dtheta/2.) %
                      np.pi) - np.pi/2 + np.pi/self.num_dtheta/2.


            v_hist_, edges_ = np.histogramdd([distances, phis, thetas],
                                             bins=(self.distance_bins,
                                                   self.phi_bins,
                                                   self.theta_bins),
                                                   normed=False, weights=weights)

            if not(v_hist_.sum() == 0.):
                # add to the full histogram
                if v_hist==None:
                    v_hist = v_hist_*1.
                else:
                    v_hist += v_hist_*1.

            self.progress_bar.update((idx+1)/num_projs*100)

        labels = ['Phi', 'dTheta']
        edges = edges_[1:]

        if p.collapse_distance:
            v_hist = v_hist.sum(axis=0)
            v_hist /= v_hist.sum()
            return Histogram(v_hist, edges, labels=labels)
        else:
            dim_info = {'Dimension': {'type': float, 'unit': 'Visual Angle'}}
            stack = ViewMap(dimension_labels()=['Distance'], dim_info=dim_info,
                              title="Co-occurence: {label0} = {value0}")
            for didx, dist in enumerate(edges_[0]):
                if didx == len(self.distance_bins)-1:
                    continue
                dist = (edges_[0][didx+1]+dist) / 2.
                hist = v_hist[didx, :, :] / v_hist[didx, :, :].sum() # Normalize individually
                stack[dist] = Histogram(hist, edges, labels=labels)
            return stack



class ChevronPlot(Plot):
    """
    Chevron plot is a specialized plot displaying co-occurence statistics
    """

    line_width_chevrons = param.Number(default=1.5, doc="""
        Chevron line width.""")

    edge_scale_chevrons = param.Number(default=64.)

    scale_circle = param.Number(default=0.08) # relativesize of segments and pivot

    scale_chevrons = param.Number(default=1.5)

    xticks= param.Boolean(default=True, doc="""
        Whether to enable xticks.""")

    sf_ = param.Number(default=100.)

    size = param.NumericTuple(default=(10, 10))

    phi_symmetry = param.Boolean(default=True)

    theta_symmetry = param.Boolean(default=True)

    _stack_type = ViewMap

    def __init__(self, stack, prior=None, radius=None, **kwargs):
        self._stack = self._check_stack(stack, Histogram)
        self.prior = prior
        self.plots = []
        super(ChevronPlot, self).__init__(**kwargs)

        edges = self._stack.last.edges[:]
        edges_phi = edges[0]
        edges_theta = edges[1]

        self.num_phi = len(edges_phi) - 1
        self.num_dtheta = len(edges_theta) - 1

        self.radius = np.ones((self.num_phi, self.num_dtheta)) if radius\
            is None else radius


    def __call__(self, axis=None, **params):
        p = ParamOverrides(self, params)
        title = self._format_title(self._stack, -1)

        self.shape = (128, 128)
        lbrt = (0, 0, 128, 128)
        ax = self._axis(axis, title, r'azimuth difference $\psi$',
                        r'orientation difference $\theta$', lbrt=lbrt)

        # Unpack histogram object
        histogram = self._stack.last
        v_hist_angle, edges = (histogram.values.copy(), histogram.edges[:])

        if p.theta_symmetry:
            v_hist_angle[:, :-1] += v_hist_angle[:, :-1][:, ::-1]
            v_hist_angle[:, -1] *= 2
        if p.phi_symmetry:
            v_hist_angle[:-1, :] += v_hist_angle[:-1, :][::-1, :]
            v_hist_angle[-1, :] *= 2

        if self.prior is not None:
            # this allows to show conditional probability by dividing by an arbitrary (prior) distribution
            prior = self.prior.copy()
            if p.theta_symmetry:
                prior[:, :-1] += prior[:, :-1][:, ::-1]
                prior[:, -1] *= 2
            if p.phi_symmetry:
                prior[:-1, :] += prior[:-1, :][::-1, :]
                prior[-1, :] *= 2
            prior /= prior.sum()
            v_hist_angle /= prior

        edges_phi = edges[0]
        edges_theta = edges[1]

        #HACK: force the chevron plot to be symmetric


        # Center around mean and normalize histogram
        v_hist_angle -= v_hist_angle.mean()
        v_max = np.absolute(v_hist_angle).max()
        v_hist_angle = np.divide(v_hist_angle, v_max)

        # Calculate Phi and Theta values
        v_phi, v_theta = edges_phi - np.pi / self.num_phi / 2, edges_theta - np.pi / self.num_dtheta / 2
        self.vtheta = v_theta
        self.vphi = v_phi
        self.i_phi_shift, self.i_theta_shift = -1, -1

        # Calculate chevron positions
        s_phi, s_theta = len(v_phi), len(v_theta)
        rad_X, rad_Y = 1. * self.shape[0] / s_theta, 1. * self.shape[1] / s_phi
        rad = min(rad_X, rad_Y) / 4.

        angle_edgelist = np.zeros((5, s_phi * s_theta * 2 ))

        ax.axis(c='b', lw=0)
        circles = {}
        for i_phi, phi in enumerate(v_phi):
            for i_theta, theta in enumerate(v_theta):
                value = v_hist_angle[(i_phi + self.i_phi_shift) % self.num_phi, (i_theta + self.i_theta_shift) % self.num_dtheta]
                score = self.radius[(i_phi + self.i_phi_shift) % self.num_phi, (i_theta + self.i_theta_shift) % self.num_dtheta]
                fc = ((np.sign(value)+1)/2, 0, (1-np.sign(value))/2, np.abs(value)**.5)
                if score:
                    colors = {'facecolor': fc, 'edgecolor': fc}
                else:
                    colors = {'facecolor': 'w', 'edgecolor': 'k', 'alpha' : .5}
                circ = patches.Circle((rad_Y * (i_phi + .5) + .5, self.shape[0] - rad_X * (s_theta - i_theta - .5) + .5),
                                      rad, lw=p.line_width_chevrons/2, **colors)

                circles[(phi, theta)] = circ
                ax.add_patch(circ)

                # first edge
                angle_edgelist[0, i_phi * s_theta + i_theta] = self.shape[0] - rad_X * (s_theta - i_theta - .5) +  rad * 0.
                angle_edgelist[1, i_phi * s_theta + i_theta] = rad_Y * (i_phi + .5) - rad * 1.
                angle_edgelist[2, i_phi * s_theta + i_theta] = phi + theta/2
                angle_edgelist[3, i_phi * s_theta + i_theta] = p.sf_
                angle_edgelist[4, i_phi * s_theta + i_theta] = 1.
                # second edge
                angle_edgelist[0, i_phi * s_theta + i_theta + s_phi * s_theta] = self.shape[0] - rad_X * (s_theta - i_theta - .5) - rad * 0.
                angle_edgelist[1, i_phi * s_theta + i_theta + s_phi * s_theta] = rad_Y * (i_phi + .5) +  rad * 1.
                angle_edgelist[2, i_phi * s_theta + i_theta + s_phi * s_theta] = phi - theta/2
                angle_edgelist[3, i_phi * s_theta + i_theta + s_phi * s_theta] = p.sf_
                angle_edgelist[4, i_phi * s_theta + i_theta + s_phi * s_theta] = 1.

        self.handles['circles'] = circles

        ax = self.show_edges(p, ax, angle_edgelist)

        eps = 0.55 # HACK to center grid. dunnon what's happening here
        ax.set_xticks([(1./(self.num_phi+1)/2)*self.shape[0], .5*self.shape[0]+eps,
                            (1. - 1./(self.num_phi+1)/2)*self.shape[0]])
        ax.set_xticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'], fontsize=self.font_size)

        ax.set_yticks([1./(self.num_dtheta+1)/2*self.shape[0], .5*self.shape[1]+eps,
                            (1. - 1./(self.num_dtheta+1)/2)*self.shape[1]])

        ax.set_yticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'], fontsize=self.font_size)

        plt.draw()

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n):
        n = n if n < len(self) else len(self) - 1
        histogram = self._stack.values()[n]

        v_hist_angle, _ = histogram.data

        if self.theta_symmetry:
            v_hist_angle[:, :-1] += np.fliplr(v_hist_angle[:, :-1])
            v_hist_angle[:, -1] *= 2
        if self.phi_symmetry:
            v_hist_angle[:-1, :] += np.flipud(v_hist_angle[:-1, :])
            v_hist_angle[-1, :] *= 2

        v_hist_angle -= v_hist_angle.mean()
        v_max = np.absolute(v_hist_angle).max()
        v_hist_angle /= v_max

        circles = self.handles['circles']
        for i_phi, phi in enumerate(self.vphi):
            for i_theta, theta in enumerate(self.vtheta):
                value = v_hist_angle[(i_phi + self.i_phi_shift) % self.num_phi, (i_theta + self.i_theta_shift) % self.num_dtheta]
                fc = ((np.sign(value)+1)/2, 0, (1-np.sign(value))/2, np.abs(value)**.5)
                circles[(phi, theta)].set_facecolor(fc)
                circles[(phi, theta)].set_edgecolor(fc)

        if self.show_title:
            self.handles['title'].set_text(self._format_title(self._stack, n))

        plt.draw()

    def __len__(self):
        return len(self._stack)


    def show_edges(self, p, ax, edges, v_min=-1., v_max=1.):
        """
        Shows the quiver plot of a set of edges.
        """

        ax.axis(c='b', lw=0)

        linewidth = p.line_width_chevrons
        scale = p.scale_chevrons

        opts= {'extent': (0, self.shape[0], self.shape[1], 0),
               'cmap': cm.gray,
               'vmin':v_min, 'vmax':v_max,
               'interpolation':'nearest',
               'origin':'upper'}

        ax.imshow([[v_max]], **opts)
        if edges.shape[1] > 0:
            # draw the segments
            segments, colors, linewidths = list(), list(), list()

            X, Y, Theta, Sf_0 = edges[1, :].real+.5, edges[0, :].real+.5, np.pi -  edges[2, :].real, edges[3, :].real
            weights = edges[4, :]

            weights = weights/(np.abs(weights)).max()

            for x, y, theta, sf_0, weight in zip(X, Y, Theta, Sf_0, weights):
                u_, v_ = np.cos(theta)*scale/sf_0*self.shape[0], np.sin(theta)*scale/sf_0*self.shape[1]
                segment = [(x - u_, y - v_), (x + u_, y + v_)]
                segments.append(segment)
                colors.append((0, 0, 0, 1))# black
                linewidths.append(linewidth) # *weight thinning byalpha...

            # TODO : put circle in front
            n_ = np.sqrt(self.shape[0]**2+self.shape[1]**2)
            for x, y, theta, sf_0, weight in zip(X, Y, Theta, Sf_0, weights):
                fc = (0, 0, 0, 1)# black
                # http://matplotlib.sourceforge.net/users/transforms_tutorial.html
                circ = patches.Circle((x,y), p.scale_circle*scale/sf_0*n_, facecolor=fc, edgecolor='none')
                ax.add_patch(circ)

            line_segments = LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
            ax.add_collection(line_segments)

        ax.axis([0, self.shape[0], self.shape[1], 0])
        plt.draw()

        return ax
