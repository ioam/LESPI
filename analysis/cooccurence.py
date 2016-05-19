import copy
import colorsys

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

import imagen


from holoviews.plotting.mpl import ElementPlot
from holoviews.ipython.widgets import ProgressBar
from holoviews import Histogram, HoloMap, QuadMesh, Store, DynamicMap

from featuremapper.analysis import *


class ChevronHistogram(QuadMesh):

    group = param.String(default='ChevronHistogram')


class CooccurenceHistogram(param.ParameterizedFunction):

    collapse_distance = param.Boolean(default=True)

    collapse_psis = param.Boolean(default=False)

    collapse_thetas = param.Boolean(default=False)

    distance_bins = param.List(default=[0, 1, 2, 3, 4])

    num_dtheta = param.Number(default=12, doc="""
        Number of orientation bins.""")

    num_phi = param.Number(default=12, doc="""
        Number of azimuth bins.""")

    num_samples = param.Number(default=100, doc="""
        Number of random non-zero samples to take (won't return
        desired number if max_attempts is exceeded.)""")

    threshold = param.Integer(default=70)


    def _initialize_bins(self, p):
        self.distance_bins = p.distance_bins
        self.phi_bins = np.linspace(-np.pi/2, np.pi/2, p.num_phi+1) + np.pi/p.num_phi/2
        self.theta_bins = np.linspace(-np.pi/2., np.pi/2., p.num_dtheta+1) + np.pi/p.num_dtheta/2

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


    def __call__(self, lateral_grid, or_pref, or_sel, xpref, ypref, **params):
        p = param.ParamOverrides(self, params)
        self._initialize_bins(p)

        self.progress_bar = ProgressBar()

        orsel = or_sel.last.data.ravel()

        v_hist = None
        num_projs = float(len(lateral_grid.keys()))

        distances = []; psis = []
        thetas = []; weights = []

        for idx, (coord, proj) in enumerate(lateral_grid.items()):
            lat_weights = proj.last.situated
            unit_theta = or_pref.last[coord]
            unit_sel = or_sel.last[coord]
            unit_pos = np.array([xpref.last[coord], ypref.last[coord]])

            threshold = np.percentile(lat_weights.data, p.threshold)
            weight_samples = lat_weights.data.ravel()
            mask = weight_samples>threshold
            weight_samples = weight_samples[mask]
            theta_samples = or_pref.last.data.ravel()[mask]
            xpref_samples = xpref.last.data.ravel()[mask]
            ypref_samples = ypref.last.data.ravel()[mask]
            orsel_samples = orsel[mask]

            dx = xpref_samples - unit_pos[0]
            dy = ypref_samples - unit_pos[1]
            d = np.sqrt(dx**2 + dy**2)

            phi = np.arctan2(-dx, dy)
            theta = theta_samples - unit_theta
            psi = phi - unit_theta
            psi -= theta/2.
            psi = ((psi + np.pi/2  - np.pi/p.num_phi/2.) % (np.pi)) - np.pi/2  + np.pi/p.num_phi/2.
            theta = ((theta + np.pi/2 - np.pi/p.num_dtheta/2.)  % (np.pi) ) - np.pi/2  + np.pi/p.num_dtheta/2.

            weights += [weight_samples*unit_sel*orsel_samples]
            psis += [psi]
            thetas += [theta]
            distances += [d]

            self.progress_bar((idx+1)/num_projs*100)

        weights = np.concatenate(weights)
        psis = np.concatenate(psis)
        thetas = np.concatenate(thetas)
        distances = np.concatenate(distances)

        bins, values = [], []
        if not p.collapse_distance:
            bins.append(self.distance_bins)
            values.append(distances)
        if not p.collapse_psis:
            bins.append(self.phi_bins)
            values.append(psis)
        if not p.collapse_thetas:
            bins.append(self.theta_bins)
            values.append(thetas)
        v_hist_, edges_ = np.histogramdd(values, bins=bins,
                                         normed=False, weights=weights)

        kdims=['$\Psi$', r'$\Delta\theta$']
        if p.collapse_distance:
            if p.collapse_phis:
                return Histogram((v_hist_, edges_[0]), kdims=kdims[1:])
            elif p.collapse_thetas:
                return Histogram((v_hist_, edges_[0]), kdims=kdims[:1])
            return ChevronHistogram(tuple(edges_)+(v_hist_.T,), kdims=kdims)
        else:
            stack = HoloMap(kdims=['Visual Angle'])
            for didx, dist in enumerate(edges_[0][:-1]):
                dist = (edges_[0][didx+1]+dist) / 2.
                stack[dist] = ChevronHistogram(tuple(edges_[1:])+(v_hist_[didx, :, :].T,), kdims=kdims)
            return stack



class ChevronPlot(ElementPlot):
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

    prior = param.Parameter(default=None)

    _stack_type = HoloMap

    def __init__(self, stack, radius=None, **kwargs):
        self.plots = []
        super(ChevronPlot, self).__init__(stack, **kwargs)

        edges_phi = self.hmap.last.data[0]
        edges_theta = self.hmap.last.data[1]

        self.num_phi = len(edges_phi) - 1
        self.num_dtheta = len(edges_theta) - 1
        self.shape = (128, 128)

        self.radius = np.ones((self.num_phi, self.num_dtheta)) if radius\
            is None else radius


    def initialize_plot(self, **ranges):
        ax = self.handles['axis']
        key = self.hmap.keys()[-1]

        # Unpack histogram object
        histogram = self.hmap.last
        edges = histogram.data[0:2]
        v_hist_angle = histogram.data[2].T.copy()

        if self.theta_symmetry:
            v_hist_angle[:, :-1] += v_hist_angle[:, :-1][:, ::-1]
            v_hist_angle[:, -1] *= 2
        if self.phi_symmetry:
            v_hist_angle[:-1, :] += v_hist_angle[:-1, :][::-1, :]
            v_hist_angle[-1, :] *= 2

        if self.prior is not None:
            # this allows to show conditional probability by dividing by an arbitrary (prior) distribution
            prior = self.prior.data[2].T
            if self.theta_symmetry:
                prior[:, :-1] += prior[:, :-1][:, ::-1]
                prior[:, -1] *= 2
            if self.phi_symmetry:
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
                                      rad, lw=self.line_width_chevrons/2, **colors)

                circles[(phi, theta)] = circ
                ax.add_patch(circ)

                # first edge
                angle_edgelist[0, i_phi * s_theta + i_theta] = self.shape[0] - rad_X * (s_theta - i_theta - .5) +  rad * 0.
                angle_edgelist[1, i_phi * s_theta + i_theta] = rad_Y * (i_phi + .5) - rad * 1.
                angle_edgelist[2, i_phi * s_theta + i_theta] = phi + theta/2
                angle_edgelist[3, i_phi * s_theta + i_theta] = self.sf_
                angle_edgelist[4, i_phi * s_theta + i_theta] = 1.
                # second edge
                angle_edgelist[0, i_phi * s_theta + i_theta + s_phi * s_theta] = self.shape[0] - rad_X * (s_theta - i_theta - .5) - rad * 0.
                angle_edgelist[1, i_phi * s_theta + i_theta + s_phi * s_theta] = rad_Y * (i_phi + .5) +  rad * 1.
                angle_edgelist[2, i_phi * s_theta + i_theta + s_phi * s_theta] = phi - theta/2
                angle_edgelist[3, i_phi * s_theta + i_theta + s_phi * s_theta] = self.sf_
                angle_edgelist[4, i_phi * s_theta + i_theta + s_phi * s_theta] = 1.

        self.handles['circles'] = circles

        ax = self.show_edges(ax, angle_edgelist)

        eps = 0.55 # HACK to center grid. dunnon what's happening here
        ax.set_xticks([(1./(self.num_phi+1)/2)*self.shape[0], .5*self.shape[0]+eps,
                            (1. - 1./(self.num_phi+1)/2)*self.shape[0]])
        ax.set_xticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'])

        ax.set_yticks([1./(self.num_dtheta+1)/2*self.shape[0], .5*self.shape[1]+eps,
                            (1. - 1./(self.num_dtheta+1)/2)*self.shape[1]])

        ax.set_yticklabels([r'$-\pi/2$', r'$0$', r'$\pi/2$'])

        self._set_labels(ax, histogram.kdims)

        # Apply title
        title = None if self.zorder > 0 else self._format_title(key)
        if self.show_title and title is not None:
            fontsize = self._fontsize('title')
            self.handles['title'] = ax.set_title(title, **fontsize)

        if ax is None: plt.close(self.handles['fig'])
        return ax if self.subplot else self.handles['fig']


    def update_frame(self, key, ranges=None, element=None):
        reused = isinstance(self.hmap, DynamicMap) and self.overlaid
        if not reused and element is None:
            element = self._get_frame(key)
        else:
            self.current_key = key
            self.current_frame = element

        if element is not None:
            self.set_param(**self.lookup_options(element, 'plot').options)
        axis = self.handles['axis']

        v_hist_angle = element.data[2].T


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
            self.handles['title'].set_text(self._format_title(key))


    def show_edges(self, ax, edges, v_min=-1., v_max=1.):
        """
        Shows the quiver plot of a set of edges.
        """

        ax.axis(c='b', lw=0)

        linewidth = self.line_width_chevrons
        scale = self.scale_chevrons

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
                circ = patches.Circle((x,y), self.scale_circle*scale/sf_0*n_, facecolor=fc, edgecolor='none')
                ax.add_patch(circ)

            line_segments = LineCollection(segments, linewidths=linewidths, colors=colors, linestyles='solid')
            ax.add_collection(line_segments)

        ax.axis([0, self.shape[0], self.shape[1], 0])

        return ax


Store.register({ChevronHistogram: ChevronPlot}, 'matplotlib')
