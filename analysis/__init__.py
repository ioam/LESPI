from collections import defaultdict
from itertools import product
from functools import partial

import numpy as np
import pandas
import scipy
from scipy.optimize import curve_fit
import scipy.special as ss
from scipy.ndimage import gaussian_filter

import param
from param import ParameterizedFunction, ParamOverrides

import holoviews as hv
from holoviews import HoloMap, Dimension, Image, Table, GridSpace, ItemTable
from holoviews.core.options import Store, Options
from holoviews.core.ndmapping import sorted_context
from holoviews.ipython.widgets import ProgressBar
from holoviews.interface.collector import Layout
from holoviews.interface.seaborn import DFrame
from holoviews.operation import MapOperation, ElementOperation, transform
from holoviews.operation.normalization import raster_normalization
from holoviews.ipython.widgets import ProgressBar

import imagen as ig
from imagen import Composite, RawRectangle
from imagen.transferfn import DivisiveNormalizeL1

try:
    from numba import jit
except ImportError:
    def jit(func):
        return func

from featuremapper.analysis import cyclic_difference, center_cyclic
from featuremapper.analysis.spatialtuning import SizeTuningPeaks, SizeTuningShift,\
    OrientationContrastAnalysis, FrequencyTuningAnalysis
from featuremapper.command import FeatureCurveCommand, DistributionStatisticFn, \
    measure_size_response, measure_orientation_contrast, DSF_MaxValue, \
    UnitCurveCommand, MeasureResponseCommand, measure_frequency_response, measure_response
import featuremapper.features as f

import topo


@jit
def lhi_inner(or_map, sigma, sx, sy):
    lhi1, lhi2 = 0., 0.
    (xsize,ysize) = or_map.shape
    for tx in xrange(0,xsize):
        for ty in xrange(0,ysize):
            lhi1+=np.exp(-((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty))/(2*sigma**2))*np.cos(2*or_map[tx,ty])
            lhi2+=np.exp(-((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty))/(2*sigma**2))*np.sin(2*or_map[tx,ty])
    return lhi1, lhi2


def similarity_analysis(hmap):
    return transform(cyclic_difference(raster_normalization(hmap*hmap.last)),
                     operator=lambda x: 1.0-(2.0*x))


# Filter out units with OR prefs significantly different from the measured location
def filter_unitorpref(df, orpref_stack, max_diff=np.pi/8):
    rows = []
    columns = list(df.columns)
    x_ix = columns.index('X')
    y_ix = columns.index('Y')
    or_ix = columns.index('Orientation_Preference')
    for row in df.values:
        x, y, orpref = row[x_ix], row[y_ix], row[or_ix]
        ref_orpref = orpref_stack.last[x, y]
        ordiff = abs(orpref - ref_orpref)
        ordiff = ordiff if orpref % np.pi <= np.pi else ordiff % np.pi
        if ordiff < max_diff:
            rows.append(row)
    return pandas.DataFrame(rows, None, df.columns)


class LocalHomogeneityIndex(ElementOperation):

    sigma = param.Number(default=0.07)

    def lhi(self, or_map, sigma):
        (xsize,ysize) = or_map.shape
        lhi = np.zeros(or_map.shape)
        for sx in xrange(0,xsize):
            for sy in xrange(0,ysize):
                lhi_current = lhi_inner(or_map, sigma, sx, sy)
                lhi[sx,sy]=np.sqrt(lhi_current[0]**2 + lhi_current[1]**2)/(2*np.pi*sigma**2)
        return lhi

    def _process(self, ormap, key=None):
        or_map = ormap.data
        sigma = int(self.p.sigma * ormap.xdensity)
        lhi = self.lhi(or_map, sigma)
        return ormap.clone(lhi.copy(), group='Local Homogeneity Index', vdims=['LHI'])


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

class TuningWidth(ElementOperation):

    def _process(self, curves, key=None):
        centered = {k: center_cyclic(v) for k, v in curves.items()}
        xs = centered.values()[0].dimension_values(0)
        p0 = [1., 0., 1.]
        dims = ['Amplitude', '$\mu$', '$\sigma$', 'Bandwidth']
        fits, fitted_curves = [], []
        for k, curve in centered.items():
            ys = curve.dimension_values(1)
            coeff, var_matrix = curve_fit(gauss, xs, ys, p0=p0, maxfev=10000)
            bw = np.rad2deg(coeff[-1]*2*np.sqrt(2*np.log(2)))/2
            fits.append((k, ItemTable(zip(dims, list(coeff)+[bw]))))
            fitted_curves.append((k, curve.clone((xs, gauss(xs, *coeff)))))
        return curves.clone(fitted_curves)() + curves.clone(fits).table()


class UnitMeasurements(MeasureResponseCommand):

    def _compute_roi(self, p, output):
        # Compute densities, positions and ROI region
        xd, yd = topo.sim[output].xdensity, topo.sim[output].ydensity
        coords = [topo.sim[output].closest_cell_center(*coord) for coord in p.coords]
        lo, bo, ro, to = p.relative_roi
        cols, rows = (ro - lo) * xd, (to - bo) * yd
        return coords, (lo, bo, ro, to), cols, rows


class measure_size_tuning(UnitMeasurements):

    coords = param.List(default=[(0, 0)])

    contrasts = param.List(default=[50, 100])

    frequency = param.Number(default=1.65)

    max_size = param.Number(default=3.0)

    max_ordiff = param.Number(default=np.pi/8)

    num_sizes = param.Integer(default=31)

    num_phase = param.Integer(default=8)

    outputs = param.List(default=['V1'])

    relative_roi = param.NumericTuple(default=(-0.25, -0.25, 0.25, 0.25))

    def __call__(self, orpref, **params):
        p = param.ParamOverrides(self, params)

        ordim = orpref.last.value_dimensions[0].name
        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.outputs[0])
        lo, bo, ro, to = lbrt_offsets

        size_dataframes = defaultdict(list)
        css_dataframes = defaultdict(list)

        measurement_params = dict(frequencies=[p.frequency], max_size=p.max_size,
                                  num_sizes=p.num_sizes, num_phase=p.num_phase,
                                  outputs=p.outputs, contrasts=p.contrasts)
        results = Layout()
        for coord in coords:
            data = measure_size_response(coords=[coord], **measurement_params)
            # Compute relative offsets from the measured coordinate
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)

            for output in p.outputs:
                size_grid = data.SizeTuning[output].sample((cols, rows), bounds=lbrt).to.curve('Size', 'Response')
                size_data = SizeTuningPeaks(size_grid)
                size_df = size_data.dframe()

                # Filter out values with significantly different OR preferences
                ref_orpref = orpref.last[coord]
                ortables = orpref.sample((rows, cols), bounds=lbrt)
                or_df = ortables[topo.sim.time(), :].dframe()
                size_df = pandas.merge(size_df, or_df, on=['x', 'y', 'Time', 'Duration'])
                filter_condition = (np.abs(size_df[ordim] - ref_orpref) % np.pi) < p.max_ordiff
                size_df = size_df[filter_condition]
                size_dataframes[output].append(size_df)

                # If multiple contrasts have been measured find the contrast dependent size tuning shift
                if len(p.contrasts) >= 2:
                    contrast_grid = size_grid.overlay(['Contrast']).grid(['x', 'y'])
                    css_grid = SizeTuningShift(contrast_grid)
                    css_df = css_grid.dframe()
                    css_df = pandas.merge(css_df, or_df, on=['x', 'y', 'Time', 'Duration'])
                    filter_condition = (np.abs(css_df[ordim] - ref_orpref) % np.pi) < p.max_ordiff
                    css_df = css_df[filter_condition]
                    css_dataframes[output].append(css_df)

                # Add Size Response to results
                path = ('SizeResponse', output)
                size_response = data.SizeTuning[output]
                if path in results:
                    results.SizeResponse[output].update(size_response)
                else:
                    results.set_path(path, size_response)

            # Stack the Size Tuning Data
            for output in p.outputs:
                size_df = pandas.concat(size_dataframes[output])
                size_stack = HoloMap(None, key_dimensions=['Time', 'Contrast'])
                for k, cdf in size_df.groupby(['Contrast']):
                    cdf = cdf.filter(['Peak Size', 'SI', 'Suppression Size', 'CS Size', 'CSI'])
                    size_stack[(topo.sim.time(), k)] = DFrame(cdf, group='Size Tuning Analysis')
                    results.set_path(('SizeTuning', output), size_stack)

                if css_dataframes:
                    css_df = pandas.concat(css_dataframes[output])
                    css_df = css_df.filter(['CSS', 'Low', 'High'])
                    contrast_stack = HoloMap((topo.sim.time(), DFrame(css_df, group='Contrast Size Tuning Shift')),
                                             key_dimensions=['Time'])
                    results.set_path(('ContrastShift', output), contrast_stack)

        return results


class measure_frequency_tuning(UnitMeasurements):

    coords = param.List(default=[(0, 0)])

    contrasts = param.List(default=[50, 100])

    max_freq = param.Number(default=5.0)

    max_ordiff = param.Number(default=np.pi/8)

    size = param.Number(default=1.0)

    num_freq = param.Integer(default=31)

    num_phase = param.Integer(default=8)

    output = param.String(default='V1')

    relative_roi = param.NumericTuple(default=(-0.15, -0.15, 0.15, 0.15))

    def __call__(self, orpref, **params):
        p = param.ParamOverrides(self, params)

        ordim = orpref.last.value_dimensions[0].name
        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.output)
        lo, bo, ro, to = lbrt_offsets

        size_dataframes = []

        measurement_params = dict(max_frequencies=p.max_freq, size=p.size,
                                  num_freq=p.num_freq, num_phase=p.num_phase,
                                  outputs=[p.output], contrasts=p.contrasts)
        results = Layout()
        for coord in coords:
            data = measure_frequency_response(coords=[coord], **measurement_params)

            # Compute relative offsets from the measured coordinate
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)

            freq_grid = data.FrequencyTuning[p.output].sample((cols, rows), bounds=lbrt).to.curve('Frequency', 'Response')
            freq_data = FrequencyTuningAnalysis(freq_grid)
            freq_df = freq_data.dframe()

            # Filter out values with significantly different OR preferences
            ref_orpref = orpref.last[coord]
            ortables = orpref.sample((rows, cols), bounds=lbrt)
            or_df = ortables[topo.sim.time(), :].dframe()
            freq_df = pandas.merge(freq_df, or_df, on=['x', 'y', 'Time', 'Duration'])
            filter_condition = (np.abs(freq_df[ordim] - ref_orpref) % np.pi) < p.max_ordiff
            freq_df = freq_df[filter_condition]
            size_dataframes.append(freq_df)

            # Add Size Response to results
            path = ('FrequencyResponse', p.output)
            freq_response = data.FrequencyTuning[p.output]
            if path in results:
                results.FrequencyResponse[p.output].update(freq_response)
            else:
                results.set_path(path, freq_response)

        # Stack the Frequency Tuning Data
        freq_df = pandas.concat(size_dataframes)
        size_stack = HoloMap(None, key_dimensions=['Time', 'Contrast'])
        for k, cdf in freq_df.groupby(['Contrast']):
            cdf = cdf.filter(['Peak', 'Bandwidth', 'QFactor', 'Lower', 'Upper'])
            size_stack[(topo.sim.time(), k)] = DFrame(cdf, label='Frequency Tuning Analysis')
        results.set_path(('FrequencyTuning', p.output), size_stack)

        return results


class measure_iso_suppression(UnitMeasurements):

    coords = param.List(default=[(0, 0)])

    sizecenter = param.Number(default=1.0, bounds=(0, None), doc="""
        The size of the central pattern to present.""")

    sizesurround = param.Number(default=3.5, bounds=(0, None), doc="""
        The size of the surround pattern to present.""")

    thickness = param.Number(default=2.0, bounds=(0, None),
                             doc="""Ring thickness.""")

    durations = param.List(default=[1],
                           doc="Contrast of the surround.")

    contrastsurround = param.List(default=[0, 30, 100],
                                  doc="Contrast of the surround.")

    contrastcenter = param.List(default=[10, 70],
                                doc="""Contrast of the center.""")

    output = param.String(default='V1')

    frequency = param.Number(default=1.65)

    num_orientation = param.Integer(default=10)

    max_ordiff = param.Number(default=np.pi/8)

    relative_roi = param.NumericTuple(default=(-0.25, -0.25, 0.25, 0.25))

    def __call__(self, orpref, **params):
        p = param.ParamOverrides(self, params)

        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.output)
        lo, bo, ro, to = lbrt_offsets

        orcs_dataframes = []

        measurement_params = dict(frequencies=[p.frequency], outputs=[p.output],
                                  contrastsurround=p.contrastsurround,
                                  thickness=p.thickness,
                                  sizecenter=p.sizecenter,
                                  sizesurround=p.sizesurround,
                                  durations=p.durations,
                                  num_orientation=p.num_orientation)
        center_dim = Dimension('ContrastCenter')

        results = Layout()
        for coord, c in product(coords, p.contrastcenter):
            measurement_params['contrastcenter'] = c
            data = measure_orientation_contrast(coords=[coord], **measurement_params)

            # Compute relative offsets from the measured coordinate
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)

            orcs = data.OrientationsurroundTuning[p.output].sample((cols, rows), bounds=lbrt).to.curve('OrientationSurround', 'Response')
            orcs_data = OrientationContrastAnalysis(orcs)
            orcs_df = orcs_data.dframe()
            orcs_df['ContrastCenter'] = c

            # Filter out values with significantly different OR preferences
            ref_orpref = orpref.last[coord]
            ortables = orpref.sample((rows, cols), bounds=lbrt)
            or_df = ortables[topo.sim.time(), :].dframe()
            orcs_df = pandas.merge(orcs_df, or_df, on=['x', 'y', 'Time', 'Duration'])
            filter_condition = (np.abs(orcs_df[orpref.last.value_dimensions[0].name] - ref_orpref) % np.pi) < p.max_ordiff
            orcs_df = orcs_df[filter_condition]
            orcs_dataframes.append(orcs_df)

            # Add Size Response to results
            path = ('OrientationContrastResponse', p.output)
            orcs_response = data.OrientationsurroundTuning[p.output]
            orcs_response = orcs_response.add_dimension(center_dim, 0, c)
            if path in results:
                results.OrientationContrastResponse[p.output].update(orcs_response)
            else:
                results.set_path(path, orcs_response)

        # Stack the Size Tuning Data
        orcs_df = pandas.concat(orcs_dataframes)
        orcs_table = Table(orcs_df, vdims=['OCSI', 'Orientation Preference'],
                           label='Orientation Contrast Analysis')
        orcs_stack = HoloMap({topo.sim.time(): orcs_table}, key_dimensions=['Time'])
        results.set_path(('OCSI_Analysis', p.output), orcs_stack)

        return results


def circular_dist(a, b, period):
    """
    Returns the distance between a and b (scalars) in a domain with `period` period.
    """
    return np.minimum(np.abs(a - b), period - np.abs(a - b))


class measure_phase_tuning(FeatureCurveCommand):
    """
    Measure phase tuning curves for orientation selective units.
    Presents full-field sine gratings to a sheet and uses the
    pre-measured orientation preference to generate phase tuning
    curves for each unit at it's preferred orientation.
    """

    subplot = param.String("Complexity")

    preference_fn = param.ClassSelector(DistributionStatisticFn,
                                        default=DSF_MaxValue(), doc="""
        Function that will be used to analyze the distributions of unit
        responses.""")

    x_axis = param.String(default='phase')

    def __call__(self, orpref=None, **params):
        results = super(measure_phase_tuning, self).__call__(**params)
        p = ParamOverrides(self, params, allow_extra_keywords=True)

        for output in p.outputs:
            self._reduce_orientations(p, output, results, orpref)

        return results

    def _reduce_orientations(self, p, output, results, orpref):
        stack = results.PhaseTuning[output]

        # Compute bounds and ROIs
        rows, cols = stack.last.data.shape

        if isinstance(orpref, HoloMap):
            orpref = orpref.last

        # Process dimension values and reduced dimensions
        orvalues = sorted(set(stack.dimension_values('Orientation')))
        phasevalues = sorted(set(stack.dimension_values('Phase')))
        reduced_dimensions = [d for d in stack.key_dimensions if d.name != 'Orientation']

        # Initialize progress bar
        phase_progress = ProgressBar(label='Phase Collapse')
        phase_progress(0)

        # Reduce the original stack by finding the response to the
        # optimally orientated stimulus for each unit for each phase
        reduced_stack = stack.clone(shared_data=False, key_dimensions=reduced_dimensions)
        for idx, phase in enumerate(phasevalues):
            reduced_response = np.zeros((rows, cols))
            phase_stack = stack.select(Phase=phase)
            keys = [tuple(v for d, v in zip(stack.dimensions('key', label=True), k) if d != 'Orientation')
                    for k in phase_stack.keys()]
            for key in set(keys):
                key_slice = list(key)
                key_slice.insert(stack.get_dimension_index('Orientation'), slice(None))
                sliced_stack = stack[tuple(key_slice)]
                for x in xrange(cols):
                    for y in xrange(rows):
                        # Find the closest match between the units preferred
                        # orientation and the measured orientations
                        unit_orpref = orpref.data[x, y]
                        or_dists = [circular_dist(v, unit_orpref, np.pi) for v in orvalues]
                        closest_orpref = orvalues[np.argmin(or_dists)]

                        # Sample the units response at the optimal orientation
                        sv = sliced_stack.select(Orientation=closest_orpref).last
                        reduced_response[x, y] = sv.data[x, y]
                reduced_stack[tuple(key)] = phase_stack.last.clone(reduced_response)
            phase_progress((float(idx+1)/p.num_phase)*100)

        results[('PhaseTuning', output)] = reduced_stack
        return results


    def _feature_list(self, p):
        return [f.Orientation(steps=p.num_orientation, preference_fn=None),
                f.Phase(steps=p.num_phase),
                f.Frequency(values=p.frequencies)]



class RFGaborFit(param.ParameterizedFunction):
    """
    RFGaborFit takes a grid of measured RFs as input and attempts
    to fit a 2D Gabor function to each RF. It returns GridSpaces of
    the Gabor fit, the fit parameters and residual from the fit.
    """

    max_iterations = param.Integer(default=1000)

    roi_radius = param.Number(default=None, allow_None=True)

    init_fit = param.NumericTuple(default=(1.0, 1.7, 0.1, 0.2, 0))

    def _validate_rf(self, rf):
        if not isinstance(rf, Image):
            raise Exception('Supplied views need to be curves.')

    @classmethod
    def _function(self, (xs, ys, x0, y0), A=1.0, f=1.7, sig_x=0.1, sig_y=0.2, phase=0, theta=0):
        if any(v < 0 or v > 10 for v in [f, sig_x, sig_y]):
            return 10000000000000000000
        try:
            theta -= np.pi/2.
            theta = -theta
            y = np.subtract.outer(np.cos(theta)*ys, np.sin(theta)*xs)
            x = np.add.outer(np.sin(theta)*ys, np.cos(theta)*xs)
            x_w = np.divide(x, sig_x)
            y_h = np.divide(y, sig_y)
            p = np.exp(-0.5*x_w*x_w + -0.5*y_h*y_h)
            result = (A * p * np.cos(2*np.pi*f*x + phase))
            result = result.ravel()
        except RuntimeError:
            result = np.ones(x.shape).ravel() * 10e6
        except FloatingPointError:
            result = np.ones(x.shape).ravel() * 10e6
        except AttributeError:
            print x, y
        return result

    def __call__(self, grid, orpreference, **params):
        self.p = param.ParamOverrides(self, params, allow_extra_keywords=True)
        results = Layout()
        normed_grid = {}
        fit_grid = {}
        residual_grid = {}
        fitvals_grid = {}

        progress = ProgressBar()
        progress(0)

        grid_items = grid.items()
        grid_length = len(grid_items)
        for idx, ((x, y), sheet_stack) in enumerate(grid_items):
            for key, view in sheet_stack.items():
                key_dict = dict(zip([k.name for k in grid.kdims], (x, y)))
                key_dict.update(dict(zip([k.name for k in sheet_stack.kdims], key)))
                key_dict.pop('Duration', None)
                theta = orpreference.select(**key_dict).last[x, y]
                processed = self._process(view, theta, key_dict)
                if processed is None: continue
                if (x, y) not in fit_grid:
                    normed_grid[(x, y)] = sheet_stack.clone({key: processed[0]})
                    fit_grid[(x, y)] = sheet_stack.clone({key: processed[1]})
                    residual_grid[(x, y)] = sheet_stack.clone({key: processed[2]})
                    fitvals_grid[(x, y)] = HoloMap({key: processed[3]}, kdims=sheet_stack.kdims)
                else:
                    normed_grid[(x, y)][key] = processed[0]
                    fit_grid[(x, y)][key] = processed[1]
                    residual_grid[(x, y)][key] = processed[2]
                    fitvals_grid[(x, y)][key] = processed[3]
            progress(((idx+1)/float(grid_length))*100)
        results.set_path(('RFGaborFit', 'RF_Fit'), grid.clone(fit_grid))
        results.set_path(('RFGaborFit', 'RF_Normed'), grid.clone(normed_grid))
        results.set_path(('RFGaborFit', 'RF_Residuals'), grid.clone(residual_grid))
        results.set_path(('RFGaborFit', 'RF_Fit_Values'), GridSpace(fitvals_grid))
        return results

    def _process(self, rf, theta, key=None):
        if hasattr(rf, 'situated'):
            rf = rf.situated
        self._validate_rf(rf)

        x, y = key['X'], key['Y']
        l, b, r, t = rf.lbrt
        if self.p.roi_radius:
            off = self.p.roi_radius
            rf = rf[x-off:x+off, y-off:y+off]
            l, b, r, t = -off, -off, off, off
            x, y = 0, 0
        
        normed = gaussian_filter(rf.data, 1)
        data = normed - normed.mean()

        rows, cols = data.shape
        xs, ys = np.linspace(l, r, cols), np.linspace(b, t, rows)
        try:
            fit, pcov = curve_fit(self._function, (xs, ys, x, y), data.ravel(),
                                  self.p.init_fit+(theta,), maxfev=self.max_iterations)
        except RuntimeError:
            return None
        fit_data = self._function((xs, ys, x, y), *fit).reshape(rows, cols)
        nx, ny= (fit[1] * fit[2], fit[1] * fit[3])
        residual = data - fit_data
        fit_table = Table(tuple(fit)+(theta, nx, ny, np.sum(np.abs(residual))),
                          vdims=['A', 'f', 'sig_x', 'sig_y', 'phase', 'theta',
                                 'real_theta', 'nx', 'ny', 'residual'])

        return [rf.clone(data), rf.clone(fit_data),
                rf.clone(residual, label='Residual'), fit_table]


def vonMises_fn(phi, k, mu):
    """
    The vonMises function generates a weighting for different
    cyclic quantities phi from a vonMises distribution centered
    around mu with a kernel width k.
    """
    return (1/(2*np.pi*ss.i0(k))) * np.e**(k*np.cos(2*(phi-mu)))


class CFvonMisesFit(param.ParameterizedFunction):
    """
    CFvonMisesFit takes a grid of lateral CFs and an orientation map as input
    and fits a vonMises function combined with a Gaussian function to both.
    Replicates Buzas 2006 model of patchy lateral excitatory connectivity in V1.
    """

    max_iterations = param.Integer(default=1000)

    projection = param.String(default='LateralExcitatory')

    threshold = param.Number(default=70, doc="""The threshold below which
        weights are ignored, expressed as a percentile""")

    lateral_size = param.Number(default=2.5, doc="""
        The size of the lateral CFs in sheet coordinates""")

    fit_aspect = param.Boolean(default=False, doc="""
        Whether to allow fitting the aspect ratio of the Gaussian""")

    sheet = param.String(default='V1Exc')


    def _function(self, orpref, k, mu1, mu2, weight=0, x=0, y=0, aspect=1,
                  ravelled=True, vonMises=True):
        l, b, r, t = orpref.bounds.lbrt()
        if k < 0 or mu1 < 0 or mu2 < 0 or x > r or x < l or y > t or y < b or aspect <= 0:
            return np.ones(orpref.data.shape).ravel()*10**6
        mask = ig.Disk(x=x, y=y, xdensity=orpref.xdensity, smoothing=0, ydensity=orpref.ydensity,
                       size=self.p.lateral_size, bounds=orpref.bounds)()
        gaussian = ig.Gaussian(x=x, y=y, xdensity=orpref.xdensity, ydensity=orpref.ydensity,
                               orientation=orpref[x, y], aspect_ratio=aspect, mask=mask,
                               size=mu1*2., bounds=orpref.bounds)
        fit = gaussian() * weight
        if vonMises:
            fit *= vonMises_fn(orpref.data, k, orpref[x, y])
        gaussian2 = ig.Gaussian(x=x, y=y, xdensity=orpref.xdensity, ydensity=orpref.ydensity,
                                mask=mask, size=mu2*2., bounds=orpref.bounds, aspect_ratio=1)
        fit += gaussian2()
        fit[fit < np.percentile(fit[fit.nonzero()], self.p.threshold)] = 0
        DivisiveNormalizeL1()(fit)
        if ravelled:
            fit = fit.ravel()
        return fit


    def __call__(self, tree, **params):
        self.p = param.ParamOverrides(self, params, allow_extra_keywords=True)
        grid = tree.CFs[self.p.projection]
        orpreference = tree.OrientationPreference[self.p.sheet]

        # Initialize Progress Bar
        progress = ProgressBar()
        progress(0)
        grid_length = len(grid.keys())

        # Initialize datastructures
        results = Layout()
        lat_grid = hv.GridSpace()
        fit_grid = hv.GridSpace()
        naive_grid = hv.GridSpace()
        error_grid = hv.GridSpace()
        naive_error = hv.GridSpace()
        naive_error_grid = hv.GridSpace()
        map_dims = grid.values()[0].kdims
        fit_tables = hv.HoloMap(kdims=['x', 'y']+map_dims)
        map_dims = [d.name for d in map_dims]
        for ((x, y), sheet_stack) in grid.items():
            for key, view in sheet_stack.items():
                orpref = orpreference.select(**dict(zip(map_dims, key)))
                if isinstance(orpref, HoloMap):
                    orpref = orpref.last
                try:
                    processed = self._process(view, orpref, (x, y))
                except RuntimeError:
                    continue
                data, fit_lateral, error, naive_lateral, naive_error, fit = processed
                fit_tables[(x, y)+key] = fit
                if (x, y) not in fit_grid:
                    lat_grid[(x, y)] = sheet_stack.clone({key: data})
                    fit_grid[(x, y)] = sheet_stack.clone({key: fit_lateral})
                    error_grid[(x, y)] = sheet_stack.clone({key: error})
                    naive_grid[(x, y)] = sheet_stack.clone({key: naive_lateral})
                    naive_error_grid[(x, y)] = sheet_stack.clone({key: naive_error})
                else:
                    lat_grid[x, y][key] = data
                    fit_grid[x, y][key] = fit_lateral
                    error_grid[x, y][key] = error
                    naive_grid[x, y][key] = naive_lateral
                    naive_error_grid[x, y][key] = naive_error
        results.set_path(('Preprocessed', 'CFs'), lat_grid)
        results.set_path(('vonMisesFit', 'CFs'), fit_grid)
        results.set_path(('vonMisesFit', 'Error'), error_grid)
        results.set_path(('NaiveFit', 'CFs'), naive_grid)
        results.set_path(('NaiveFit', 'Error'), naive_error_grid)
        results.set_path(('Results', 'Table'), fit_tables)
        return results

    def _process(self, cf, orpref, key=None):
        x, y = key
        lat_data = cf.situated.data.copy()
        lat_data[lat_data < np.percentile(lat_data[lat_data.nonzero()], self.p.threshold)] = 0
        DivisiveNormalizeL1()(lat_data)
        lateral_cf = cf.situated.clone(lat_data)
        initial = (1, 1, 0.1, 1, x, y)

        vdims = ['MSE', hv.Dimension((r'$r^2$', 'r2')), 'k',
                 'mu1', 'mu2', 'weight', 'xfit', 'yfit']
        if self.p.fit_aspect:
            vdims += ['aspect']
            initial += (1.,)

        # Fit naive and vonMises model
        fit, _ = curve_fit(self._function, orpref, lat_data.ravel(), p0=initial)
        naive_function = partial(self._function, vonMises=False)
        naive_fit, _ = curve_fit(naive_function, orpref,
                                 lat_data.ravel(), p0=initial)

        # Get fitted data and compute error
        observed_mean = lat_data.mean()
        fit_data = self._function(orpref, *fit, ravelled=False)
        error = (lat_data-fit_data)
        sum_of_squares = ((lat_data-observed_mean)**2).sum()
        rsquared = 1 - (error**2).sum()/sum_of_squares
        # Wrap data in HoloViews objects
        fit_lateral = hv.Image(fit_data, bounds=orpref.bounds)
        error_img = fit_lateral.clone(error, group='vonMises Error',
                                  vdims=['Square Error'])

        naive_data = naive_function(orpref, *fit, ravelled=False)
        naive_error = (lat_data-naive_data)
        rsquared_naive = 1 - (naive_error**2).sum()/sum_of_squares

        naive_lateral = hv.Image(naive_data, bounds=orpref.bounds)
        naive_error_img = fit_lateral.clone(naive_error, group='vonMises Error',
                                        vdims=['Square Error'])
        fit_results = [('vonMises', (error**2).mean(), rsquared) + tuple(fit),
                       ('Naive', (naive_error**2).mean(),
                        rsquared_naive) + tuple(naive_fit)]
        fit_table = hv.Table(fit_results, kdims=['Model'], vdims=vdims)
        return [lateral_cf, fit_lateral, error_img,
                naive_lateral, naive_error_img, fit_table]



class ComplexityAnalysis(ParameterizedFunction):
    """
    The complexity analysis takes HoloMap of Phase tuning curves as input
    and derives the modulation ratio for each unit by applying a FFT
    to a specified number of samples of the curve. The DC component (f0)
    and first harmonic (f1) are taken from the FFT and used to compute
    the modulation ratio f1/f0. The modulation ratio is a standard measure
    of the complexity of a cell, where values near 0 indicate fully complex
    cells, while values near 2.0 identify simple cells.
    """

    fft_sampling = param.Integer(default=8, doc="""
       Number of samples of the Phase tuning curves to take the FFT over.""")

    roi = param.NumericTuple(default=None, length=4, doc="""
       Region for which to compute the complexity.""")

    def __call__(self, grid, **params):
        p = ParamOverrides(self, params)

        return self._process(p, grid)

    def _process(self, p, phase_tuning):
        phase_image = phase_tuning.last
        cols, rows = phase_image.data.shape
        bounds = phase_image.bounds
        label = phase_image.label
        roi = p.roi if p.roi is not None else bounds.lbrt()
        with sorted_context(False):
            phase_tuning_df = DFrame(phase_tuning.sample((cols, rows), bounds=roi).dframe())
            grid = phase_tuning_df.curve('Phase', 'Response', ['Time', 'x', 'y']).grid(['x', 'y'])

        results = Layout()
        sheet_stack = HoloMap(None, key_dimensions=grid.values()[0].key_dimensions)
        for idx, ((x, y), curve_stack) in enumerate(grid.items()):
            for key, curve in curve_stack.data.items():
                if key not in sheet_stack:
                    complexity = np.zeros((int(rows), int(cols)))
                    sheet_stack[key] = Image(complexity, bounds, label=label,
                                             group='Modulation Ratio',
                                             value_dimensions=[Dimension('Modulation Ratio',
                                                                         range=(0, 2))])
                row, col = phase_image.sheet2matrixidx(x, y)
                ydata = curve.dimension_values(1)
                fft = np.fft.fft(list(ydata) * p.fft_sampling)
                dc_value = abs(fft[0])
                if dc_value != 0:
                    modulation_ratio = 2 * abs(fft[p.fft_sampling]) / dc_value
                    sheet_stack.data[key].data[row, col] = modulation_ratio

        results.set_path((label,), sheet_stack)
        return results



class response_latency(ElementOperation):

    def _process(self, view, key=None):
        xvals = view.dimension_values(0)
        yvals = view.dimension_values(1)

        if np.sum(yvals) == 0:
            peak_duration = np.NaN
        else:
            peak_duration = xvals[yvals.argmax()]

        return [ItemTable({'Peak Latency': peak_duration},
                          key_dimensions=[Dimension('Peak_Latency')],
                          label=view.label)]



class measure_response_latencies(UnitMeasurements):

    coords = param.List(default=[(0, 0)])

    durations = param.List(default=[0.05*i for i in range(21)])

    pattern = param.Parameter(default=ig.Gaussian)

    outputs = param.List(default=['V1Exc', 'V1PV', 'V1Sst'])

    relative_roi = param.NumericTuple(default=(-0.05, -0.05, 0.05, 0.05))

    def __call__(self, **params):
        p = param.ParamOverrides(self, params)

        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.outputs[0])
        lo, bo, ro, to = lbrt_offsets

        pattern_name = ''.join([p.pattern.__name__, 'Response'])

        results = Layout()
        output_results = [HoloMap(key_dimensions=['X', 'Y']) for output in p.outputs]
        for coord in coords:
            pattern = p.pattern(x=coord[0], y=coord[1])
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)
            data = measure_response(pattern_generator=pattern, durations=p.durations, outputs=p.outputs)
            for oidx, output in enumerate(p.outputs):
                response_grid = data[pattern_name][output].sample((cols, rows), bounds=lbrt).to.curve('Duration', 'Response').grid(['x', 'y'])
                latency_table = response_latency(response_grid)
                output_results[oidx][coord] = DFrame(latency_table.dframe().dropna())

        for output, result in zip(p.outputs, output_results):
            data = {(topo.sim.time(),): DFrame(result.dframe())}
            vmap = HoloMap(data, key_dimensions=[f.Time])
            results.set_path((''.join([pattern_name, 'Latencies']), output), vmap)

        return results


class TargetFlanker(Composite):
    """
    Draws target and flanker rectangles, replicating the stimulus protocol
    from Kapadia et al., Neuron (1995). The lateral and vertical offset,
    relative orientation of the flanker and their respective contrasts can
    be controlled.
    """

    targetcontrast = param.Number(default=0.9, doc="""
       Contrast of the target line segment.""")

    flankercontrast = param.Number(default=1.0, doc="""
       Contrast of the flanker line segment.""")

    height = param.Number(default=1.0, doc="""
       Height of the target and flanker.""")

    width = param.Number(default=0.1, doc="""
        Width of the target and flanker.""")

    x = param.Number(default=0, bounds=(-1.0, 1.0), softbounds=(-0.5, 0.5),
                     doc="X center of the target")

    y = param.Number(default=0, bounds=(-1.0, 1.0), softbounds=(-0.5, 0.5),
                     doc="X center of the target")

    xoffset = param.Number(default=0.0, doc="""
        Flanker X offet""")

    yoffset = param.Number(default=0.0, doc="""
        Flanker Y offet""")

    orientationoffset = param.Number(default=0, bounds=(-np.pi / 2, np.pi / 2),
                                     doc="""Flanker orientation offset""")

    def function(self, p):
        aspect = p.width/p.height

        # Apply x and y positional offsets
        x = p.x - p.xoffset
        y = p.y - p.yoffset - p.height

        # Apply orientation offset
        if p.orientationoffset:
            offset = np.sin(p.orientationoffset) * p.height/2.
            sign = -1 if p.orientationoffset < 0 else 1
            x += offset
            y += offset * sign

        bar_1 = RawRectangle(x=p.x, y=p.y, aspect_ratio=aspect,
                                    size=p.height, scale=p.targetcontrast)
        bar_2 = RawRectangle(x=x, y=y, aspect_ratio=aspect, size=p.height,
                                    orientation=p.orientationoffset, scale=p.flankercontrast)

        orientation = p.orientation - np.pi/2.

        return Composite(generators=[bar_1, bar_2], bounds=p.bounds, operator=np.add,
                         orientation=orientation, xdensity=p.xdensity,
                         ydensity=p.ydensity)()


OrientationOffset = f.Feature('OrientationOffset', range=(-np.pi/2., np.pi/2.))
FlankerContrast = f.Feature('FlankerContrast', range=(0, 1))
XOffset = f.Feature('XOffset', range=(-2, 2))
YOffset = f.Feature('YOffset', range=(-2, 2))


class measure_flanker_modulation(UnitCurveCommand):
    """
    Base class to implement Kapadia et al., Neuron (1995), target/flanker
    protocol. Allows varying the target and flanker X, Y and OR offsets and
    their individual contrasts.
    """

    pattern_generator = param.Callable(TargetFlanker())

    targetcontrast = param.Number(default=0.5)

    flankercontrasts = param.List(default=[30, 50, 70, 100])

    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)

        results = {}
        for coord in p.coords:
            p.orientation = p.preference_lookup_fn('orientation', p.outputs[0],
                                                 coord, default=0)

            p.x = p.preference_lookup_fn('x', p.outputs[0], coord,
                                         default=coord[0])
            p.y = p.preference_lookup_fn('y', p.outputs[0], coord,
                                         default=coord[1])

            results[coord] = self._compute_curves(p)
        results = self._populate_grid(results)

        self._restore_presenter_defaults()
        return results



class measure_flanker_ormodulation(measure_flanker_modulation):
    """
    Measures the modulation of the response to a target bar in the presence
    of a flanker with varying orientation offset.
    """

    num_orientation = param.Integer(default=13)

    x_axis = param.String(default='orientationoffset')

    xoffset = param.Number(default=0)

    yoffset = param.Number(default=0.5)

    static_parameters= param.List(
        default=["targetcontrast", "orientation", "x", "y",
                 "xoffset", "yoffset"])


    def _feature_list(self, p):
        return [OrientationOffset(steps=p.num_orientation),
                FlankerContrast(values=[float(c)/100 for c in p.flankercontrasts],
                                preference_fn=None)]



class measure_flanker_xoffsetmodulation(measure_flanker_modulation):
    """
    Measures the modulation of the response to a target bar in the presence
    of a flanker with X position offset.
    """

    num_offsets = param.Integer(default=13)

    max_offset = param.Number(default=1.0)

    x_axis = param.String(default='xoffset')

    orientationoffset = param.Number(default=0)

    yoffset = param.Number(default=0.5)

    static_parameters= param.List(
        default=["targetcontrast", "orientation", "x", "y",
                 "orientationoffset", "yoffset"])

    def _feature_list(self, p):
        return [XOffset(range=(-p.max_offset,p.max_offset), steps=p.num_offsets),
                FlankerContrast(values=[float(c)/100 for c in p.flankercontrasts],
                                preference_fn=None)]



class measure_flanker_yoffsetmodulation(measure_flanker_modulation):
    """
    Measures the modulation of the response to a target bar in the presence
    of a flanker with Y position offset.
    """

    num_offsets = param.Integer(default=13)

    max_offset = param.Number(default=1.0)

    x_axis = param.String(default='yoffset')

    orientationoffset = param.Number(default=0)

    xoffset = param.Number(default=0)

    static_parameters= param.List(
        default=["targetcontrast", "orientation", "x", "y",
                 "orientationoffset", "xoffset"])


    def _feature_list(self, p):
        return [YOffset(range=(0,p.max_offset), steps=p.num_offsets),
                FlankerContrast(values=[float(c)/100 for c in p.flankercontrasts],
                                preference_fn=None)]


options = Store.options(backend='matplotlib')

options.DFrame.Size_Tuning_Analysis = Options('plot')
options.DFrame.Contrast_Size_Tuning_Shift = Options('plot')
