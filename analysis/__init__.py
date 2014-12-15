import numpy as np
import pandas
from scipy.optimize import curve_fit

import param
from param import ParameterizedFunction, ParamOverrides

from holoviews import ViewMap, Dimension, Matrix, Table, Grid, ItemTable, AttrTree
from holoviews.core.options import options, PlotOpts
from holoviews.ipython.widgets import ProgressBar
from holoviews.interface.seaborn import DFrame
from holoviews.operation import MapOperation, ViewOperation

import imagen
from imagen import Composite, RawRectangle, Gaussian

from featuremapper.analysis.spatialtuning import SizeTuningPeaks, SizeTuningShift,\
    OrientationContrastAnalysis, FrequencyTuningAnalysis
from featuremapper.command import FeatureCurveCommand, DistributionStatisticFn, \
    measure_size_response, measure_orientation_contrast, DSF_MaxValue, \
    UnitCurveCommand, MeasureResponseCommand, measure_frequency_response, measure_response
import featuremapper.features as f

import topo


class SheetReductionCurve(MapOperation):
    """
    Takes in a Stack of SheetViews, reduces them using the provided
    reduce_fn and collates them across the provided dimension.
    """

    dimension = param.String(default='Time')

    reduce_fn = param.Callable(default=np.mean)

    def _process(self, view):
        return [view.reduce(X=self.p.reduce_fn, Y=self.p.reduce_fn).collate(self.p.dimension)]


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

        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.outputs[0])
        lo, bo, ro, to = lbrt_offsets

        size_dataframes = defaultdict(list)
        css_dataframes = defaultdict(list)

        measurement_params = dict(frequencies=[p.frequency], max_size=p.max_size,
                                  num_sizes=p.num_sizes, num_phase=p.num_phase,
                                  outputs=p.outputs, contrasts=p.contrasts)
        results = AttrTree()
        for coord in coords:
            data = measure_size_response(coords=[coord], **measurement_params)
            # Compute relative offsets from the measured coordinate
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)

            for output in p.outputs:
                size_grid = data.SizeTuning[output].sample((cols, rows), bounds=lbrt).collate('Size')
                size_data = SizeTuningPeaks(size_grid)
                size_df = size_data.dframe()

                # Filter out values with significantly different OR preferences
                ref_orpref = orpref.last[coord]
                ortables = orpref.sample((rows, cols), bounds=lbrt)
                or_df = ortables[topo.sim.time(), :].dframe()
                or_df.rename(columns={'X': 'Grid_X', 'Y': 'Grid_Y'}, inplace=True)
                size_df = pandas.merge(size_df, or_df, on=['Grid_X', 'Grid_Y', 'Time', 'Duration'])
                filter_condition = (np.abs(size_df[orpref.last.value.name] - ref_orpref) % np.pi) < p.max_ordiff
                size_df = size_df[filter_condition]
                size_dataframes[output].append(size_df)

                # If multiple contrasts have been measured find the contrast dependent size tuning shift
                if len(p.contrasts) >= 2:
                    contrast_grid = size_grid.map(lambda x, k: x.overlay(['Contrast']))
                    css_grid = SizeTuningShift(contrast_grid)
                    css_df = css_grid.dframe()
                    css_df = pandas.merge(css_df, or_df, on=['Grid_X', 'Grid_Y', 'Time', 'Duration'])
                    filter_condition = (np.abs(css_df[orpref.last.value.name] - ref_orpref) % np.pi) < p.max_ordiff
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
                size_stack = ViewMap(None, dimensions=['Time', 'Contrast'])
                for k, cdf in size_df.groupby(['Contrast']):
                    cdf = cdf.filter(['Peak_Size', 'SI', 'Suppression_Size', 'CS_Size', 'CSI'])
                    size_stack[(topo.sim.time(), k)] = DFrame(cdf, label='Size Tuning Analysis')
                    results.set_path(('SizeTuning', output), size_stack)

                if css_dataframes:
                    css_df = pandas.concat(css_dataframes[output])
                    css_df = css_df.filter(['CSS', 'Low', 'High'])
                    contrast_stack = ViewMap((topo.sim.time(), DFrame(css_df, label='Contrast Size Tuning Shift')),
                                             dimensions=['Time'])
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

        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.output)
        lo, bo, ro, to = lbrt_offsets

        size_dataframes = []

        measurement_params = dict(max_frequencies=p.max_freq, size=p.size,
                                  num_freq=p.num_freq, num_phase=p.num_phase,
                                  outputs=[p.output], contrasts=p.contrasts)
        results = AttrTree()
        for coord in coords:
            data = measure_frequency_response(coords=[coord], **measurement_params)

            # Compute relative offsets from the measured coordinate
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)

            freq_grid = data.FrequencyTuning[p.output].sample((cols, rows), bounds=lbrt).collate('Frequency')
            freq_data = FrequencyTuningAnalysis(freq_grid)
            freq_df = freq_data.dframe()

            # Filter out values with significantly different OR preferences
            ref_orpref = orpref.last[coord]
            ortables = orpref.sample((rows, cols), bounds=lbrt)
            or_df = ortables[topo.sim.time(), :].dframe()
            or_df.rename(columns={'X': 'Grid_X', 'Y': 'Grid_Y'}, inplace=True)
            freq_df = pandas.merge(freq_df, or_df, on=['Grid_X', 'Grid_Y', 'Time', 'Duration'])
            filter_condition = (np.abs(freq_df[orpref.last.value.name] - ref_orpref) % np.pi) < p.max_ordiff
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
        size_stack = ViewMap(None, dimensions=['Time', 'Contrast'])
        for k, cdf in freq_df.groupby(['Contrast']):
            cdf = cdf.filter(['Peak', 'Bandwidth', 'QFactor', 'Lower', 'Upper'])
            size_stack[(topo.sim.time(), k)] = DFrame(cdf, label='Frequency Tuning Analysis')
        results.set_path(('FrequencyTuning', p.output), size_stack)

        return results


class measure_iso_suppression(UnitMeasurements):

    coords = param.List(default=[(0, 0)])

    sizecenter = param.Number(default=1.0, bounds=(0, None), doc="""
        The size of the central pattern to present.""")

    sizesurround = param.Number(default=3.0, bounds=(0, None), doc="""
        The size of the surround pattern to present.""")

    thickness = param.Number(default=2.0, bounds=(0, None), softbounds=(0, 1.5),
                             doc="""Ring thickness.""")

    contrastsurround = param.List(default=[30, 100],
                                  doc="Contrast of the surround.")

    contrastcenter = param.Number(default=100, bounds=(0, 100),
                                  doc="""Contrast of the center.""")

    output = param.String(default='V1')

    frequency = param.Number(default=1.65)

    num_orientation = param.Integer(default=9)

    max_ordiff = param.Number(default=np.pi/8)

    relative_roi = param.NumericTuple(default=(-0.25, -0.25, 0.25, 0.25))

    def __call__(self, orpref, **params):
        p = param.ParamOverrides(self, params)

        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.output)
        lo, bo, ro, to = lbrt_offsets

        orcs_dataframes = []

        measurement_params = dict(frequencies=[p.frequency], outputs=[p.output],
                                  contrastsurround=p.contrastsurround,
                                  contrastcenter=p.contrastcenter,
                                  thickness=p.thickness,
                                  num_orientation=p.num_orientation)
        results = AttrTree()
        for coord in coords:
            data = measure_orientation_contrast(coords=[coord], **measurement_params)

            # Compute relative offsets from the measured coordinate
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)

            orcs = data.OrientationsurroundTuning[p.output].sample((cols, rows), bounds=lbrt).collate('OrientationSurround')
            orcs_data = OrientationContrastAnalysis(orcs)
            orcs_df = orcs_data.dframe()

            # Filter out values with significantly different OR preferences
            ref_orpref = orpref.last[coord]
            ortables = orpref.sample((rows, cols), bounds=lbrt)
            or_df = ortables[topo.sim.time(), :].dframe()
            or_df.rename(columns={'X': 'Grid_X', 'Y': 'Grid_Y'}, inplace=True)
            orcs_df = pandas.merge(orcs_df, or_df, on=['Grid_X', 'Grid_Y', 'Time', 'Duration'])
            filter_condition = (np.abs(orcs_df[orpref.last.value.name] - ref_orpref) % np.pi) < p.max_ordiff
            orcs_df = orcs_df[filter_condition]
            orcs_dataframes.append(orcs_df)

            # Add Size Response to results
            path = ('OrientationContrastResponse', p.output)
            orcs_response = data.OrientationsurroundTuning[p.output]
            if path in results:
                results.OrientationContrastResponse[p.output].update(orcs_response)
            else:
                results.set_path(path, orcs_response)

        # Stack the Size Tuning Data
        orcs_df = pandas.concat(orcs_dataframes)
        orcs_stack = ViewMap(None, dimensions=['Time', 'ContrastSurround'])
        for k, cdf in orcs_df.groupby(['ContrastSurround']):
            cdf = cdf.filter(['OCSI'])
            orcs_stack[(topo.sim.time(), k)] = DFrame(cdf, label='Orientation Contrast Analysis')
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

    roi = param.NumericTuple(default=(-0.48, -0.48, 0.48, 0.48))

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
        l, b, r, t = stack.last.lbrt
        width = r - l

        lo, bo, ro, to = p.roi
        roi_width = ro-lo
        roiratio = roi_width/width

        if isinstance(orpref, ViewMap):
            orpref = orpref.last

        # Process dimension values and reduced dimensions
        orvalues = stack.dim_values('Orientation')
        phasevalues = stack.dim_values('Phase')
        reduced_dimensions = [d for d in stack.dimensions if d.name != 'Orientation']

        # Initialize progress bar
        phase_progress = ProgressBar(label='Phase Collapse')
        phase_progress(0)

        # Reduce the original stack by finding the response to the
        # optimally orientated stimulus for each unit for each phase
        reduced_stack = stack.clone(None, dimensions=reduced_dimensions)
        for idx, phase in enumerate(phasevalues):
            reduced_response = np.zeros((rows, cols))
            phase_stack = stack.select(Phase=phase)
            keys = [tuple(v for d, v in zip(stack.dimension_labels, k) if d != 'Orientation')
                    for k in phase_stack.keys()]
            for key in set(keys):
                key_slice = list(key)
                key_slice.insert(stack.dim_index('Orientation'), slice(None))
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

        grid = reduced_stack.sample((roiratio*rows+2, roiratio*cols+2), bounds=p.roi).collate('Phase')

        del results.PhaseTuning[output]
        results.set_path(('PhaseTuning', output), grid)
        return results


    def _feature_list(self, p):
        return [f.Orientation(steps=p.num_orientation, preference_fn=None),
                f.Phase(steps=p.num_phase),
                f.Frequency(values=p.frequencies)]



class RFGaborFit(param.ParameterizedFunction):
    """
    RFGaborFit takes a grid of measured RFs as input and attempts
    to fit a 2D Gabor function to each RF. It returns Grids of
    the Gabor fit, the fit parameters and residual from the fit.
    """

    max_iterations = param.Integer(default=1000)

    roi_radius = param.Number(default=None, allow_None=True)

    init_fit = param.NumericTuple(default=(1.0, 1.7, 0.1, 0.2, 0, 0))

    def _validate_rf(self, rf):
        if not isinstance(rf, Matrix):
            raise Exception('Supplied views need to be curves.')

    def _function(self, (x, y), A=1.0, f=1.7, sig_x=0.1, sig_y=0.2, phase=0, theta=0):
        try:
            x = (x-self.x0) * np.cos(theta) + (y+self.y0)*np.sin(theta)
            y = -(x-self.x0) * np.sin(theta) + (y+self.y0)*np.cos(theta)
            result = (A * np.exp(-(x/np.sqrt(2*sig_x))**2 - (y/np.sqrt(2*sig_y))**2) * np.cos(2*np.pi*f*x + phase)).ravel()
        except RuntimeError:
            result = np.ones(x.shape).ravel() * 10000
        except FloatingPointError:
            result = np.ones(x.shape).ravel() * 10000
        return result

    def __call__(self, grid, **params):
        self.p = param.ParamOverrides(self, params, allow_extra_keywords=True)
        results = AttrTree()

        fit_grid = grid.clone()
        residual_grid = grid.clone()
        fitvals_grid = Grid(None)
        for idx, ((x, y), sheet_stack) in enumerate(grid.items()):
            for key, view in sheet_stack.items():
                processed = self._process(view, dict(grid.key_items((x, y)), **sheet_stack.key_items(key)))
                if processed is None: continue
                if (x, y) not in fit_grid:
                    fit_grid[(x, y)] = sheet_stack.clone({key: processed[0]})
                    residual_grid[(x, y)] = sheet_stack.clone({key: processed[1]})
                    fitvals_grid[(x, y)] = ViewMap({key: processed[2]}, dimensions=sheet_stack.dimensions)
                else:
                    fit_grid[(x, y)][key] = processed[0]
                    residual_grid[(x, y)][key] = processed[1]
                    fitvals_grid[(x, y)][key] = processed[2]
        results.set_path(('RFGaborFit', 'RF_Fit'), fit_grid)
        results.set_path(('RFGaborFit', 'RF_Residuals'), residual_grid)
        results.set_path(('RFGaborFit', 'RF_Fit_Values'), fitvals_grid)
        return results

    def _process(self, rf, key=None):
        if hasattr(rf, 'situated'):
            rf = rf.situated
        self._validate_rf(rf)

        x, y = key['X'], key['Y']
        self.x0, self.y0 = x, y
        l, b, r, t = rf.lbrt
        if self.p.roi_radius:
            off = self.p.roi_radius
            rf = rf[x-off:x+off, y-off:y+off]
        data = rf.data - rf.data.mean()
        rows, cols = data.shape

        xs, ys = np.linspace(l, r, cols), np.linspace(b, t, rows)
        xs, ys = np.meshgrid(xs, ys)
        try:
            fit, pcov = curve_fit(self._function, (xs, ys), data.ravel(), self.p.init_fit, maxfev=self.max_iterations)
        except RuntimeError:
            return None
        fit_data = self._function((xs, ys), *fit).reshape(rows, cols)
        nx, ny= (fit[1] * fit[2], fit[1] * fit[3])
        residual = data - fit_data
        fit_table = Table(dict(zip(['A', 'f', 'sig_x', 'sig_y', 'phase','theta'], fit), **{'nx':nx, 'ny': ny, 'residual': np.sum(np.abs(residual))}))

        return [rf.clone(fit_data), rf.clone(residual, label='Residual'), fit_table]



class ComplexityAnalysis(ParameterizedFunction):
    """
    The complexity analysis takes a Grid of Phase tuning curves as input
    and derives the modulation ratio for each unit by applying a FFT
    to a specified number of samples of the curve. The DC component (f0)
    and first harmonic (f1) are taken from the FFT and used to compute
    the modulation ratio f1/f0. The modulation ratio is a standard measure
    of the complexity of a cell, where values near 0 indicate fully complex
    cells, while values near 2.0 identify simple cells.
    """

    fft_sampling = param.Integer(default=8, doc="""
       Number of samples of the Phase tuning curves to take the FFT over.""")

    def __call__(self, grid, **params):
        p = ParamOverrides(self, params)

        return self._process(p, grid)

    def _process(self, p, grid):
        l, b, r, t = grid.lbrt()
        width, height = r - l, t - b
        cols, rows = round(width * grid.xdensity), round(height * grid.ydensity)

        results = AttrTree()

        sheet_stack = ViewMap(None, dimensions=grid.values()[0].dimensions)
        for idx, ((x, y), curve_stack) in enumerate(grid.items()):
            for key in curve_stack.keys():
                if key not in sheet_stack:
                    complexity = np.zeros((int(rows), int(cols)))
                    sheet_stack[key] = Matrix(complexity, grid.bounds,
                                              label='Complexity Analysis',
                                              value='Modulation Ratio')
                row, col = grid.sheet2matrixidx(x, y)
                ydata = curve_stack[key].data[:, 1]
                fft = np.fft.fft(list(ydata) * p.fft_sampling)
                dc_value = abs(fft[0])
                if dc_value != 0:
                    modulation_ratio = 2 * abs(fft[p.fft_sampling]) / dc_value
                    sheet_stack[key].data[row, col] = modulation_ratio

        results.set_path(('Complexity', 'V1'), sheet_stack)
        return results



class response_latency(ViewOperation):

    def _process(self, view, key=None):
        xvals = view.data[:, 0]
        yvals = view.data[:, 1]

        if np.sum(yvals) == 0:
            peak_duration = np.NaN
        else:
            peak_duration = xvals[yvals.argmax()]

        return [ItemTable({'Peak Latency': peak_duration},
                          dimensions=[Dimension('Peak_Latency')],
                          label=view.label)]



class measure_response_latencies(UnitMeasurements):

    coords = param.List(default=[(0, 0)])

    durations = param.List(default=[0.05*i for i in range(21)])

    pattern = param.Parameter(default=imagen.Gaussian)

    outputs = param.List(default=['V1Exc', 'V1PV', 'V1Sst'])

    relative_to = param.String(default=['V1Exc'])

    relative_roi = param.NumericTuple(default=(-0.05, -0.05, 0.05, 0.05))

    def __call__(self, **params):
        p = param.ParamOverrides(self, params)

        coords, lbrt_offsets, cols, rows = self._compute_roi(p, p.outputs[0])
        lo, bo, ro, to = lbrt_offsets

        pattern_name = ''.join([p.pattern.__name__, 'Response'])

        results = AttrTree()
        output_results = [ViewMap(dimensions=['Grid_X', 'Grid_Y']) for output in p.outputs]
        for coord in coords:
            pattern = p.pattern(x=coord[0], y=coord[1])
            lbrt = (coord[0]+lo, coord[1]+bo, coord[0]+ro, coord[1]+to)
            data = measure_response(pattern_generator=pattern, durations=p.durations, outputs=p.outputs)
            for oidx, output in enumerate(p.outputs):
                response_grid = data[pattern_name][output].sample((cols, rows), bounds=lbrt).collate('Duration')
                latency_table = response_latency(response_grid)
                output_results[oidx][coord] = DFrame(latency_table.dframe().dropna())

        for output, result in zip(p.outputs, output_results):
            data = {(topo.sim.time(),): DFrame(result.dframe())}
            vmap = ViewMap(data, dimensions=[f.Time])
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


options.Size_Tuning_Analysis_DFrameView = PlotOpts()
options.Contrast_Size_Tuning_Shift_DFrameView = PlotOpts()
