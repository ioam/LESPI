import os
import re
import time
import glob as glob
import datetime as dt
from dateutil import tz
from functools import partial

import numpy as np
import param
import lancet
from lancet.filetypes import FileType, CustomFile
from holoviews.core.io import Unpickler
import holoviews as hv

custom=CustomFile(metadata_fn=lambda f: Unpickler.key(f),
                  data_fn = lambda f: {e: Unpickler.load(f, [e])
                                       for e in Unpickler.entries(f)})

def load_table(path):
    date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}\_[0-9]{4})", path).group(1)
    dirs = glob.glob(os.path.join(path, date+'*'))
    if not dirs:
        raise Exception('No data found in directory.')
    dir_suffix = dirs[0].split(date)[-1]
    match = re.search('\_t\d{1,}\_', dir_suffix)
    if not match:
        raise Exception('No directories indexed by tid found.')
    file_pattern = ''.join(['*', dir_suffix[:match.start(0)], '_t{tid:d}_*/*.hvz'])
    viewfnames = lancet.FilePattern('filename', os.path.join(path, file_pattern))
    filedata = lancet.FileInfo(viewfnames, 'filename', custom)
    file_df = filedata.dframe
    file_df['time'] = [float(t) for t in file_df.time]
    file_df['timestamps'] = np.array([dt.datetime.fromtimestamp(os.path.getmtime(fn),
                                                                tz.tzutc()).isoformat()
                                      for fn in viewfnames.dframe.filename],
                                     dtype=np.datetime64)
    return hv.Table(file_df, vdims=['filename'])


class LancetProgress(param.Parameterized):

    display_transforms = param.HookList(default=[lambda x: x])

    file_pattern = param.String(default='')

    path = param.String(default='')

    def __init__(self, path, **params):
        super(LancetProgress, self).__init__(path=path, **params)
        self.file_path = os.path.join(self.path, self.file_pattern)
        pre = time.time()
        self.table = self.get_table()
        self.timeout = (time.time() - pre) * 2
        self.timer = time.time()
        self.layout = self.get_layout()

    def get_table(self):
        viewfnames = lancet.FilePattern('filename', self.file_path)
        filedata = lancet.FileInfo(viewfnames, 'filename', custom)
        file_df = filedata.dframe
        file_df['time'] = [int(t) for t in file_df.time]
        file_df['timestamps'] = np.array([dt.datetime.fromtimestamp(os.path.getmtime(fn),tz.tzutc()).isoformat()
                                          for fn in viewfnames.dframe.filename], dtype=np.datetime64)
        return hv.Table(file_df, vdims=['filename']).sort(['tid', 'time'])

    def get_layout(self):
        return hv.Layout([hv.DynamicMap(partial(self.update, transform=transform), kdims=['Time'])
                          for transform in self.display_transforms])

    def update(self, i, transform=None):
        if time.time() - self.timer > self.timeout:
            self.timer = time.time()
            self.table = self.get_table()
            self.timeout = (time.time() - self.timer) * 2
            self.timer = time.time()
        return (time.ctime(), transform(self.table))


class ProgressWidget(param.ParameterizedFunction):

    path = param.String(default='')

    title_format = param.String(default='{run} {time}')

    width = param.Integer(default=1000)

    def __call__(self, path, **params):
        p = param.ParamOverrides(self, dict(params, path=path))
        log_path = glob.glob(os.path.join(p.path, '*.log'))
        if not log_path:
            raise IOError('No valid log found at supplied path.')
        log=lancet.Log(log_path[0])

        date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}\_[0-9]{4})", path).group(1)
        dirs = glob.glob(os.path.join(path, date+'*'))
        if not dirs:
            raise Exception('No data found in directory.')
        dir_suffix = dirs[0].split(date)[-1]
        match = re.search('\_t\d{1,}\_', dir_suffix)
        if not match:
            raise Exception('No directories indexed by tid found.')
        file_pattern = ''.join(['*', dir_suffix[:match.start(0)], '_t{tid:d}_*/*.hvz'])

        run = ''.join(os.path.basename(log.log_path).split('.')[:-1])
        time_idx = log.constant_keys.index('times')
        times = log.constant_items[time_idx][1]
        interval = np.diff(times)[0]
        num_bins = int((max(times)-min(times))/interval)+1
        bin_range = (-interval/2., max(times)+interval/2.)
        title = p.title_format.format(run=run, time=date)
        varying = [d for d in log.varying_keys if d != 'tid']

        def current_times(table):
            agg = table.reindex(['tid'], ['time']).aggregate(['tid'], np.max)
            hist = agg.hist(adjoin=False, num_bins=num_bins,
                            bin_range=bin_range, normed=False)
            plot_opts = dict(width=p.width, height=int(p.width/6.), xaxis=None,
                             tools=['hover'],shared_axes=False)
            completion = (agg['time'].sum()/float(times[-1]*len(log.dframe)))*100.
            progress = ' - Current Progress: {:.3g} %'.format(completion)
            return hist.clone(group=title+progress)(plot=plot_opts, norm=dict(axiswise=True))

        def count_times(table):
            hist = table.hist(dimension='time', adjoin=False,
                              num_bins=num_bins, bin_range=bin_range, normed=False)
            plot_opts = dict(width=p.width, height=int(p.width/6.), tools=['hover'],
                             shared_axes=False)
            return hist(plot=plot_opts, norm=dict(axiswise=True))

        def scatter_times(table):
            points = table.to.points(['tid', 'timestamps'],
                                     ['time'] + varying, [])
            points = points.clone(vdims=[hv.Dimension('time',range=(0, 80))]+varying)
            plot_opts=dict(width=p.width, height=int(p.width/3.), color_index=2,
                           size_index=None, tools=['hover'], xrotation=90,
                           xticks=list(range(points.range('tid')[1]+1)))
            style_opts=dict(size=8, marker='square', cmap='Blues', line_color='black')
            return points(plot=plot_opts, style=style_opts)

        def format_table(table):
            return table(plot=dict(width=p.width, height=int(p.width/5)))

        display_transforms = [current_times, count_times, scatter_times, format_table]
        progress = LancetProgress(path=p.path, file_pattern=file_pattern,
                                  display_transforms=display_transforms)
        return progress.layout.cols(1)


from holoviews.ipython import display
from ipywidgets import interact

def create_widget(path):
    try:
        display(ProgressWidget(path))
    except Exception as e:
        display(str(e))
