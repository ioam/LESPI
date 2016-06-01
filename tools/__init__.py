"""
IPython Tools taken from Jean-Luc Stevens TCAL repository.
"""

import os, sys, time, re
from xml.etree import ElementTree as et

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import holoviews as hv
from holoviews import NdMapping
from holoviews.interface.collector import AttrTree
from holoviews.plotting.mpl import ElementPlot, Element

# Open svg file with ElementTree XML parser

def fix_alpha(svg_file):
    xmlRoot = et.parse(svg_file).getroot()

    opacityTarget = re.compile(r'((?<=^opacity:)|(?<=;opacity:))\d+\.?\d*')
    for target in xmlRoot.findall('.//*[@style]'):
        m = opacityTarget.search(target.attrib['style'])
        if m and float(m.group(0)) < 1.0:
            # replace opacity target to 100%
            target.attrib['style'] = re.sub(opacityTarget, '1', target.attrib['style'])

            # move opacity to new fill-opacity attribute, supported by pdfs
            target.attrib['style'] += ';fill-opacity:{0}'.format(m.group(0))

    # write to output file (can be the same file as input)
    with open(svg_file, 'wb') as outputFile:
        et.ElementTree(xmlRoot).write(outputFile)



class MPLFigure(Element):
    pass

class MPLFigPlot(ElementPlot):

    def _init_axis(self, fig, axis):
        return None, None

    def initialize_plot(self, ranges=None):
        self.handles['fig'] = self.hmap.last.data
        return self.hmap.last.data

def embed_object(element, fig=None, ax=None):
    fig = plt.gcf() if fig is None else fig
    ax = plt.gca() if ax is None else ax
    plot = hv.Store.registry['matplotlib'][type(element)]
    return plot(element, fig=fig, axis=ax).initialize_plot()




def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    cmap = matplotlib.cm.get_cmap(cmap)

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


hv.Store.register({MPLFigure:MPLFigPlot}, 'matplotlib')
