import matplotlib.colors as pltcolors
import numpy as np


def plotmat(m, ax):
    img = ax.imshow(m, cmap = 'RdBu_r', clim=(m.min(), m.max()),
                    norm = MidpointNormalize(midpoint=0,
                                             vmin=m.min(),
                                             vmax=m.max())
                   )
    return img


class MidpointNormalize(pltcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        pltcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
