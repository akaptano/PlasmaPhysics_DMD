## @package plot_attributes
## Defines simple plotting line sizes, ticks,
## colors, and so on, for uniform plotting parameters
## across all files in this project.
import matplotlib
import matplotlib.animation as animation
matplotlib.use("Agg")
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe
rcParams.update({'figure.autolayout': True})
from numpy import pi
import numpy as np
## Default output directory for all images and movies
out_dir = 'Pictures/'
## Toroidal angle labels for the (R,phi) contour plots
clabels = ['0',r'$\frac{\pi}{2}$', \
    r'$\pi$', \
    r'$\frac{3\pi}{2}$', r'$2\pi$']
## Permeability of free space
mu0 = 4*pi*10**(-7)
## Fontsize for axis labels and titles
fs = 40
## Fontsize for axis ticks
ts = 30
## Fontsize for pyplot legends
ls = 30
## Markersize for scatter plots
ms = 20
## Linewidth for line plots
lw = 4
## Figure size in horizontal dir
figx = 14
## Figure size in vertical dir
figy = 10
## pixels per inch
ppi = 100
## How transparent to make scatter plots with multiple entries
transparency = 1.0
## Colors to distinguish between dictionaries
Colors = np.random.rand(100,3)
## colors specifically for plotting two-temperature MHD results
colors2T = ['r','b','g','k','c']
## Styles to distinguish between discharges with different
## injector frequencies
Styles = ['dashed','dotted','-.','solid',':']
## Labels to distinguish between discharges with different
## data sources
Labels = ['Experimental ', 'NIMROD ', 'NIMROD 2T ', 'PSI-Tet ', 'PSI-Tet 2T ']
## The default colormap to use for contour plots
colormap = 'PiYG'
