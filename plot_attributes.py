## @package plot_attributes
# Defines simple plotting line sizes, ticks,
# colors, and so on, for uniform plotting parameters
# across all files in this project.
import matplotlib
import matplotlib.animation as animation
matplotlib.use("Agg")
import matplotlib.colors as colors
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.patheffects as pe
rcParams.update({'figure.autolayout': True})
from numpy import pi
import numpy as np
# @var Default output directory for all images and movies
out_dir = 'Pictures/'
# @var Toroidal angle labels for the (R,phi) contour plots
clabels = ['0',r'$\frac{\pi}{2}$', \
    r'$\pi$', \
    r'$\frac{3\pi}{2}$', r'$2\pi$']
# @var Phase colorbar labels for the (R,phi) phase contour plots
cbarlabels = ['0','',r'$\frac{\pi}{3}$', \
    '', r'$\frac{2\pi}{3}$', '', \
    r'$\pi$', '', r'$\frac{4\pi}{3}$', \
    '',r'$\frac{5\pi}{3}$','',
    r'$2\pi$']
# @var mu0 permeability of free space
mu0 = 4*pi*10**(-7)
# @var fs Fontsize for axis labels and titles
fs = 50
# @var ts Fontsize for axis ticks
ts = 38
ls = 30
# @var ms Markersize for scatter plots
ms = 20
# @var lw Linewidth for line plots
lw = 7
# @var figx Figure size in horizontal dir
figx = 14
# @var figy Figure size in vertical dir
figy = 10
# @var ppi pixels per inch
ppi = 100
# @var transparency How transparent to make scatter plots with multiple entries
transparency = 1.0
# @var Colors Colors to distinguish between dictionaries
Colors = np.random.rand(100,3)
# @var colors2T colors specifically for plotting 2T results 
colors2T = ['r','b','g','k','c']
# @var Styles Styles to distinguish between injector frequencies
Styles = ['dashed','dotted','-.','solid',':']
# @var Labels Labels to distinguish between data sources
Labels = ['Experimental ', 'NIMROD ', 'NIMROD 2T ', 'PSI-Tet ', 'PSI-Tet 2T ']
# @var colormap The default colormap to use for contour plots
colormap = 'PiYG' # plt.cm.plasma
