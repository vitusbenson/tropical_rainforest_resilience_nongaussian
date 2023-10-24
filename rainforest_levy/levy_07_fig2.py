# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
import scipy.integrate as integrate
import matplotlib.colors as clr
import seaborn as sns
from numba import jit
from scipy.stats import levy_stable, powerlaw, pareto
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib as mpl
from matplotlib import rc
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# %%
def func(x, c):
    return 0.3*(c)/(0.5+c)*x*(1-x/90) - 0.15*x*10/(x+10) - 0.11*x*64**7/(x**7+64**7) 
def func_int(x, c, x_fix = 73.33071370180014, extra_width = 1.):
    dT = 0.3*(c)/(0.5+c)*x*(1-x/90) - 0.15*x*10/(x+10) - 0.11*x*64**7/(x**7+64**7) 
    dT -= (0.5 * np.tanh((c - 3.) * 25) + 0.5) * (x - x_fix)/extra_width**3 * np.exp(- (x - x_fix)**2 / (2*extra_width**2))
    return dT
all_ys = []
all_ys_old = []
all_xs = []
cs = np.array([5., 4.7, 4.4, 4.1, 3.8, 3.5, 3.2, 2.9, 2.6, 2.3, 2.0])#np.linspace(5.0, 2.0, 31)
x_fix = 76.
for c in cs:
    if x_fix > 55. and x_fix < 69.:
        x_fix = 44.
    x_fix = fsolve(func, x0 = np.array([x_fix]), args = np.array([c]))[0]
    extra_width = 0.2 * np.exp(5 - c)# * (1 + (2.943 - c))#(0.5+c)/(c) #1.0#2.943
    ys = [-integrate.quad(func_int, 0.0, x, args = (c,x_fix, extra_width))[0] for x in np.linspace(0.1,100., 1000)]
    all_xs.append(np.linspace(0.1,100., 1000))
    all_ys.append(ys)
    ys_old = [-integrate.quad(func, 0.0, x, args = (c,))[0] for x in np.linspace(0.1,100., 1000)]
    all_ys_old.append(ys_old)
    
    #plt.plot(np.linspace(0.1,100., 1000), ys, label = c)
# plt.figure(dpi = 300, figsize = (4,6))
# plt.legend(bbox_to_anchor=(1,1), loc="upper left")


# %%
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc
# %%
red = sns.color_palette("rocket", as_cmap=True)(np.linspace(0.25, 0.8, 256))
blue = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.1, 0.75, 256))
all_colors = np.vstack((blue,3*[[0.0, 0.0, 0.0,1.0]], red))
redblue = clr.LinearSegmentedColormap.from_list(
    'terrain_map', all_colors)
divnorm = clr.TwoSlopeNorm(vmin=1.0, vcenter=2.9, vmax=5.0)
# %%
plt.figure(dpi = 300)
lc2 = multiline(all_xs, all_ys, cs, cmap = redblue, norm = divnorm, alpha = 0.5, lw = 1.5)
lc = multiline(all_xs, all_ys_old, cs, cmap = redblue, norm = divnorm, lw = 0.5, ls = "--")
plt.colorbar(lc)
# %%

# %%
x_fix = 76.
dt = 0.01
cs2 = np.linspace(5.0, 3.0, 11)
disturbs = np.concatenate([[0.01, 0.05],np.linspace(0.1, 5.0, 50)])
times_xs = []
times_ts = []
times_xs_old = []
times_ts_old = []
for i, c in enumerate(cs2):
    if x_fix > 55. and x_fix < 69.:
        x_fix = 44.
    x_fix = fsolve(func, x0 = np.array([80]), args = np.array([c]))[0]
    extra_width = 0.2 * np.exp(5 - c)
    m = 0
    x0 = x_fix - 5 +m
    while not np.isclose(fsolve(func, x0 = np.array([x0]), args = np.array([c]))[0], x_fix):
        print(x_fix,fsolve(func, x0 = np.array([x0]), args = np.array([c]))[0], x, c,m)
        m+=1
        x0 = x_fix - 5 +m
    #for j, disturb in enumerate(disturbs):
    #x = x_fix - disturb
    t = 0.0
    ts = [t]
    print(x0, x_fix)
    x = x0
    xs = [x-x_fix]
    while np.abs(x - x_fix) > 0.01:
        x += dt * func_int(x, c, x_fix = x_fix, extra_width = extra_width)
        t += dt
        ts.append(t)
        xs.append(x-x_fix)
    times_xs.append(xs)
    times_ts.append([t-ct for ct in ts])

    x = x0
    t = 0.0
    ts = [t]
    xs = [x-x_fix]
    while np.abs(x - x_fix) > 0.01:
        x += dt * func(x, c)
        t += dt
        ts.append(t)
        xs.append(x-x_fix)
    times_xs_old.append(xs)
    times_ts_old.append([t-ct for ct in ts])

# %%
mpl.rcParams['xtick.labelsize'] = 12 
mpl.rcParams['ytick.labelsize'] = 12 
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
fig, axs = plt.subplots(1,2, dpi = 300, figsize = (8,4), gridspec_kw={'width_ratios': [1, 1.2]})

lw_my = 1.2
lw_vn = 1.2
ls_my = "-"
ls_vn = (0, (2, 1, 1, 1))#(0, (1, 0.5))
alpha_my = 1.
alpha_vn = 1.#0.5


lc2 = multiline(all_xs, all_ys_old, cs, cmap = redblue, norm = divnorm, alpha = alpha_vn, lw = lw_vn, ls=ls_vn, ax = axs[0])
lc = multiline(all_xs, all_ys, cs, cmap = redblue, norm = divnorm, lw = lw_my, ax = axs[0], alpha = alpha_my, ls=  ls_my)

axs[0].set_ylabel(r'Potential $V(T)$')
axs[0].set_xlabel(r'Tree Cover $T [\%]$')
axs[0].text(-0.15, 1.0, "a)", transform=axs[0].transAxes, 
            size=14, weight='bold')

x1, x2, y1, y2 = 68, 78, -44, -15#-1.5, -0.9, -2.5, -1.9  # subregion of the original image
axins = axs[0].inset_axes(
    [0.2, 0.5, 0.37, 0.47],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

lc2 = multiline(all_xs, all_ys_old, cs, cmap = redblue, norm = divnorm, alpha = alpha_vn, lw = lw_vn, ls=ls_vn, ax = axins)
lc = multiline(all_xs, all_ys, cs, cmap = redblue, norm = divnorm, lw = lw_my, ax = axins, alpha = alpha_my, ls=  ls_my)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axs[0].indicate_inset_zoom(axins, edgecolor="black",)

#axins.imshow(Z2, extent=extent, origin="lower")


lc4 = multiline(times_xs_old, times_ts_old, cs2, cmap = redblue, norm = divnorm, alpha = alpha_vn, lw = lw_vn, ls=ls_vn, ax = axs[1])
lc3 = multiline(times_xs, times_ts, cs2, cmap = redblue, norm = divnorm, lw = lw_my , ls=  ls_my, alpha = alpha_my, ax = axs[1])

axs[1].set_yscale("log")
axs[1].set_ylabel(r'Return Time $t_R$')
axs[1].set_xlabel(r'Disturbance of Tree Cover $D_T [\%]$')
axs[1].text(-0.15, 1.0, "b)", transform=axs[1].transAxes, 
            size=14, weight='bold')
plt.colorbar(lc, ax = axs[1], label = r'Precipitation $P [\frac{mm}{day}]$')
custom_lines = [Line2D([0], [0], color="black", lw = lw_my, alpha = alpha_my,ls=  ls_my,),
                Line2D([0], [0], color="black", alpha = alpha_vn, lw = lw_vn, ls=ls_vn),
                ]
plt.figlegend(custom_lines, ['Present Work','Potential in van Nes et al. [35]'], loc = 'lower center', ncols = 2,bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
plt.savefig("/Users/vbenson/Coding/amaz_ews_powerlaw/plots/fig2.pdf", dpi = 300, bbox_inches = "tight", transparent = True)
## %%

# %%
