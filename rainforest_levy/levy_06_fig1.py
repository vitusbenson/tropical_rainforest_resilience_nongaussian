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
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
mpl.rcParams['xtick.labelsize'] = 12 
mpl.rcParams['ytick.labelsize'] = 12 
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['legend.title_fontsize'] = 12

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# %%
from sympy.solvers import solve
from sympy import diff
from sympy.abc import c, x
from sympy import lambdify

# %%
expr = 0.3*(c)/(0.5+c)*x*(1-x/90) - 0.15*x*10/(x+10) - 0.11*x*64**7/(x**7+64**7)

sols = []
for c_curr in tqdm(np.linspace(5.0, 0.5, 46)):
    sol = solve(expr.subs({c: c_curr}), x)
    sol = np.array([s for s in [s for s in sol if s.is_real] if (float(s) >= 0.0)]).astype(float)
    sols.append((c_curr, sol))


all_c = []
all_x0 = []
for c_curr, s in sols:
    for x0 in s:
        all_c.append(c_curr)
        all_x0.append(x0)

xs = np.array(all_x0)
cs = np.array(all_c)
idxs = np.argsort(xs)
xs = xs[idxs]
cs = cs[idxs]

y_f = interp1d(xs[11*4+2:], cs[11*4+2:], "cubic")
xshr = np.linspace(xs[11*4+2:].min(), xs.max(), 501)
cshr = y_f(xshr)

df = pd.DataFrame({"P": np.concatenate([[0.5, 5.0],cshr]), "T": np.concatenate([[0.0, 0.0],xshr])})
df.to_csv("../data/fixpoints.csv", index=False)

# %%
df = pd.read_csv("../data/fixpoints.csv")

# %%
fig = plt.figure(figsize = (6,6))
ax = plt.gca()
idxs = np.where(np.sign(df.P.diff()).diff().ne(0))[0][3:]
df[0:2].plot(x = "P", y = "T", legend = False, ylabel = r'Equilibrium Tree Cover $T$ $[\%]$', xlabel = r'Precipitation $P [\frac{mm}{day}]$', xlim = (0,5.2), ylim = (-5,102), ax = ax, color = "black")
df[1:idxs[0]].plot(x = "P", y = "T", legend = False, ylabel = r'Equilibrium Tree Cover $T$ $[\%]$', xlabel = r'Precipitation $P [\frac{mm}{day}]$', xlim = (0,5.2), ylim = (-5,102), ax = ax, color = "black", ls = "--")
df[idxs[0]:idxs[1]].plot(x = "P", y = "T", legend = False, ylabel = r'Equilibrium Tree Cover $T$ $[\%]$', xlabel = r'Precipitation $P [\frac{mm}{day}]$', xlim = (0,5.2), ylim = (-5,102), ax = ax, color = "black")
df[idxs[1]:idxs[2]].plot(x = "P", y = "T", legend = False, ylabel = r'Equilibrium Tree Cover $T$ $[\%]$', xlabel = r'Precipitation $P [\frac{mm}{day}]$', xlim = (0,5.2), ylim = (-5,102), ax = ax, color = "black", ls = "--")
df[idxs[2]:].plot(x = "P", y = "T", legend = False, ylabel = r'Equilibrium Tree Cover $T$ $[\%]$', xlabel = r'Precipitation $P [\frac{mm}{day}]$', xlim = (0,5.2), ylim = (-5,102), ax = ax, color = "black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.plot(1, -5, ">k", transform = ax.get_yaxis_transform(), clip_on = False)
ax.plot(0, 1, "^k", transform = ax.get_xaxis_transform(), clip_on = False)
handles = [Line2D([0], [0], color="black", lw=1), Line2D([0], [0], color="black", lw=1, ls = "--")]
plt.legend(handles, ['Stable State', 'Unstable State'])
plt.savefig("../plots/fig1a.png", dpi = 300, bbox_inches = "tight", transparent = True)

# %%
fig = plt.figure(figsize=(6,6), dpi =300)
ax = plt.gca()
x2 = np.linspace(2.2,20, 100).tolist()
for alpha in [0.5, 1.0, 1.5, 1.75, 1.95, 2.0]:
    ax.loglog(x2, x2*levy_stable.pdf(x2, alpha, 1),label=alpha if alpha<2.0 else f"2.0 (Gaussian)")
ax.set_ylim(np.e**-8,np.e**0)
ax.set_xlim(2.2, 22)
#ax.set_yscale("log", base = 10)
#ax.set_xscale("log", base = 10)
ax.legend(loc = "lower left",title = r'Anomality index $\alpha$')
#plt.yticks([np.e**-8,np.e**-6,np.e**-4,np.e**-2], labels = [-8, -6, -4, -2])
ax.set_yticks([1e-3, 1e-2, 1e-1])
ax.set_xticks([2.5, 5, 10, 20])
#ax.set_xticklabels([2.5, 5, 10, 20])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
#ax.axis["xzero"].set_axisline_style("-|>")
ax.plot((1), (np.e**-8), ">k", transform = ax.get_yaxis_transform(), clip_on = False)
ax.plot(2.2, 1, "^k", transform = ax.get_xaxis_transform(), clip_on = False)
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: f""))
ax.set_xlabel(r'Disturbance size')
ax.set_ylabel(r'Disturbance frequency')
fig.tight_layout()
plt.savefig("../plots/fig1b.png", dpi = 300, bbox_inches = "tight", transparent = True)

# %%
df = pd.read_csv("../data/share/ts_example.csv", index_col = 0)
df_gauss = pd.read_csv("../data/share/ts_example_gauss.csv", index_col = 0)
df_cauchy = pd.read_csv("../data/share/ts_example_cauchy.csv", index_col = 0)
df_pink = pd.read_csv("../data/share/ts_example_pink.csv", index_col = 0)
# %%
fig, axs = plt.subplots(3,1, sharex=False, figsize = (6,6), dpi = 300, gridspec_kw={'height_ratios': [0.7,1, 0.7]})


axs[0].axvline(x=1000, ymin =0, ymax = 1., color = "tab:purple", lw = 2., clip_on=False)
axs[0].axvline(x=1500, ymin = 0, ymax = 1., color = "tab:purple", lw = 2., clip_on=False)
df.Ps.plot(ax = axs[0], color = "tab:blue", lw = 1.5)
axs[0].axvline(x=df.ac1.dropna().index[-1], ymin = 0.0, ymax = 1., ls = "--", color = "black", lw = 1, clip_on=False)
axs[0].set_ylim(2, 5.5)
axs[0].set_yticks([2.5, 3, 4, 5])
axs[0].set_ylabel(r'Precipitation $P [\frac{mm}{day}]$')
axs[0].set_xlim(0, 2000)
axs[0].xaxis.set_ticks([])
axs[0].spines[["left", "bottom"]].set_position(('outward', 10))
axs[0].spines[["right", "top", "bottom"]].set_visible(False)

axs[1].axvline(x=1000, ymin =0, ymax = 1.18, color = "tab:purple", lw = 2., clip_on=False)
axs[1].axvline(x=1500, ymin = 0, ymax = 1.18, color = "tab:purple", lw = 2., clip_on=False)

df.Ts_0.plot(ax = axs[1], color = "tab:orange", lw = 1.)
df_cauchy.Ts_16.plot(ax = axs[1], color = "tab:red", lw = 1.)
df_pink.Ts_0.plot(ax = axs[1], color = "tab:pink", lw = 1.)
df_gauss.Ts_1.plot(ax = axs[1], color = "tab:brown", lw = 1.)
df.Ts_det.plot(ax = axs[1], color = "tab:gray", lw = 1., ls = "--")
axs[1].axvline(x=df.ac1.dropna().index[-1], ymin = 0.0, ymax = 1.18, ls = "--", color = "black", lw = 1, clip_on=False)
axs[1].set_ylim(30,85)
axs[1].set_yticks([30, 50, 70, 85])
axs[1].set_xlim(0, 2000)
axs[1].xaxis.set_ticks([])
axs[1].set_ylabel(r'Tree Cover $T [\%]$')
custom_lines = [ Line2D([0], [0], color="tab:brown", lw=2),
                Line2D([0], [0], color="tab:orange", lw=2),
                Line2D([0], [0], color="tab:red", lw=2),
                Line2D([0], [0], color="tab:pink", lw=2),
                Line2D([0], [0], color="tab:grey", lw=2, ls = "--"),
                ]
axs[1].legend(custom_lines, [r'Gauss $\alpha = 2.0$',r'Lévy $\alpha = 1.75$', r'Cauchy $\alpha = 1.0$', 'Pink', 'Deterministic'], loc = 'lower left')

axs[1].spines[["left", "bottom"]].set_position(('outward', 10))
axs[1].spines[["right", "top", "bottom"]].set_visible(False)

axs[2].axvline(x=1000, ymin = -0.087, ymax = 1.24, color = "tab:purple", lw = 2., clip_on=False)
axs[2].axvline(x=1500, ymin = -0.087, ymax = 1.24, color = "tab:purple", lw = 2., clip_on=False)
df.ac1.dropna().plot(ax = axs[2], color = "tab:orange", lw = 0.75)
df_cauchy.ac1.dropna().plot(ax = axs[2], color = "tab:red", lw = 0.75)
df_pink.ac1.dropna().plot(ax = axs[2], color = "tab:pink", lw = 0.75)
df_gauss.ac1.dropna().plot(ax = axs[2], color = "tab:brown", lw = 0.75)
axs[2].axvline(x=df.ac1.dropna().index[-1], ymin = -0.087, ymax = 1.24, ls = "--", color = "black", lw = 1, clip_on=False)

#axs[2].axvline(x=df.index[(df.Ts_0 <= 56.69).argmax()], ymin = -0.087, ymax = 1.18, ls = "--", color = "black", lw = 0.5, clip_on=False)
axs[2].set_ylim(-0.2, 1.0)
axs[2].set_ylabel(r'Rolling $AC(1)$')
axs[2].set_xlim(0, 2000)
axs[2].set_xlabel(r"$t$ in model years")
axs[2].spines[["right", "top"]].set_visible(False)
axs[2].spines[["left", "bottom"]].set_position(('outward', 10))
fig.align_ylabels(axs)
fig.tight_layout()
plt.savefig("../plots/fig1c.png", dpi = 300, bbox_inches = "tight", transparent = True)

# %%
df.loc[1000:].ac1.dropna().plot(color = "tab:orange", lw = 5.0)
plt.axis("off")
plt.savefig("../plots/fig1d_ts.png", dpi = 300, bbox_inches = "tight", transparent = True)

# %%
coefs = pd.read_csv("../data/coefs_t_climatechange=500-alpha=1.75-sigma=0.1-only_negative_disturbances=True-add_short_timescale=True.csv", index_col = 0)
# %%
sns.kdeplot(coefs.kendalltau_ac1_0[1:],fill = True)
plt.axvline(x = coefs.kendalltau_ac1_0[0], color = "tab:red")
plt.axis("off")
plt.savefig("../plots/fig1d_hist.png", dpi = 300, bbox_inches = "tight", transparent = True)
# %%

# %%
plt.plot(np.concatenate([[0.5, 5.0],cshr]), np.concatenate([[0.0, 0.0],xshr]))
# %%

plt.axhspan(ymin=69, ymax = 100,alpha = 0.2, color = "tab:green")
plt.axhspan(ymin=17, ymax = 69,alpha = 0.2, color = "tab:orange")
plt.axhspan(ymin=0, ymax = 17,alpha = 0.2, color = "tab:red")
df.plot(x = "P", y = "T", legend = False, ylabel = r'$T$', xlabel = r'$P$', xlim = (0,5), ylim = (-5,100), ax = plt.gca(), color = "black")

# %%
def func(x, c):
    return 0.3*(c)/(0.5+c)*x*(1-x/90) - 0.15*x*10/(x+10) - 0.11*x*64**7/(x**7+64**7) # %%
# %%
def find_rainforest_fixpoint():
    Ps = np.linspace(5.0, 2.9, 21001)
    Ts = np.zeros_like(Ps)
    for i in range(len(Ps)):
        Ts[i] = fsolve(func, x0 = np.array([90.]), args = np.array([Ps[i]]))[0]

    return Ps, Ts
# %%
x = 0.
x_new = 76.
dt = 0.1
cs = np.linspace(5.0, 0.0, 51)
xs = []
for c in cs:
    while not np.isclose(func(x_new, c), 0.0, atol = 0.001):
        x = x_new
        x_new = x + dt * func(x, c)
    xs.append(x_new)
x_new = 1
xs2 = []
for c in cs[::-1]:
    while not np.isclose(func(x_new, c), 0.0, atol = 0.001):
        x = x_new
        x_new = x + dt * func(x, c)
    xs2.append(x_new)

# %%

x_fix = 90.
cs = np.linspace(5.0, 0.0, 501)
x_fixs = []
all_cs = []
for c in cs:
    zeros = []
    for x0 in np.linspace(90., 1., 90):
        x_fix = fsolve(func, x0 = np.array([x0]), args = np.array([c]))[0]
        if not any([np.isclose(x_fix, z, atol = 0.1) for z in zeros]) and x_fix >= 0 and x_fix < 76:
            zeros.append(x_fix)
    all_cs += len(zeros)*[c]
    x_fixs += zeros
# %%
plt.scatter(all_cs, x_fixs)
# %%
