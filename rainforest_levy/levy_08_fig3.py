# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.patches as patches
# %%
recall = pd.read_csv("../data/share/recall.csv", index_col=0)
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
sns.heatmap(recall[recall.only_negative_disturbances & recall.add_short_timescale & (recall.ews == "iqr") & (recall.valid > 0.5)].pivot(index = "alpha", columns = "sigma", values = "p_kendalltau_adj"), vmin = 0.0, vmax = 1.0, cmap = "crest", ax = axs[0], cbar=False)
axs[0].text(-0.15, 1.0, "a)", transform=axs[0].transAxes, 
            size=14, weight='bold')
axs[0].set_ylabel(r'Anomaly index $\alpha$')
axs[0].set_xlabel(r'Noise amplitude $\sigma$')
axs[0].set_title("Our Model, IQR, Adj. Kendall-Tau slopes", fontdict={"fontsize":12})

xmin, xmax = axs[0].get_xlim()
ymin, ymax = axs[0].get_ylim()
xy = (xmin,ymin)
width = xmax - xmin
height = ymax - ymin

# create the patch and place it in the back of countourf (zorder!)
p = patches.Rectangle(xy, width, height, hatch='//', edgecolor="gray", fill=None, zorder=-10)
axs[0].add_patch(p)
axs[0].text(0.8, 0.85, 'strong noise\n regime', horizontalalignment='center', verticalalignment='center',fontsize=10, bbox=dict(facecolor='white', alpha=0.9, ), transform = axs[0].transAxes)

sns.heatmap(recall[recall.only_negative_disturbances & ~recall.add_short_timescale & (recall.ews == "std_dev") & (recall.valid > 0.5)].pivot(index = "alpha", columns = "sigma", values = "p_linear"), vmin = 0.0, vmax = 1.0, cmap = "crest", ax = axs[1], cbar=True, cbar_kws={"label": "Recall"})
axs[1].text(-0.15, 1.0, "b)", transform=axs[1].transAxes, 
            size=14, weight='bold')
axs[1].set_ylabel(r'Anomaly index $\alpha$')
axs[1].set_xlabel(r'Noise amplitude $\sigma$')
axs[1].set_title("Van Nes Model, Std. Dev., Linear slopes", fontdict={"fontsize":12})

xmin, xmax = axs[1].get_xlim()
ymin, ymax = axs[1].get_ylim()
xy = (xmin,ymin)
width = xmax - xmin
height = ymax - ymin

# create the patch and place it in the back of countourf (zorder!)
p = patches.Rectangle(xy, width, height, hatch='//', edgecolor="gray", fill=None, zorder=-10)
axs[1].add_patch(p)
axs[1].text(0.8, 0.85, 'strong noise\n regime', horizontalalignment='center', verticalalignment='center',fontsize=10, bbox=dict(facecolor='white', alpha=0.9, ), transform = axs[1].transAxes)

fig.tight_layout()
plt.savefig("../plots/fig3.pdf", dpi = 300, bbox_inches = "tight", transparent = True)
# %%
