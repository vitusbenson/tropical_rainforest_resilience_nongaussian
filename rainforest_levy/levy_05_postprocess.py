# %%
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes, percentileofscore
from pathlib import Path
from tqdm import tqdm

# %%
pvals = pvals = pd.read_csv("../data/amaz_ews_20221202/pvals2.csv", index_col= 0)

# %%
fpts = pd.read_csv("../data/amaz_ews_20221202/fpts.csv", index_col = 0)

# %%
fpts = fpts.set_index(["alpha", "sigma", "seed", "only_negative_disturbances", "add_short_timescale"])

valid_runs = fpts.FPT >= fpts.T_transition

pvals = pvals.set_index(["alpha", "sigma", "seed", "only_negative_disturbances", "add_short_timescale", "ews"])

pvals_valid = pvals.join(valid_runs.rename("valid"))

pvals_plot = pvals_valid[pvals_valid.valid].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "add_short_timescale", "ews"]).apply(lambda x: (x < 0.05).mean())[["p_linear", "p_kendalltau", "p_theilsen","p_kendalltau_adj"]].join(pvals_valid["valid"].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "add_short_timescale", "ews"]).mean()["valid"]).reset_index()

# %%
pvals_plot.to_csv("../data/share/recall.csv")



pd.read_csv("../data/amaz_ews_falseposrate_20230106/pvals2.csv", index_col= 0)

# %%
fpts = pd.read_csv("../data/amaz_ews_20221202/fpts.csv", index_col = 0)

# %%
fpts = fpts.set_index(["alpha", "sigma", "seed", "only_negative_disturbances", "add_short_timescale"])

valid_runs = fpts.FPT >= fpts.T_transition

pvals = pvals.set_index(["alpha", "sigma", "seed", "only_negative_disturbances", "add_short_timescale", "ews"])

pvals_valid = pvals.join(valid_runs.rename("valid"))

pvals_plot = pvals_valid[pvals_valid.valid].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "add_short_timescale", "ews"]).apply(lambda x: (x < 0.05).mean())[["p_linear", "p_kendalltau", "p_theilsen","p_kendalltau_adj"]].join(pvals_valid["valid"].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "add_short_timescale", "ews"]).mean()["valid"]).reset_index()

# %%
pvals_plot.to_csv("../data/share/falseposrate.csv")


# %%

ts = pd.read_csv("../data/amaz_ts_20221202/ts_t_climatechange=500-alpha=1.75-sigma=0.1-only_negative_disturbances=True-add_short_timescale=True.csv", index_col = 0)

ews = pd.read_csv("../data/amaz_ews_20221202/seed=0/ews_t_climatechange=500-alpha=1.75-sigma=0.1-only_negative_disturbances=True-add_short_timescale=True-seed=0.csv", index_col = 0)


ts[["Ps", "Ts_det", "Tfixs", "Ts_0"]].join(ews).to_csv("../data/share/ts_example.csv")