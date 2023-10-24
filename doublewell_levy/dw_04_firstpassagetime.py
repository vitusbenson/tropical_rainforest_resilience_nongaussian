
# %%

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
# %%


alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
sigmas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.5]
t_climatechange = 500

fpts = []

for alpha in tqdm(alphas, position=0, desc = "alpha", leave=False):
    for sigma in tqdm(sigmas, position=1, desc = "sigma", leave=False):
        for only_negative_disturbances in [False, True]:
            
            #print("ts_t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma}-only_negative_disturbances={only_negative_disturbances}-add_short_timescale={add_short_timescale}.csv")
            df = pd.read_csv(f"../data/dw_ts_20230717/ts_t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma}-only_negative_disturbances={only_negative_disturbances}.csv", index_col = 0)

            x_barrier = 60.0
            c_bifurcation = 0.0
            c_only_rainforest = -1.
            idx_only_rainforest = np.where(df.cs >= c_only_rainforest)[0][0]

            for seed in tqdm(range(100), position=2, desc = "seed", leave=False):
                xs_curr = df[f"xs_{seed}"]
                try:
                    transition_idx = min(np.where(df.cs.iloc[idx_only_rainforest:] >= c_bifurcation)[0][0], np.where(xs_curr.iloc[idx_only_rainforest:] <= x_barrier)[0][0]) + idx_only_rainforest
                
                
                    x_transition = xs_curr.index[transition_idx]

                    FPT = xs_curr.index[np.where(xs_curr <= x_barrier)[0][0]]

                except IndexError:
                    x_transition = xs_curr.index[idx_only_rainforest]
                    FPT = np.NaN
                
                max_x_after_steadystate = xs_curr.loc[2*t_climatechange:].max()

                first_x_after_steadstate = xs_curr.loc[2*t_climatechange:].iloc[0]

                fpts.append({
                    "alpha": alpha,
                    "sigma": sigma,
                    "only_negative_disturbances": only_negative_disturbances,
                    "seed": seed,
                    "FPT": FPT,
                    "x_transition": x_transition,
                    "max_x_after_steadystate": max_x_after_steadystate,
                    "first_x_after_steadstate": first_x_after_steadstate
                })

FPTs = pd.DataFrame.from_records(fpts)

FPTs.to_csv("../data/dw_ews_20230717/fpts.csv")
# %%
