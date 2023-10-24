
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
            for add_short_timescale in [False, True]:
                #print("ts_t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma}-only_negative_disturbances={only_negative_disturbances}-add_short_timescale={add_short_timescale}.csv")
                df = pd.read_csv(f"../data/amaz_ts_20221202/ts_t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma}-only_negative_disturbances={only_negative_disturbances}-add_short_timescale={add_short_timescale}.csv", index_col = 0)

                T_barrier = 56.688667775
                T_barrier_desert = 5.
                P_bifurcation = 2.943
                P_only_rainforest = 4.414
                idx_only_rainforest = np.where(df.Ps <= P_only_rainforest)[0][0]

                for seed in tqdm(range(100), position=2, desc = "seed", leave=False):
                    Ts_curr = df[f"Ts_{seed}"]
                    try:
                        transition_idx = min(np.where(df.Ps.iloc[idx_only_rainforest:] <= P_bifurcation)[0][0], np.where(Ts_curr.iloc[idx_only_rainforest:] <= T_barrier)[0][0]) + idx_only_rainforest
                    
                    
                        T_transition = Ts_curr.index[transition_idx]

                        FPT = Ts_curr.index[np.where(Ts_curr <= T_barrier)[0][0]]

                    except IndexError:
                        T_transition = Ts_curr.index[idx_only_rainforest]
                        FPT = np.NaN
                    
                    try:
                        FPT_desert = Ts_curr.index[np.where(Ts_curr <= T_barrier_desert)[0][0]]
                    except IndexError:
                        FPT_desert = np.NaN
                    
                    max_T_after_steadystate = Ts_curr.loc[2*t_climatechange:].max()

                    first_T_after_steadstate = Ts_curr.loc[2*t_climatechange:].iloc[0]

                    fpts.append({
                        "alpha": alpha,
                        "sigma": sigma,
                        "only_negative_disturbances": only_negative_disturbances,
                        "add_short_timescale": add_short_timescale,
                        "seed": seed,
                        "FPT": FPT,
                        "T_transition": T_transition,
                        "FPT_desert": FPT_desert,
                        "max_T_after_steadystate": max_T_after_steadystate,
                        "first_T_after_steadstate": first_T_after_steadstate
                    })

FPTs = pd.DataFrame.from_records(fpts)

FPTs.to_csv("../data/amaz_ews_20221202/fpts.csv")
# %%
