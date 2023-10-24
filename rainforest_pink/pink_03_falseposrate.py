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
def fourrier_surrogates(ts, ns, rng = np.random.default_rng()):
    # From https://github.com/niklasboers/AMOC_EWS/blob/main/EWS_functions.py
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(rng.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts.squeeze()



# %%
def compute_falseposrate_from_df(ews_path, ews_years = 50, ews_steps = 50, window_years = 10, window_size = 100, downsample_factor_ews = 10, T_end = 1000, n_surrogates = 1000, verbose = True, n_seeds = 100):

    all_coefs = []
    all_pvals = []

    for seed in tqdm(range(n_seeds)) if verbose else range(n_seeds):
        
        ews = pd.read_csv(ews_path.replace("seed=0",f"seed={seed}"), index_col = 0)

        coefs = {}

        for ews_name, curr_ews in ews.items():
            
            for start_year in range(0, T_end-ews_years, ews_steps):
                try:
                    ts = curr_ews.loc[start_year:start_year+ews_years]
                    xrange = np.arange(len(ts))
                    xrange = xrange[~np.isnan(ts)]
                    ts = ts[~np.isnan(ts)]

                    if len(ts) < 2:
                        continue

                    tsc = ts - ts.mean()

                    ts_slope, ts_intercept = np.polyfit(xrange,tsc,deg=1)
                    ts_kendalltau, _ = kendalltau(xrange, tsc)
                    ts_theilsenslope, ts_theilsenintercept, _, _ = theilslopes(tsc, xrange)

                    tsf = fourrier_surrogates(tsc, n_surrogates, rng = np.random.default_rng(seed * n_surrogates))

                    xrange = np.arange(tsf.shape[1])

                    surr_slope, surr_intercept = np.polyfit(xrange,tsf.T,deg=1)

                    surr_slope = np.concatenate([[ts_slope], surr_slope])
                    surr_intercept = np.concatenate([[ts_intercept], surr_intercept])

                    surr_kendalltau = np.full_like(surr_slope, np.NaN)
                    surr_kendalltau[0] = ts_kendalltau
                    surr_theilsenslope = np.full_like(surr_slope, np.NaN)
                    surr_theilsenslope[0] = ts_theilsenslope
                    surr_theilsenintercept = np.full_like(surr_slope, np.NaN)
                    surr_theilsenintercept[0] = ts_theilsenintercept
                    for i in range(n_surrogates):
                        surr_kendalltau[i+1], _ = kendalltau(xrange, tsf[i])
                        surr_theilsenslope[i+1], surr_theilsenintercept[i+1], _, _ = theilslopes(tsf[i], xrange)

                    coefs[f"slope_{ews_name}_{seed}_{start_year}"] = surr_slope
                    coefs[f"intercept_{ews_name}_{seed}_{start_year}"] = surr_intercept
                    coefs[f"kendalltau_{ews_name}_{seed}_{start_year}"] = surr_kendalltau
                    coefs[f"theilsenslope_{ews_name}_{seed}_{start_year}"] = surr_theilsenslope
                    coefs[f"theilsenintercept_{ews_name}_{seed}_{start_year}"] = surr_theilsenintercept

                    all_pvals.append({
                        "ews": ews_name,
                        "seed": seed,
                        "start_year": start_year,
                        "p_linear": 1 - percentileofscore(surr_slope[1:], surr_slope[0]) / 100.,
                        "p_kendalltau": 1 - percentileofscore(surr_kendalltau[1:], surr_kendalltau[0]) / 100.,
                        "p_theilsen": 1 - percentileofscore(surr_theilsenslope[1:], surr_theilsenslope[0]) / 100.
                    })
                except:
                    print(f"start year {start_year}, seed {seed} does not work for {ews_path}")
                    continue

        all_coefs.append(pd.DataFrame(data = coefs, index = ["ts"] + [f"surr_{i}" for i in range(n_surrogates)]))

    pvals = pd.DataFrame.from_records(all_pvals)
    coefs = pd.concat(all_coefs, axis = 1)

    return pvals, coefs

# %%
#pvals, coefs = compute_falseposrate_from_df("../data/amaz_ews_20221202/seed=0/ews_t_climatechange=500-alpha=2.0-sigma=0.02-only_negative_disturbances=True-add_short_timescale=True-seed=0.csv")
# %%


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, help='simulation idx')
    args = parser.parse_args()
    
    alphas = [2.0]
    sigmas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.5]
    #ts_climatechange = [500, 10000]
    
    t_climatechange = 500
    
    sim_kwargs = [{"alpha": alpha, "sigma": sigma, "t_climatechange": t_climatechange, "only_negative_disturbances": only_negative_disturbances, "add_short_timescale": add_short_timescale, "seeds": range(100)} for only_negative_disturbances in [False, True] for add_short_timescale in [False, True] for alpha in alphas for sigma in sigmas][args.idx]

    ews_path = f"../data/amazpink_ews_20230712/seed=0/ews_t_climatechange={t_climatechange}-alpha={sim_kwargs['alpha']}-sigma={sim_kwargs['sigma']}-only_negative_disturbances={sim_kwargs['only_negative_disturbances']}-add_short_timescale={sim_kwargs['add_short_timescale']}-seed=0.csv"
    #curr_df = df[["Ps", "Ts_det", "Tfixs", "Ts_0", "Ts_1"]]

    pvals, coefs = compute_falseposrate_from_df(ews_path, n_seeds=100)

    
    pvals.to_csv(f"../data/amazpink_ews_falseposrate_20230712/pvals_t_climatechange={t_climatechange}-alpha={sim_kwargs['alpha']}-sigma={sim_kwargs['sigma']}-only_negative_disturbances={sim_kwargs['only_negative_disturbances']}-add_short_timescale={sim_kwargs['add_short_timescale']}.csv")

    coefs.to_csv(f"../data/amazpink_ews_falseposrate_20230712/coefs_t_climatechange={t_climatechange}-alpha={sim_kwargs['alpha']}-sigma={sim_kwargs['sigma']}-only_negative_disturbances={sim_kwargs['only_negative_disturbances']}-add_short_timescale={sim_kwargs['add_short_timescale']}.csv")