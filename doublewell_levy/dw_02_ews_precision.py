
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
def compute_ews_from_df(df, window_years = 10, window_size = 100, downsample_factor_ews = 10, T_start = 1000, n_surrogates = 1000, verbose = True):

    downsample_factor = (window_years * 100)//window_size

    xs_det = df.xs_det

    x_barrier = 60.#0.0
    c_bifurcation = 1.
    c_only_rainforest = -1.
    idx_only_rainforest = np.where(df.cs >= c_only_rainforest)[0][0]
    
    n_seeds = len([c for c in df.columns if c.startswith("xs_") and c != "xs_det"])

    all_ews = []
    all_coefs = []
    all_pvals = []

    for seed in tqdm(range(n_seeds)) if verbose else range(n_seeds):

        xs_curr = df[f"xs_{seed}"]
    
        transition_idx = min(np.where(df.cs.iloc[idx_only_rainforest:] >= c_bifurcation)[0][0] + 5000, np.where(xs_curr.iloc[idx_only_rainforest:] <= x_barrier)[0][0] - 1000) + idx_only_rainforest

        xs_diff =  (xs_curr.iloc[:transition_idx] - xs_det.iloc[:transition_idx])[::downsample_factor]

        std_dev = xs_diff.rolling(window = window_size, min_periods = 10).std()

        ac1 = xs_diff.rolling(window = window_size, min_periods = 10).apply(lambda x: x.autocorr(), raw=False)

        iqr = xs_diff.rolling(window = window_size, min_periods = 10).quantile(0.75) - xs_diff.rolling(window = window_size).quantile(0.25) 

        ews = pd.DataFrame(data = {
                f"std_dev": std_dev.values,
                f"ac1": ac1.values,
                f"iqr": iqr.values
            }, index = std_dev.index).iloc[::downsample_factor_ews]

        all_ews.append({"seed": seed, "ews": ews})

        coefs = {}

        for ews_name, curr_ews in ews.items():

            ts = curr_ews.loc[T_start:]
            xrange = np.arange(len(ts))
            xrange = xrange[~np.isnan(ts)]
            ts = ts[~np.isnan(ts)]

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

            coefs[f"slope_{ews_name}_{seed}"] = surr_slope
            coefs[f"intercept_{ews_name}_{seed}"] = surr_intercept
            coefs[f"kendalltau_{ews_name}_{seed}"] = surr_kendalltau
            coefs[f"theilsenslope_{ews_name}_{seed}"] = surr_theilsenslope
            coefs[f"theilsenintercept_{ews_name}_{seed}"] = surr_theilsenintercept

            all_pvals.append({
                "ews": ews_name,
                "seed": seed,
                "p_linear": 1 - percentileofscore(surr_slope[1:], surr_slope[0]) / 100.,
                "p_kendalltau": 1 - percentileofscore(surr_kendalltau[1:], surr_kendalltau[0]) / 100.,
                "p_theilsen": 1 - percentileofscore(surr_theilsenslope[1:], surr_theilsenslope[0]) / 100.
            })

        all_coefs.append(pd.DataFrame(data = coefs, index = ["ts"] + [f"surr_{i}" for i in range(n_surrogates)]))

    #ews = pd.concat(all_ews, axis = 1)
    pvals = pd.DataFrame.from_records(all_pvals)
    coefs = pd.concat(all_coefs, axis = 1)

    return all_ews, pvals, coefs


# %%

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, help='simulation idx')
    args = parser.parse_args()
    
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    sigmas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    t_climatechange = 500
    
    sim_kwargs = [{"alpha": alpha, "sigma": sigma, "t_climatechange": t_climatechange, "only_negative_disturbances": only_negative_disturbances, "seeds": range(100)} for only_negative_disturbances in [False, True] for alpha in alphas for sigma in sigmas][args.idx]

    df = pd.read_csv(f"../data/dw_ts_20230717/ts_t_climatechange={t_climatechange}-alpha={sim_kwargs['alpha']}-sigma={sim_kwargs['sigma']}-only_negative_disturbances={sim_kwargs['only_negative_disturbances']}.csv", index_col = 0)
    #curr_df = df[["Ps", "Ts_det", "Tfixs", "Ts_0", "Ts_1"]]

    all_ews, pvals, coefs = compute_ews_from_df(df)

    for ews in all_ews:
        savepath = Path("../data/dw_ews_20230717/")/f"seed={ews['seed']}"
        savepath.mkdir(parents = True, exist_ok = True)
        ews['ews'].to_csv(savepath/f"ews_t_climatechange={t_climatechange}-alpha={sim_kwargs['alpha']}-sigma={sim_kwargs['sigma']}-only_negative_disturbances={sim_kwargs['only_negative_disturbances']}-seed={ews['seed']}.csv")
    
    pvals.to_csv(f"../data/dw_ews_20230717/pvals_t_climatechange={t_climatechange}-alpha={sim_kwargs['alpha']}-sigma={sim_kwargs['sigma']}-only_negative_disturbances={sim_kwargs['only_negative_disturbances']}.csv")

    coefs.to_csv(f"../data/dw_ews_20230717/coefs_t_climatechange={t_climatechange}-alpha={sim_kwargs['alpha']}-sigma={sim_kwargs['sigma']}-only_negative_disturbances={sim_kwargs['only_negative_disturbances']}.csv")