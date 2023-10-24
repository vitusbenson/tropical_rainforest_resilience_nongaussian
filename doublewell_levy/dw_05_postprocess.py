# %%
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes, percentileofscore
from pathlib import Path
from tqdm import tqdm

def adjust_pval_row(row):

    sigma = [s for s in [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.5] if s == row['sigma']][0]

    ews_path = f"../data/dw_ews_20230717/seed={row['seed']}/ews_t_climatechange={row['t_climatechange']}-alpha={row['alpha']}-sigma={sigma}-only_negative_disturbances={row['only_negative_disturbances']}-seed={row['seed']}.csv"

    ews = pd.read_csv(ews_path, index_col = 0)

    if ('start_year' in row) and (ews[row['start_year']: row['start_year']+50]).ac1.max() < 0.5:
        row['p_kendalltau_adj'] = 2
    elif ('start_year' not in row) and  (ews[1000:]).ac1.max() < 0.5:
        row['p_kendalltau_adj'] = 2
    else:
        row['p_kendalltau_adj'] = row['p_kendalltau']

    return row

# %%
if __name__ == "__main__":
    # print("FPR")

    # pvals = pd.read_csv("../data/amaz_ews_falseposrate_20230106/pvals.csv", index_col= 0)

    # pvals2 = pvals[pvals.p_kendalltau < 0.05].apply(adjust_pval_row, axis = 1)

    # pvals['p_kendalltau_adj'] = pvals['p_kendalltau']
    # pvals3 = pd.concat([pvals2, pvals[(pvals.p_kendalltau >= 0.05)]])

    # pvals3.to_csv("../data/amaz_ews_falseposrate_20230106/pvals2.csv")

    # %%
    
    all_pvals = []
    for path in Path("../data/dw_ews_20230717/").glob("pvals*.csv"):
        if len(path.name) <= 10:
            continue
        t_climatechange = 500#int(path.name.split("=")[1].split("-")[0])
        alpha = float(path.name.split("=")[2].split("-")[0])
        sigma = float(path.name.split("=")[3].split("-only")[0])
        only_negative_disturbances = (path.name.split("=")[4].split(".")[0]) == "True"
        curr_pvals = pd.read_csv(path, index_col = 0)
        curr_pvals["alpha"] = alpha
        curr_pvals["sigma"] = sigma
        curr_pvals["t_climatechange"] = t_climatechange
        curr_pvals["only_negative_disturbances"] = only_negative_disturbances
        all_pvals.append(curr_pvals)

    pvals = pd.concat(all_pvals, axis = 0)
    pvals.reset_index(drop = True).to_csv("../data/dw_ews_20230717/pvals.csv")



    print("Recall")
    pvals = pd.read_csv("../data/dw_ews_20230717/pvals.csv", index_col= 0)

    pvals2 = pvals[pvals.p_kendalltau < 0.05].apply(adjust_pval_row, axis = 1)

    pvals['p_kendalltau_adj'] = pvals['p_kendalltau']
    pvals3 = pd.concat([pvals2, pvals[(pvals.p_kendalltau >= 0.05)]])

    pvals3.to_csv("../data/dw_ews_20230717/pvals2.csv")


    # %%
    pvals = pd.read_csv("../data/dw_ews_20230717/pvals2.csv", index_col= 0)

    # %%
    fpts = pd.read_csv("../data/dw_ews_20230717/fpts.csv", index_col = 0)

    # %%
    fpts = fpts.set_index(["alpha", "sigma", "seed", "only_negative_disturbances"])

    valid_runs = fpts.FPT >= fpts.x_transition

    pvals = pvals.set_index(["alpha", "sigma", "seed", "only_negative_disturbances", "ews"])

    pvals_valid = pvals.join(valid_runs.rename("valid"))

    pvals_plot = pvals_valid[pvals_valid.valid].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "ews"]).apply(lambda x: (x < 0.05).mean())[["p_linear", "p_kendalltau", "p_theilsen","p_kendalltau_adj"]].join(pvals_valid["valid"].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "ews"]).mean()["valid"]).reset_index()

    # %%
    pvals_plot.to_csv("../data/share/recall_dw.csv")


    if True:
        # %%
        all_pvals = []
        for path in Path("../data/dw_ews_falseposrate_20230717/").glob("pvals*.csv"):
            if len(path.name) <= 10:
                continue
            t_climatechange = 500#int(path.name.split("=")[1].split("-")[0])
            alpha = float(path.name.split("=")[2].split("-")[0])
            sigma = float(path.name.split("=")[3].split("-only")[0])
            only_negative_disturbances = (path.name.split("=")[4].split(".")[0]) == "True"
            curr_pvals = pd.read_csv(path, index_col = 0)
            curr_pvals["alpha"] = alpha
            curr_pvals["sigma"] = sigma
            curr_pvals["t_climatechange"] = t_climatechange
            curr_pvals["only_negative_disturbances"] = only_negative_disturbances
            all_pvals.append(curr_pvals)

        pvals = pd.concat(all_pvals, axis = 0)
        pvals.reset_index(drop = True).to_csv("../data/dw_ews_falseposrate_20230717/pvals.csv")

        print("FPR")
        pvals = pd.read_csv("../data/dw_ews_falseposrate_20230717/pvals.csv", index_col= 0)

        pvals2 = pvals[pvals.p_kendalltau < 0.05].apply(adjust_pval_row, axis = 1)

        pvals['p_kendalltau_adj'] = pvals['p_kendalltau']
        pvals3 = pd.concat([pvals2, pvals[(pvals.p_kendalltau >= 0.05)]])

        pvals3.to_csv("../data/dw_ews_falseposrate_20230717/pvals2.csv")



        # %%
        pvals = pd.read_csv("../data/dw_ews_falseposrate_20230717/pvals2.csv", index_col= 0)

        # %%
        fpts = pd.read_csv("../data/dw_ews_20230717/fpts.csv", index_col = 0)

        # %%
        fpts = fpts.set_index(["alpha", "sigma", "seed", "only_negative_disturbances"])

        valid_runs = fpts.FPT >= fpts.x_transition

        pvals = pvals.set_index(["alpha", "sigma", "seed", "only_negative_disturbances", "ews"])

        pvals_valid = pvals.join(valid_runs.rename("valid"))

        pvals_plot = pvals_valid[pvals_valid.valid].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "ews"]).apply(lambda x: (x < 0.05).mean())[["p_linear", "p_kendalltau", "p_theilsen","p_kendalltau_adj"]].join(pvals_valid["valid"].reset_index().groupby(["alpha", "sigma", "only_negative_disturbances", "ews"]).mean()["valid"]).reset_index()

        # %%
        pvals_plot.to_csv("../data/share/falseposrate_dw.csv")
    # %%
