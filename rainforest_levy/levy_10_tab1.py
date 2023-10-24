# %%
import pandas as pd

# %%
recall = pd.read_csv("../data/recall.csv", index_col=0)
fpr = pd.read_csv("../data/falseposrate.csv", index_col=0)
# %%

configs = [
    ["vannes", "std_dev", "p_linear"],
    ["vannes", "iqr", "p_kendalltau"],
    ["ours", "iqr", "p_linear"],
    ["ours", "iqr", "p_theilsen"],
    ["ours", "iqr", "p_kendalltau"],
    ["ours", "std_dev", "p_kendalltau"],
    ["ours", "ac1", "p_kendalltau"],
    ["ours", "iqr", "p_kendalltau_adj"],
]

results = []
for potential, ews, slope in configs:
    vals = {"potential": potential, "ews": ews, "slope": slope}
    for curr_val, curr_df in zip(["recall", "fpr"], [recall, fpr]):

        curr_df = curr_df[(curr_df.valid > 0.5) & (curr_df.only_negative_disturbances)]

        # Van Nes vs Ours
        curr_df = curr_df[(curr_df.add_short_timescale == (potential == "ours"))]

        # Van Nes vs Ours
        curr_df = curr_df[(curr_df.ews == ews)]

        vals[curr_val] = curr_df[slope].mean()
    results.append(vals)


# %%
df = pd.DataFrame.from_records(results)
print(df.T.to_latex())
# %%
recall[(recall.valid > 0.5) & (recall.only_negative_disturbances)].melt(id_vars = ["alpha", "sigma", "add_short_timescale", "ews"], value_vars = ["p_linear", "p_kendalltau" ,"p_theilsen","p_kendalltau_adj"], var_name ="slopetype", value_name = "slope").groupby(["add_short_timescale", "ews", "slopetype"]).mean().sort_values("slope")



results = [{
        "recall": recall[(~recall.add_short_timescale) & (recall.ews == "std_dev") & (recall.valid > 0.5) & (recall.only_negative_disturbances)].p_linear.mean(),
        "fpr": fpr[(~fpr.add_short_timescale) & (fpr.ews == "std_dev") & (fpr.valid > 0.5) & (fpr.only_negative_disturbances)].p_linear.mean()
},
{
        "recall": recall[(~recall.add_short_timescale) & (recall.ews == "ac1") & (recall.valid > 0.5) & (recall.only_negative_disturbances)].p_kendalltau.mean(),
        "fpr": fpr[(~fpr.add_short_timescale) & (fpr.ews == "ac1") & (fpr.valid > 0.5) & (fpr.only_negative_disturbances)].p_kendalltau.mean()
}
,
{
        "recall": recall[(recall.add_short_timescale) & (recall.ews == "ac1") & (recall.valid > 0.5) & (recall.only_negative_disturbances)].p_kendalltau.mean(),
        "fpr": fpr[(~fpr.add_short_timescale) & (fpr.ews == "ac1") & (fpr.valid > 0.5) & (fpr.only_negative_disturbances)].p_kendalltau.mean()
}
]
# %%
