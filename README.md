# Code for Benson et. al., ArXiv (2023) - Measuring tropical rainforest resilience under non-Gaussian disturbances

Preprint: https://arxiv.org/abs/2310.16021

## Instructions

Run the scripts in the subfolders in their respective order to reproduce the figures in the paper.

Note: The scripts will create roughly 2 TB of data in the folder `data/`, so make sure you have enough disk space. If you just want to reproduce figures, we provide postprocessed data under `data/share`.

Note: scripts `02_falseposrate` and `03_ews_precision` require a lot of compute. Thus, they are split into different noise parameter settings and the script takes a command line argument, that specifies the index of the specific setting to run. For example:
```
python rainforest_levy/levy_02_ews_precision.py 0
```

You may wish to create a SLURM script to run all of these simulations as a batch job.


## Conda Environment

```
conda create -n amazews python=3.10
conda activate amazews
conda install -c conda-forge numpy scipy ipykernel seaborn matplotlib pandas tqdm numba
```
