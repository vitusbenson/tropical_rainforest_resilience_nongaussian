
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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import warnings

warnings.simplefilter('ignore', category=RuntimeWarning)

@jit(nopython = True)
def dxdt(x, c, tau):
    a = 20
    b = -3
    update = tau*(-(x/a + b)**3 + (x/a) + b - np.sqrt(4/27) * 4/np.sqrt(27) * c)/a
    #update = tau*(-x**3 + x - np.sqrt(4/27) * c)
    return update

@jit(nopython = True)
def new_x(x, c, L_gauss, L_levy, alpha = 2.0, sigma = 0.02, dt = 0.01, tau = 1):

    nextx = x + dxdt(x, c, tau) * dt + sigma * dt**(1/alpha) * L_levy + dt**0.5 * L_gauss

    if nextx < 0.0:
        nextx = 0.0
    elif nextx > 100.0:
        nextx = 100.0


    return nextx

# %%
#@jit(nopython = True)
def loop_x(xs, cs, L_gauss, L_levy, alpha = 2.0, sigma = 0.02, dt = 0.01, tau = 1):

    for i, c in enumerate(cs[:-1], 1):

        xs[i] = new_x(xs[i-1], c, L_gauss[i-1], L_levy[i-1], alpha = alpha, sigma = sigma, dt = dt, tau = tau)
    return xs
# %%
#@jit
def simulate_x(x_start = 83.5, c_start = -1.5, c_end = 1.5, t_steadystate = 1000, t_climatechange = 500, t_afterchange = 500, dt = 0.01, alpha = 2.0, sigma = 0.02, only_negative_disturbances = False, seeds = [42, 69, 420], tau = 1000):
    
    t_end = t_steadystate + t_climatechange + t_afterchange
    
    N_steps = int(t_end/dt)

    xs = np.zeros(N_steps+1)
    xs[0] = x_start

    cs = np.zeros(N_steps+1)
    cs[:int(t_steadystate/dt)+1] = c_start
    cs[int(t_steadystate/dt)+1:int((t_steadystate+t_climatechange)/dt)+1] = np.linspace(c_start, c_end, int(t_climatechange/dt))
    cs[int((t_steadystate+t_climatechange)/dt)+1:] = c_end

    
    Ls = np.zeros(N_steps)
    
    xs_det = loop_x(xs, cs, Ls, Ls, alpha = 2.0, sigma = 0.0, dt = dt, tau = tau).copy()

    all_xs = {}
    for seed in seeds:
        rng = np.random.default_rng(seed = seed)

        # if only_negative_disturbances and (alpha == 2.0):
        #     L_levy = np.sqrt(2) * rng.standard_normal(N_steps)
        #     L_levy[L_levy > 0] *= -1
        #     L_gauss = 0.01 * rng.standard_normal(N_steps)
        if only_negative_disturbances:
            L_levy = levy_stable.rvs(alpha, -1.0, size = N_steps, random_state = rng)
            L_gauss = 0.01 * rng.standard_normal(N_steps)
        elif alpha == 2.0:
            L_levy = np.sqrt(2) * rng.standard_normal(N_steps)
            L_gauss = np.zeros(N_steps)
        else:
            L_levy = levy_stable.rvs(alpha, 0.0, size = N_steps, random_state = rng)
            L_gauss = 0.01 * rng.standard_normal(N_steps)

        Ls = Ls.clip(-5., 5.)
        
        xs = np.zeros(N_steps+1)
        xs[0] = x_start
        xs = loop_x(xs, cs, L_gauss, L_levy, alpha = alpha, sigma = sigma, dt = dt, tau = tau).copy()
        all_xs[f'xs_{seed}'] = xs

    ts = np.zeros_like(xs)
    ts[1:] = np.linspace(0.0, t_end, N_steps)
    
    return ts, cs, xs_det, all_xs


def save_one_simulation(t_climatechange = 500, alpha = 2.0, sigma = 0.02, only_negative_disturbances = False, seeds = [42, 69, 420]):

    t_steadystate = 2 * t_climatechange
    t_afterchange = t_climatechange

    ts, cs, xs_det, all_xs = simulate_x(dt = 0.01, t_steadystate = t_steadystate, t_climatechange= t_climatechange, t_afterchange = t_afterchange, alpha = alpha, sigma = sigma, only_negative_disturbances = only_negative_disturbances, seeds = seeds)

    data = pd.DataFrame(data = {
        "cs": cs,
        "xs_det": xs_det,
        **all_xs
    }, index = ts)

    data.to_csv(f"../data/dw_ts_20230717/ts_t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma}-only_negative_disturbances={only_negative_disturbances}.csv")
# %%

def process_one(kwargs):
    save_one_simulation(**kwargs)

if __name__ == "__main__":


    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    sigmas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.5]
    ts_climatechange = [500]#, 10000]
    
    for t_climatechange in ts_climatechange:
        print(t_climatechange)
        sim_kwargs = [{"alpha": alpha, "sigma": sigma, "t_climatechange": t_climatechange, "only_negative_disturbances": only_negative_disturbances, "seeds": range(100)} for only_negative_disturbances in [False, True] for alpha in alphas for sigma in sigmas]#[142:]

        #for sim_kwarg in tqdm(sim_kwargs):
        #    process_one(sim_kwarg)
        with ProcessPoolExecutor(max_workers = 64) as pool:
            _ = list(tqdm(pool.map(process_one, sim_kwargs), total = len(sim_kwargs)))

    # for t_climatechange in ts_climatechange:
    #     for alpha in alphas:
    #         for sigma in sigmas:
    #             for only_negative_disturbances in [False, True]:
    #                 for add_short_timescale in [False, True]:
    #                     print(f"t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma:.4f}-only_negative_disturbances={only_negative_disturbances}-add_short_timescale={add_short_timescale}")
    #                     save_one_simulation(t_climatechange= t_climatechange, alpha = alpha, sigma = sigma, only_negative_disturbances= only_negative_disturbances, add_short_timescale= add_short_timescale, seeds = seeds)