
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


def func(x, c):
    return 0.3*(c)/(0.5+c)*x*(1-x/90) - 0.15*x*10/(x+10) - 0.11*x*64**7/(x**7+64**7) 

@jit(nopython = True)
def dTdt(T, P, T_fix, add_short_timescale = True, h_p = 0.5, r_m = 0.3, k = 90, m_A = 0.15, h_A = 10, m_f = 0.11, h_f = 64, p = 7):
    update = (P / (h_p + P))* r_m * T * (1 - T/k) - m_A * T * (h_A / (T + h_A)) - m_f * T * (h_f**p / (h_f**p + T**p))
    if add_short_timescale:
        extra_width = 0.2 * np.exp(5 - P)
        update -= (0.5 * np.tanh((P - 3.) * 25) + 0.5) * (T - T_fix)/extra_width**3 * np.exp(- (T - T_fix)**2 / (2*extra_width**2))
    return update

# %%
@jit(nopython = True)
def new_T(T, P, L_gauss, L_levy, T_fix, add_short_timescale = True, alpha = 2.0, sigma = 0.02, dt = 0.01):
    nextT = T + dTdt(T, P, T_fix, add_short_timescale = add_short_timescale) * dt + sigma * dt**(1/alpha) * L_levy + dt**0.5 * L_gauss
    if nextT < 0.0:
        nextT = 0.0
    elif nextT > 100.0:
        nextT = 100.0
    return nextT

# %%
#@jit(nopython = True)
def loop_T(Ts, Ps, L_gauss, L_levy, Tfixs, add_short_timescale = True, alpha = 2.0, sigma = 0.02, dt = 0.01):
    # P_curr = None
    for i, (P, T_fix) in enumerate(zip(Ps[1:], Tfixs), 1):
        # if P != P_curr:
        #     T_fix = fsolve(func, x0 = np.array([80.]), args = np.array([P]))[0]
        Ts[i] = new_T(Ts[i-1], P, L_gauss[i-1], L_levy[i-1], T_fix, add_short_timescale = add_short_timescale, alpha = alpha, sigma = sigma, dt = dt)
    return Ts
# %%
#@jit
def simulate_T(T_start = 75.57915418, P_start = 5.0, P_end = 2.3, t_steadystate = 1000, t_climatechange = 500, t_afterchange = 500, dt = 0.01, alpha = 2.0, sigma = 0.02, only_negative_disturbances = False, add_short_timescale = True, seeds = [42, 69, 420]):
    
    t_end = t_steadystate + t_climatechange + t_afterchange
    
    N_steps = int(t_end/dt)

    Ts = np.zeros(N_steps+1)
    Ts[0] = T_start

    Ps = np.zeros(N_steps+1)
    Ps[:int(t_steadystate/dt)+1] = P_start
    Ps[int(t_steadystate/dt)+1:int((t_steadystate+t_climatechange)/dt)+1] = np.linspace(P_start, P_end, int(t_climatechange/dt))
    Ps[int((t_steadystate+t_climatechange)/dt)+1:] = P_end

    P_curr = None
    T_fix = 76.
    Tfixs = np.zeros_like(Ps)
    for i,P in enumerate(Ps):
        if P != P_curr:
            if T_fix > 55. and T_fix < 69.:
                T_fix = 44.
            T_fix = fsolve(func, x0 = np.array([T_fix]), args = np.array([P]))[0]
            P_curr = P
        Tfixs[i] = T_fix

    
    Ls = np.zeros(N_steps)
    
    Ts_det = loop_T(Ts, Ps, Ls, Ls, Tfixs, add_short_timescale = add_short_timescale, alpha = 2.0, sigma = 0.0, dt = dt).copy()

    all_Ts = {}
    for seed in seeds:
        rng = np.random.default_rng(seed = seed)

        if only_negative_disturbances:
            L_levy = levy_stable.rvs(alpha, -1.0, size = N_steps, random_state = rng)
            L_gauss = 0.01 * rng.standard_normal(N_steps)
        elif alpha == 2.0:
            L_levy = np.sqrt(2) * rng.standard_normal(N_steps)
            L_gauss = np.zeros(N_steps)
        else:
            L_levy = levy_stable.rvs(alpha, 0.0, size = N_steps, random_state = rng)
            L_gauss = 0.01 * rng.standard_normal(N_steps)

        Ls = Ls.clip(-30., 30.)
        
        Ts = np.zeros(N_steps+1)
        Ts[0] = T_start
        Ts = loop_T(Ts, Ps, L_gauss, L_levy, Tfixs, add_short_timescale = add_short_timescale, alpha = alpha, sigma = sigma, dt = dt).copy()
        all_Ts[f'Ts_{seed}'] = Ts

    ts = np.zeros_like(Ts)
    ts[1:] = np.linspace(0.0, t_end, N_steps)
    
    return ts, Ps, Tfixs, Ts_det, all_Ts


def save_one_simulation(t_climatechange = 500, alpha = 2.0, sigma = 0.02, only_negative_disturbances = False, add_short_timescale = True, seeds = [42, 69, 420]):

    t_steadystate = 2 * t_climatechange
    t_afterchange = t_climatechange

    ts, Ps, Tfixs, Ts_det, all_Ts = simulate_T(dt = 0.01, t_steadystate = t_steadystate, t_climatechange= t_climatechange, t_afterchange = t_afterchange, alpha = alpha, sigma = sigma, only_negative_disturbances = only_negative_disturbances, add_short_timescale = add_short_timescale,  seeds = seeds)

    data = pd.DataFrame(data = {
        "Ps": Ps,
        "Ts_det": Ts_det,
        "Tfixs": Tfixs,
        **all_Ts
    }, index = ts)

    data.to_csv(f"../data/amaz_ts_20221202/ts_t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma}-only_negative_disturbances={only_negative_disturbances}-add_short_timescale={add_short_timescale}.csv")
# %%

def process_one(kwargs):
    save_one_simulation(**kwargs)

if __name__ == "__main__":


    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    sigmas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.5]
    ts_climatechange = [500, 10000]
    
    for t_climatechange in ts_climatechange:
        print(t_climatechange)
        sim_kwargs = [{"alpha": alpha, "sigma": sigma, "t_climatechange": t_climatechange, "only_negative_disturbances": only_negative_disturbances, "add_short_timescale": add_short_timescale, "seeds": range(100)} for only_negative_disturbances in [False, True] for add_short_timescale in [False, True] for alpha in alphas for sigma in sigmas]

        with ProcessPoolExecutor() as pool:
            _ = list(tqdm(pool.map(process_one, sim_kwargs), total = len(sim_kwargs)))

    # for t_climatechange in ts_climatechange:
    #     for alpha in alphas:
    #         for sigma in sigmas:
    #             for only_negative_disturbances in [False, True]:
    #                 for add_short_timescale in [False, True]:
    #                     print(f"t_climatechange={t_climatechange}-alpha={alpha}-sigma={sigma:.4f}-only_negative_disturbances={only_negative_disturbances}-add_short_timescale={add_short_timescale}")
    #                     save_one_simulation(t_climatechange= t_climatechange, alpha = alpha, sigma = sigma, only_negative_disturbances= only_negative_disturbances, add_short_timescale= add_short_timescale, seeds = seeds)