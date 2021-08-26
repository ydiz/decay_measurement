import os
import numpy as np
import pickle
from util import mean_err, T, V, M_K, N_K, M_pi, N_pi, C1, C2

t_seps = [10, 12, 14, 16, 18, 20, 22, 24] # For pion_Hw_K, use tseps=[18,20,22,24]; only plot these tseps

# assume tK=0; return e^{M_K (t_x - t_K)  + M_pi (t_pi - t_x)} for every t_x; if not tK<t_x<t_pi, return 0
def exp_factor(t_pi): 
    tx = np.arange(T)
    rst = np.exp(M_K * tx + M_pi * (t_pi - tx))
    rst = np.where(tx>=t_pi, 0, rst)
    return rst

coefficients = { 
    'cs_Hw_T1D1Q1': 1, 'cs_Hw_T1D1Q2': -1,  # type I is for uBar g5 u
    'cs_Hw_T2D1Q1': 1, 'cs_Hw_T2D1Q2': -1,  # type II is for dBar g5 d
}

data = {}
all_trajs = sorted([int(f[:4]) for f in os.listdir('./Kaon_results/')])
for traj in all_trajs:
    data[traj] = pickle.load(open(f'./Kaon_results/{traj}.pkl', 'rb'))


def calc_qBar_q_Q_K_one_traj(traj): # calculate <uBar g5  u | Qi | K> and <dBar g5  d | Qi | K> for one trajectory
    
    uBar_u_Q1_K = data[traj]['cs_Hw_T1D1Q1'] * coefficients['cs_Hw_T1D1Q1'] # shape (tseps, t_x)
    uBar_u_Q2_K = data[traj]['cs_Hw_T1D1Q2'] * coefficients['cs_Hw_T1D1Q2']
    dBar_d_Q1_K = data[traj]['cs_Hw_T2D1Q1'] * coefficients['cs_Hw_T2D1Q1']
    dBar_d_Q2_K = data[traj]['cs_Hw_T2D1Q2'] * coefficients['cs_Hw_T2D1Q2']
    
    for i, t_sep in enumerate(t_seps):
        uBar_u_Q1_K[i] *= exp_factor(t_sep)
        uBar_u_Q2_K[i] *= exp_factor(t_sep)
        dBar_d_Q1_K[i] *= exp_factor(t_sep)
        dBar_d_Q2_K[i] *= exp_factor(t_sep)
        
    pi_Hw_K_hadron_coef = 2 * M_K/N_K * 2 * M_pi/N_pi / V

    uBar_u_Q1_K *= pi_Hw_K_hadron_coef   #  uBar_u_Q1_K = <uBar g5 u | Q1(0) |K0>
    uBar_u_Q2_K *= pi_Hw_K_hadron_coef  
    dBar_d_Q1_K *= pi_Hw_K_hadron_coef   
    dBar_d_Q2_K *= pi_Hw_K_hadron_coef 

    uBar_u_Q1_K = uBar_u_Q1_K.real 
    uBar_u_Q2_K = uBar_u_Q2_K.real 
    dBar_d_Q1_K = dBar_d_Q1_K.real 
    dBar_d_Q2_K = dBar_d_Q2_K.real 
    return uBar_u_Q1_K, uBar_u_Q2_K, dBar_d_Q1_K, dBar_d_Q2_K


def calc_qBar_q_Hw_K_multiple_trajs(trajs):

    uBar_u_Q1_K, uBar_u_Q2_K, dBar_d_Q1_K, dBar_d_Q2_K = [], [], [], []
    for traj in trajs:
        uBar_u_Q1_K_one_traj, uBar_u_Q2_K_one_traj,\
        dBar_d_Q1_K_one_traj, dBar_d_Q2_K_one_traj = calc_qBar_q_Q_K_one_traj(traj)   # shape (tsep, tx)

        uBar_u_Q1_K.append(uBar_u_Q1_K_one_traj)
        uBar_u_Q2_K.append(uBar_u_Q2_K_one_traj)
        dBar_d_Q1_K.append(dBar_d_Q1_K_one_traj)
        dBar_d_Q2_K.append(dBar_d_Q2_K_one_traj)

    uBar_u_Q1_K = np.array(uBar_u_Q1_K)  # # (trajs, tsep, tx)
    uBar_u_Q2_K = np.array(uBar_u_Q2_K)
    dBar_d_Q1_K = np.array(dBar_d_Q1_K)  # # (trajs, tsep, tx)
    dBar_d_Q2_K = np.array(dBar_d_Q2_K)
    

    # 1/sqrt(2) is the factor in K_L
    uBar_u_Hw_K = 1. / np.sqrt(2) * (C1 * uBar_u_Q1_K + C2 * uBar_u_Q2_K)
    dBar_d_Hw_K = 1. / np.sqrt(2) * (C1 * dBar_d_Q1_K + C2 * dBar_d_Q2_K)
    
    return uBar_u_Hw_K, dBar_d_Hw_K
    


def calc_qBar_q_Hw_K_avg_tsep_tx(trajs, min_tsep=18, min_tx=8, max_tx=15): # min_tx, max_tx should be determined from the plot of amp v.s. tx
    
    uBar_u_Hw_K, dBar_d_Hw_K = calc_qBar_q_Hw_K_multiple_trajs(trajs) # (trajs, tsep, T)
    
    # 1. average over tx
    uBar_u_Hw_K = uBar_u_Hw_K[..., min_tx:max_tx+1].mean(axis=-1)   # (trajs, tsep)
    dBar_d_Hw_K = dBar_d_Hw_K[..., min_tx:max_tx+1].mean(axis=-1)
    
     # 2. average over trajs to calculate mean and err
    uBar_u_Hw_K_avg, uBar_u_Hw_K_err = mean_err(uBar_u_Hw_K)     # (tsep,)
    dBar_d_Hw_K_avg, dBar_d_Hw_K_err = mean_err(dBar_d_Hw_K)    
    
    # 3. average over tsep
    min_tsep_idx = 0  # only keep the tseps that satisfy tsep >= min_tsep
    while t_seps[min_tsep_idx] < min_tsep:
        min_tsep_idx += 1
    uBar_u_Hw_K_avg, uBar_u_Hw_K_err = uBar_u_Hw_K_avg[min_tsep_idx:], uBar_u_Hw_K_err[min_tsep_idx:]
    dBar_d_Hw_K_avg, dBar_d_Hw_K_err = dBar_d_Hw_K_avg[min_tsep_idx:], dBar_d_Hw_K_err[min_tsep_idx:]
    
    weights = 1 / uBar_u_Hw_K_err**2
#     print(weights)
    uBar_u_Hw_K_avg = np.average(uBar_u_Hw_K_avg, weights=weights)
    dBar_d_Hw_K_avg = np.average(dBar_d_Hw_K_avg, weights=weights)

    return uBar_u_Hw_K_avg, dBar_d_Hw_K_avg
