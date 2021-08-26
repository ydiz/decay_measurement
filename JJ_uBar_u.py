import os
import numpy as np
import pickle
from util import mean_err, Z_V, M_pi, N_pi

data = {}
all_trajs = sorted([int(f[:4]) for f in os.listdir('./Kaon_results/')])
for traj in all_trajs:
    data[traj] = pickle.load(open(f'./Kaon_results/{traj}.pkl', 'rb'))


def calc_JJ_uBar_u_one_traj(traj):  # calculate < J J | pi> for one trajectory
    
    JJ_pion_hadron_coef = Z_V**2 * 2 * M_pi / N_pi
#     diagram_coeffient = 1 / (3 * np.sqrt(2))  # for <J J|pi^0>
    diagram_coeffient = 4 / 9  # 4/9 for <J J|uBar g5 u>; 1/9 for <J J|dBar g5 d> # factor 2 is multiplied in C++
    JJ_uBar_u = data[traj]['JJ_pion'].real * JJ_pion_hadron_coef * diagram_coeffient
    return JJ_uBar_u   # (tsep, T)

def calc_JJ_uBar_u_multiple_trajs(trajs): 

    JJ_uBar_u = []
    for traj in trajs:
        rst_one_traj = calc_JJ_uBar_u_one_traj(traj)   # shape (tsep, T)
        JJ_uBar_u.append(rst_one_traj)
    JJ_uBar_u = np.array(JJ_uBar_u)  
    return JJ_uBar_u         # (trajs, tsep, T)

def calc_JJ_uBar_u_sum_tu(trajs, MAX_ut=10): # MAX_ut should be determined from the plot of amp v.s. ut
    
    JJ_uBar_u = calc_JJ_uBar_u_multiple_trajs(trajs) # (trajs, tsep, T)
    
    # 1. sum over t_u
    JJ_uBar_u = JJ_uBar_u[..., :MAX_ut].sum(axis=-1)  # (trajs, tsep)
    
    # 2. average over trajs
    JJ_uBar_u_avg, JJ_uBar_u_err = mean_err(JJ_uBar_u)  # (tsep, )
    
    return JJ_uBar_u_avg, JJ_uBar_u_err
 
# weighted average over t_pi.   Weight is proportional to 1/sigma^2
def calc_JJ_uBar_u_avg_tpi_sum_tu(trajs, MAX_ut=10):
    
    JJ_uBar_u_avg, JJ_uBar_u_err = calc_JJ_uBar_u_sum_tu(trajs, MAX_ut) # (tsep, )
    
    weights = 1 / JJ_uBar_u_err**2
#     print(weights)
    JJ_uBar_u_avg = np.average(JJ_uBar_u_avg, weights=weights)  # average over t_pi with weights
    return JJ_uBar_u_avg
    
