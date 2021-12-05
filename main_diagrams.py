import os
import numpy as np
import pickle
from util import mean_err, T, V, Z_V, M_K, N_K, M_pi, N_pi, C1, C2

typeI_typeII_t_seps = [6, 8, 10, 12, 14]    # t_x - t_K
typeIII_IV_V_tsep_max = 20

# Only typeIII and typeIV_D1 involve single pion intermediate state

coefficients = {
#     'typeI_D1aQ1': -2/9, 'typeI_D1aQ2': -2/9, 'typeI_D1bQ1': -2/9, 'typeI_D1bQ2': -2/9, 
#     'typeI_D2aQ1': 2/9, 'typeI_D2aQ2': 2/9, 'typeI_D2bQ1': 2/9, 'typeI_D2bQ2': 2/9, 

#     'typeII_D1aQ1': -1/9, 'typeII_D1aQ2': 1/9, 'typeII_D1bQ1': -1/9, 'typeII_D1bQ2': 1/9,  

#     'typeIII_D1aQ1': 4/9, 'typeIII_D1aQ2': -4/9, 'typeIII_D1bQ1': 4/9, 'typeIII_D1bQ2': -4/9, 

#     'typeIV_D1aQ1': -1/9, 'typeIV_D1aQ2': 1/9, 
#     'typeIV_D1bQ1': -1/9, 'typeIV_D1bQ2': 1/9,

#     'typeIV_D2aQ1': -1/9, 'typeIV_D2aQ2': 1/9, 
#     'typeIV_D2bQ1': -1/9, 'typeIV_D2bQ2': 1/9,

    'typeV_D1Q1': 5/9, 'typeV_D1Q2': -5/9,  # disconnected # Error is too large (about 100% - 200%) with 35 configurations # add this later
    'typeV_D2Q1': 1/9, 'typeV_D2Q2': -1/9,

    'sBar_d_T1D1a': -1/9, 'sBar_d_T1D1b': -1/9, 
    'sBar_d_T2D1a': 1/9, 'sBar_d_T2D1b': 1/9, 'sBar_d_T2D2a': 1/9, 'sBar_d_T2D2b': 1/9,
#     'sBar_d_T3D1': -5/9, 'sBar_d_T3D2': -1/9,   # disconnected
}

diagrams = list(coefficients.keys())

Q1_diagrams = [d for d in diagrams if d.endswith('Q1')] 
Q2_diagrams = [d for d in diagrams if d.endswith('Q2')] 
sBar_d_diagrams = [d for d in diagrams if d.startswith('sBar_d')]

print(f'Q1_diagrams: {Q1_diagrams}')
print(f'Q2_diagrams: {Q2_diagrams}')
print(f'sBar_d_diagrams: {sBar_d_diagrams}')

data = {}
all_trajs = sorted([int(f[:4]) for f in os.listdir('./Kaon_results/')])
for traj in all_trajs:
    data[traj] = pickle.load(open(f'./Kaon_results/{traj}.pkl', 'rb'))

# p.s. zyd: for now, I am using simple average, rather than weighted average over tK.
# calculate the weight of each tsep for each diagram; the weights are used to average over tK
weights_tsep = {} 
for d in diagrams:
    amp = np.zeros((len(all_trajs), data[traj][d].shape[0]))  # (trajs, tseps)
    tv_min = -4
    tv_max = 3
    for i, traj in enumerate(all_trajs):
        amp[i] = data[traj][d][:, tv_min:].sum(axis=-1).real + data[traj][d][:, :tv_max+1].sum(axis=-1).real
        
    avg, err = mean_err(amp)
        
    weights_tsep[d] = 1 / err**2

def get_amp(traj, d): # d is diagram
    
    if data[traj][d].shape[0] == len(typeI_typeII_t_seps): # type I adn type II
        rst = data[traj][d]
    elif data[traj][d].shape[0] == typeIII_IV_V_tsep_max: # type III, IV, V
        rst = data[traj][d][typeI_typeII_t_seps]
    else: 
        assert False, "Number of tseps is different than expected"
    return rst  # shape:  len(typeI_typeII_t_seps, 64)

def get_amp_avg_tK(traj, d, tsep_min=6, tsep_max=14): # d is diagram
    
    if data[traj][d].shape[0] == len(typeI_typeII_t_seps): # type I and type II
        rst = data[traj][d]                 # (tsep, tv)
        weights = weights_tsep[d]           # (tsep, )
    elif data[traj][d].shape[0] == typeIII_IV_V_tsep_max: # type III, IV, V
        rst = data[traj][d][tsep_min:tsep_max+1]
        weights = weights_tsep[d][tsep_min:tsep_max+1]  # (tsep, )
    else: 
        assert False, "Number of tseps is different than expected"

    rst = rst.mean(axis=0)   # simple average over t_K  
#     rst = np.average(rst, weights=weights, axis=0)    # Uncomment me to use weighted average over t_K  # jackknife error is a little bigger compared to simple average; jackknife mean is also a little more different than experimental value
    return rst  # (tv, )

def calc_amp_one_traj(traj, avg_tK):
    
    hadron_coef = Z_V**2 * 2 * M_K / N_K
    if avg_tK:
        amp_Q1 = sum([get_amp_avg_tK(traj, d) * coefficients[d] * hadron_coef for d in Q1_diagrams])
        amp_Q2 = sum([get_amp_avg_tK(traj, d) * coefficients[d] * hadron_coef for d in Q2_diagrams])
    else:
        amp_Q1 = sum([get_amp(traj, d) * coefficients[d] * hadron_coef for d in Q1_diagrams])
        amp_Q2 = sum([get_amp(traj, d) * coefficients[d] * hadron_coef for d in Q2_diagrams])
    amp_Q1 = amp_Q1.real # take real part
    amp_Q2 = amp_Q2.real 
    
    return amp_Q1, amp_Q2 

def calc_amp_multiple_trajs(trajs, avg_tK):
    amp_Q1, amp_Q2 = [], []
    for traj in trajs:
        amp_Q1_one_traj, amp_Q2_one_traj = calc_amp_one_traj(traj, avg_tK)   # shape (tsep, tx)
        amp_Q1.append(amp_Q1_one_traj)
        amp_Q2.append(amp_Q2_one_traj)
    amp_Q1 = np.array(amp_Q1)  # (trajs, tsep, tx)
    amp_Q2 = np.array(amp_Q2)

    amp_Hw = 1. / np.sqrt(2) * (C1 * amp_Q1 + C2 * amp_Q2) # 1. / np.sqrt(2) is the factor in K_L
    return amp_Hw


def calc_amp_sum_tv(trajs, tv_min = -4, tv_max=3): # tv_max is inclusive!  # for jackknife
    amp_Hw = calc_amp_multiple_trajs(trajs, avg_tK=True)  # (trajs, tv)
    amp_Hw = amp_Hw[:, tv_min:].sum(axis=-1) + amp_Hw[:, :tv_max+1].sum(axis=-1)  # sum over tv # (trajs,)
    amp_Hw_avg = amp_Hw.mean(axis=0) # average over trajs
    return amp_Hw_avg     # amp_Hw_avg is a single number
    
