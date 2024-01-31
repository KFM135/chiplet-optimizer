import numpy as np
import math
import json
from itertools import permutations, combinations
from numpy.random import rand
from numpy.random import uniform
from numpy.random import randn
from numpy import exp
import random
import datetime
from numpy.random import Generator
import sys
import matplotlib.pyplot as plt

def factor_pairs(number):
    pairs = [(number / n, n) for n in range(1, number + 1) if number % n == 0]
    diff = [abs(pairs[i][0] - pairs[i][1]) for i in range(0, len(pairs))]
    idx = diff.index(min(diff))
    return pairs[idx]

def LATENCY(x_dim, y_dim, hbm_idx):
    latencyAI2AI = x_dim + y_dim - 2
    latencyAI2HBM_init = 1
    latencyAI2AI_init = 1
    latencyAI2HBM = 0

    positions = ['left', 'right', 'top', 'bottom', 'middle', '3D-stacked']
    hbm_loc = []
    for i in range(1, len(positions) + 1):
        hbm_loc.append(list(combinations(positions, i)))
    idx = hbm_idx
    loc = [e for b in hbm_loc for e in b]
    loc_dict = {i: k for i, k in enumerate(loc)}

    if len(loc_dict[idx]) in [1, 6] and '3D-stacked' in loc_dict[idx]:
        latencyAI2HBM = 1  # considering that there are 5 HBM chiplets, all are 3D-stacked
        hbm_num = 5
        if (x_dim % 2 == 0) and (y_dim % 2 == 0):
            latencyAI2HBM = max(x_dim, y_dim) / 2
        else:
            latencyAI2HBM = math.ceil(max(x_dim, y_dim) / 2) - 1

    else:
        hbm_num = len(loc_dict[idx]) - 1 if '3D-stacked' in loc_dict[idx] else len(loc_dict[idx])
        # no. of hbm = 1
        if hbm_num == 1:
            tpl_con = [a for a in loc_dict[idx]]
            if 'left' in tpl_con or 'right' in tpl_con:
                # x + floor (y/2)
                latencyAI2HBM = x_dim + math.floor(y_dim / 2)
            if 'top' in tpl_con or 'bottom' in tpl_con:
                # floor(x/2) + y
                latencyAI2HBM = math.floor(x_dim / 2) + y_dim
            if 'middle' in tpl_con:
                # ceil(x/2) + ceil (y/2) - 1
                latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2) - 1

        # no. of hbm = 2
        if hbm_num == 2:
            tpl_con = [a for a in loc_dict[idx]]
            if '3D-stacked' in tpl_con:
                if 'left' in tpl_con or 'right' in tpl_con:
                    # x + floor (y/2) - 1
                    latencyAI2HBM = x_dim + math.floor(y_dim / 2) - 1
                if 'top' in tpl_con or 'bottom' in tpl_con:
                    # floor(x/2) + y
                    latencyAI2HBM = math.floor(x_dim / 2) + y_dim
                if 'middle' in tpl_con:
                    if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                        latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2)
                    elif (x_dim % 2 != 0) and (y_dim % 2 != 0):
                        latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2) - 2
                    else:
                        latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2) - 1
            else:
                if 'left' in tpl_con and 'right' in tpl_con:
                    # ceil(x/2) + floor(y/2)
                    latencyAI2HBM = math.ceil(x_dim / 2) + math.floor(y_dim / 2)
                elif 'top' in tpl_con and 'bottom' in tpl_con:
                    # floor(x/2) + ceil(y/2)
                    latencyAI2HBM = math.floor(x_dim / 2) + math.ceil(y_dim / 2)
                elif 'middle' in tpl_con:
                    # ceil(x/2) + ceil(y/2) -1
                    latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2) - 1
                else:
                    latencyAI2HBM = min(x_dim, y_dim) + math.floor(max(x_dim, y_dim) / 2)

        # no. of HBM = 3
        if hbm_num == 3:
            tpl_con = [a for a in loc_dict[idx]]
            # 3D-stacked conditions
            if '3D-stacked' in tpl_con:
                if 'left' in tpl_con and 'right' in tpl_con:
                    # ceil(x/2) + floor(y/2) -1
                    latencyAI2HBM = math.ceil(x_dim / 2) + math.floor(y_dim / 2) - 1
                elif 'top' in tpl_con and 'bottom' in tpl_con:
                    # floor(x/2) + ceil(y/2) - 1
                    latencyAI2HBM = math.floor(x_dim / 2) + math.ceil(y_dim / 2) - 1
                elif 'middle' in tpl_con:
                    # give even odd conditions
                    if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                        # latency = ceil(x/2) + ceil(y/2)
                        latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2)
                    elif (x_dim % 2 != 0) and (y_dim % 2 != 0):
                        # latency = ceil(x/2) + ceil(y/2) -2
                        latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2) - 2
                    else:
                        latencyAI2HBM = math.ceil(x_dim / 2) + math.ceil(y_dim / 2) - 1

                else:
                    # min(x,y) + floor (max(x,y)/2) - 1
                    latencyAI2HBM = min(x_dim, y_dim) + math.floor(max(x_dim, y_dim))
            else:
                # 2.5D conditions
                if 'middle' in tpl_con:
                    if ('left' in tpl_con and 'right' in tpl_con) or ('top' in tpl_con and 'bottom' in tpl_con):
                        # floor(x/2+y/2) -1
                        latencyAI2HBM = math.floor(x_dim / 2 + y_dim / 2) - 1
                    else:
                        if (x_dim % 2 == 0 and y_dim % 2 == 0):
                            # latency = min(x,y)-1
                            latencyAI2HBM = min(x_dim, y_dim) - 1
                        else:
                            # latency = min(x,y)
                            latencyAI2HBM = min(x_dim, y_dim)

                else:
                    latencyAI2HBM = math.floor(x_dim / 2 + y_dim / 2)

        ## no of cpu = 4
        if hbm_num == 4:
            tpl_con = [a for a in loc_dict[idx]]
            if '3D-stacked' in tpl_con:
                if 'middle' in tpl_con:
                    if ('left' in tpl_con and 'right' in tpl_con) or ('top' in tpl_con and 'bottom' in tpl_con):
                        ##condition here
                        if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                            # latency = floor(x/2+y/2) -2
                            latencyAI2HBM = math.floor(x_dim / 2 + y_dim / 2) - 2
                        else:
                            latencyAI2HBM = math.floor(x_dim / 2 + y_dim / 2) - 1
                            # floor(x/2+y/2)-1
                    else:
                        if (x_dim % 2 != 0) and (y_dim % 2 != 0):
                            latencyAI2HBM = min(x_dim, y_dim) - 1
                            # min(x,y)-1
                        else:
                            latencyAI2HBM = min(x_dim, y_dim)
                else:
                    # without middle conditions here
                    latencyAI2HBM = math.floor(x_dim / 2 + y_dim / 2) - 1

            else:
                ##2.5D conditions
                if 'left' in tpl_con and 'right' in tpl_con and 'top' in tpl_con and 'bottom' in tpl_con:
                    if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                        latencyAI2HBM = max(x_dim, y_dim) / 2 + 1
                    else:
                        latencyAI2HBM = math.ceil(max(x_dim, y_dim) / 2)
                else:
                    if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                        latencyAI2HBM = max(x_dim, y_dim) / 2
                    else:
                        latencyAI2HBM = math.ceil(max(x_dim, y_dim) / 2)

        ## no of hbm = 5
        if hbm_num == 5:
            tpl_con = [a for a in loc_dict[idx]]
            if '3D-stacked' in tpl_con:
                if 'middle' in tpl_con:
                    if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                        latencyAI2HBM = max(x_dim, y_dim) / 2 + 1
                    elif (x_dim % 2 != 0) and (y_dim % 2 != 0):
                        latencyAI2HBM = math.ceil(max(x_dim, y_dim) / 2) - 1
                    else:
                        latencyAI2HBM = math.ceil(max(x_dim, y_dim) / 2)
                else:
                    if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                        latencyAI2HBM = max(x_dim, y_dim) / 2
                    else:
                        latencyAI2HBM = math.ceil(max(x_dim, y_dim) / 2) - 1

            else:
                if (x_dim % 2 == 0) and (y_dim % 2 == 0):
                    latencyAI2HBM = max(x_dim, y_dim) / 2 + 1
                else:
                    latencyAI2HBM = math.ceil(max(x_dim, y_dim) / 2)

    # print(f'Dictionary location:{json.dumps(loc_dict, indent=4)}')
    latencyAI2HBM = max(latencyAI2HBM_init, latencyAI2HBM)
    latencyAI2AI = max(latencyAI2AI_init, latencyAI2AI)
    return latencyAI2AI, latencyAI2HBM, loc_dict

def action_refined(action_base):
    arch_type = action_base[0]
    num_chiplet = int(action_base[1])
    chiplet_pairs = num_chiplet // 2
    ai2ai_intrcnct_2p5D, ai2ai_dr_2p5D, ai2ai_pd_2p5D, ai2ai_tr_2p5D, ai2ai_intrcnct_3D, ai2ai_dr_3D, ai2ai_pd_3D,\
        ai2ai_tr_3D, ai2hbm_intrcnct, ai2hbm_dr, ai2hbm_pd, ai2hbm_tr = action_base[3:]

    num_chiplet = 1 if num_chiplet == 0 else num_chiplet
    chiplet_pairs = num_chiplet // 2 if num_chiplet > 1 else 1
    ai2ai_dr_2p5D = 1 if ai2ai_dr_2p5D == 0 else ai2ai_dr_2p5D
    ai2ai_pd_2p5D = 1 if ai2ai_pd_2p5D == 0 else ai2ai_pd_2p5D
    ai2ai_tr_2p5D = 1 if ai2ai_tr_2p5D == 0 else ai2ai_tr_2p5D
    ai2ai_dr_3D = 1 if ai2ai_dr_3D == 0 else ai2ai_dr_3D
    ai2ai_pd_3D = 1 if ai2ai_pd_3D == 0 else ai2ai_pd_3D
    ai2ai_tr_3D = 1 if ai2ai_tr_3D == 0 else ai2ai_tr_3D
    ai2hbm_dr = 1 if ai2hbm_dr == 0 else ai2hbm_dr
    ai2hbm_pd = 1 if ai2hbm_pd == 0 else ai2hbm_pd
    ai2hbm_tr = 1 if ai2hbm_tr == 0 else ai2hbm_tr

    # arch type =  2p5D
    if arch_type == 0:
        if ai2ai_intrcnct_2p5D > 1: ai2ai_intrcnct_2p5D = 1
        if ai2ai_dr_2p5D > 20: ai2ai_dr_2p5D = 20
        if ai2hbm_intrcnct > 1: ai2hbm_intrcnct = 1
        if ai2hbm_dr > 20: ai2hbm_dr = 20
        ai2ai_intrcnct_3D, ai2ai_dr_3D, ai2ai_pd_3D, ai2ai_tr_3D = np.zeros(4)

    # arch type = 5p5D_MoL
    if arch_type == 1:
        if ai2ai_intrcnct_2p5D > 1: ai2ai_intrcnct_2p5D = 1
        if ai2ai_dr_2p5D > 20: ai2ai_dr_2p5D = 20
        if ai2hbm_intrcnct < 2: ai2hbm_intrcnct = 2
        if ai2hbm_dr < 10: ai2hbm_dr = 10
        ai2ai_intrcnct_3D, ai2ai_dr_3D, ai2ai_pd_3D, ai2ai_tr_3D  = np.zeros(4)
        ai2hbm_tr = 1

    # arch type = 5p5D_LoL
    if arch_type == 2:
        # num_chiplet = chiplet_pairs
        if ai2ai_intrcnct_2p5D > 1: ai2ai_intrcnct_2p5D = 1
        if ai2ai_dr_2p5D > 20: ai2ai_dr_2p5D = 20
        if ai2hbm_intrcnct > 1: ai2hbm_intrcnct = 1
        if ai2hbm_dr > 20: ai2hbm_dr = 20
        if ai2ai_intrcnct_3D < 2: ai2ai_intrcnct_3D = 2
        if ai2ai_dr_3D < 10: ai2ai_dr_3D = 10
        ai2ai_tr_3D = 1

    action_new = [arch_type, num_chiplet, action_base[2], ai2ai_intrcnct_2p5D, ai2ai_dr_2p5D, ai2ai_pd_2p5D, ai2ai_tr_2p5D, ai2ai_intrcnct_3D, ai2ai_dr_3D, ai2ai_pd_3D, ai2ai_tr_3D, ai2hbm_intrcnct, ai2hbm_dr,
                  ai2hbm_pd, ai2hbm_tr]

    return action_new

def ENERGY(tr_len, energy_min, energy_max):
    tr_len_min = 0.5
    tr_len_max = 10
    slope = (energy_max - energy_min) / (tr_len_max - tr_len_min)
    constant = energy_min - (slope * tr_len_min)
    energy = slope * tr_len + constant
    rdl_min = 2
    rdl_max = 14
    slope_rdl = (rdl_min - rdl_max) / (tr_len_max - tr_len_min)
    constant_rdl = rdl_max -(slope_rdl * tr_len_min)
    rdl_count = slope_rdl * tr_len + constant_rdl
    return energy, rdl_count

def energy_cost(action):
    arch_type = action[0]
    num_chiplet = int(action[1])
    chiplet_pairs = num_chiplet // 2
    ai2ai_intrcnct_2p5D, ai2ai_dr_2p5D, ai2ai_pd_2p5D, ai2ai_tr_2p5D, ai2ai_intrcnct_3D, ai2ai_dr_3D, ai2ai_pd_3D,\
        ai2ai_tr_3D, ai2hbm_intrcnct, ai2hbm_dr, ai2hbm_pd, ai2hbm_tr= action[3:]
    comm_energy = 0
    rdl_ai_2p5D = 0
    rdl_ai_3D = 0
    rdl_hbm = 0
    energy_min_ai = 0
    energy_max_ai = 0
    energy_min_hbm = 0
    energy_max_hbm = 0
    energy_min_ai_2p5D = 0
    energy_max_ai_2p5D = 0
    energy_min_ai_3D = 0
    energy_max_ai_3D = 0
    #
    ### only 2p5D
    if arch_type ==  0:
        if ai2ai_intrcnct_2p5D == 0:       # CoWoS
            energy_min_ai = 0.2
            energy_max_ai = 0.6
        elif ai2ai_intrcnct_2p5D == 1:     # EMIB
            energy_min_ai = 0.17
            energy_max_ai = 0.7

        if ai2hbm_intrcnct == 0:       # CoWoS
            energy_min_hbm = 0.2
            energy_max_hbm = 0.6
        elif ai2hbm_intrcnct == 1:     # EMIB
            energy_min_hbm = 0.1
            energy_max_hbm = 0.5

        ai2ai_energy_per_bit_2p5D, rdl_ai_2p5D = ENERGY(ai2ai_tr_2p5D, energy_min_ai, energy_max_ai)
        ai2hbm_energy_per_bit, rdl_hbm = ENERGY(ai2hbm_tr, energy_min_hbm, energy_max_hbm)
        ai2ai_energy_tot = ai2ai_energy_per_bit_2p5D * 1e-12 * ai2ai_dr_2p5D * 1e9 * ai2ai_pd_2p5D * num_chiplet   #todo: not very sure about this model, think about it later on
        ai2hbm_energy_tot = ai2hbm_energy_per_bit * 1e-12 * ai2hbm_dr * 1e9 * ai2hbm_pd * num_chiplet
        comm_energy = ai2ai_energy_tot + ai2hbm_energy_tot

    # 5p5D_MoL
    if arch_type == 1:
        if ai2ai_intrcnct_2p5D == 0:       # CoWoS
            energy_min_ai = 0.2
            energy_max_ai = 0.6
        elif ai2ai_intrcnct_2p5D == 1:     # EMIB
            energy_min_ai = 0.17
            energy_max_ai = 0.7

        if ai2hbm_intrcnct == 2:       # SoIC
            energy_min_hbm = 0.1
            energy_max_hbm = 0.2
        elif ai2hbm_intrcnct == 3:     # FOVEROS
            energy_min_hbm = 0.01
            energy_max_hbm = 0.05

        ai2ai_energy_per_bit_2p5D, rdl_ai_2p5D = ENERGY(ai2ai_tr_2p5D, energy_min_ai, energy_max_ai)
        ai2hbm_energy_per_bit, rdl_hbm = ENERGY(ai2hbm_tr, energy_min_hbm, energy_max_hbm)
        ai2ai_energy_tot = ai2ai_energy_per_bit_2p5D * 1e-12 * ai2ai_dr_2p5D * 1e9 * ai2ai_pd_2p5D * num_chiplet   #todo: not very sure about this model, think about it later on
        ai2hbm_energy_tot = ai2hbm_energy_per_bit * 1e-12 * ai2hbm_dr * 1e9 * ai2hbm_pd * num_chiplet
        comm_energy = ai2ai_energy_tot + ai2hbm_energy_tot

    # 5p5D_LoL
    if arch_type == 2:
        if ai2ai_intrcnct_2p5D == 0:  # CoWoS
            energy_min_ai_2p5D = 0.2
            energy_max_ai_2p5D = 0.6
        elif ai2ai_intrcnct_2p5D == 1:  # EMIB
            energy_min_ai_2p5D = 0.17
            energy_max_ai_2p5D = 0.7

        if ai2hbm_intrcnct == 0:  # CoWoS
            energy_min_hbm = 0.2
            energy_max_hbm = 0.6
        elif ai2hbm_intrcnct == 1:  #EMIB
            energy_min_hbm = 0.17
            energy_max_hbm = 0.7

        if ai2ai_intrcnct_3D == 2:  # SoIC
            energy_min_ai_3D = 0.1
            energy_max_ai_3D = 0.2
        elif ai2ai_intrcnct_3D == 3:
            energy_min_ai_3D = 0.01
            energy_max_ai_3D = 0.02


        ai2ai_energy_per_bit_2p5D, rdl_ai_2p5D = ENERGY(ai2ai_tr_2p5D, energy_min_ai_2p5D, energy_max_ai_2p5D)
        ai2ai_energy_per_bit_3D, rdl_ai_3D = ENERGY(ai2ai_tr_3D, energy_min_ai_3D, energy_max_ai_3D)
        ai2hbm_energy_per_bit, rdl_hbm = ENERGY(ai2hbm_tr, energy_min_hbm, energy_max_hbm)
        ai2ai_energy_tot_2p5D = ai2ai_energy_per_bit_2p5D * 1e-12 * ai2ai_dr_2p5D * 1e9 * ai2ai_pd_2p5D * chiplet_pairs  # todo: not very sure about this model, think about it later on
        ai2ai_energy_tot_3D = ai2ai_energy_per_bit_3D * 1e-12 * ai2ai_dr_3D * 1e9 * ai2ai_pd_3D * chiplet_pairs
        ai2ai_energy_tot = ai2ai_energy_tot_2p5D + ai2ai_energy_tot_3D
        ai2hbm_energy_tot = ai2hbm_energy_per_bit * 1e-12 * ai2hbm_dr * 1e9 * ai2hbm_pd * num_chiplet
        comm_energy = ai2ai_energy_tot + ai2hbm_energy_tot

    return comm_energy, round(rdl_ai_2p5D), round(rdl_ai_3D), round(rdl_hbm)


def cost(base_cost, pin_count, rdl_count):
    return base_cost + pin_count + rdl_count

def package_cost(action):
    base_cost = 1
    arch_type = action[0]
    num_chiplet = int(action[1])
    chiplet_pairs = num_chiplet // 2
    ai2ai_intrcnct_2p5D, ai2ai_dr_2p5D, ai2ai_pd_2p5D, ai2ai_tr_2p5D, ai2ai_intrcnct_3D, ai2ai_dr_3D, ai2ai_pd_3D, \
    ai2ai_tr_3D, ai2hbm_intrcnct, ai2hbm_dr, ai2hbm_pd, ai2hbm_tr = action[3:]
    cost_tot = 0
    base_cost_ai_2p5D = 0
    base_cost_hbm = 0
    base_cost_ai_3D = 0
    # arch type 2p5D
    if arch_type == 0:
        if ai2ai_intrcnct_2p5D == 0:  # CoWoS
            base_cost_ai_2p5D = 3
        elif ai2ai_intrcnct_2p5D == 1:  # EMIB
            base_cost_ai_2p5D = 1

        if ai2hbm_intrcnct == 0:  # CoWoS
            base_cost_hbm = 3
        elif ai2hbm_intrcnct == 1:  # EMIB
            base_cost_hbm = 1

        _, rdl_ai_2p5D, _, rdl_hbm = energy_cost(action)
        ai2ai_cost = cost(base_cost_ai_2p5D, ai2ai_pd_2p5D, rdl_ai_2p5D)
        ai2hbm_cost = cost(base_cost_hbm, ai2hbm_pd, rdl_hbm)
        cost_tot = ai2ai_cost + ai2hbm_cost

    # arch type 5p5D_MoL
    if arch_type == 1:
        if ai2ai_intrcnct_2p5D == 0:  # CoWoS
            base_cost_ai_2p5D = 3
        elif ai2ai_intrcnct_2p5D == 1:  # EMIB
            base_cost_ai_2p5D = 1

        if ai2hbm_intrcnct == 2:  # SoIC
            base_cost_hbm = 6
        elif ai2hbm_intrcnct == 3:  # FOVEROS
            base_cost_hbm = 8

        _, rdl_ai_2p5D, _, rdl_hbm = energy_cost(action)
        ai2ai_cost = cost(base_cost_ai_2p5D, ai2ai_pd_2p5D, rdl_ai_2p5D)
        ai2hbm_cost = cost(base_cost_hbm, ai2hbm_pd, rdl_hbm)
        cost_tot = ai2ai_cost + ai2hbm_cost

    # arch type 5p5D_LoL
    if arch_type == 2:
        if ai2ai_intrcnct_2p5D == 0:  # CoWoS
            base_cost_ai_2p5D = 3
        elif ai2ai_intrcnct_2p5D == 1:  # EMIB
            base_cost_ai_2p5D = 1

        if ai2hbm_intrcnct == 0:  # CoWoS
            base_cost_hbm = 3
        elif ai2hbm_intrcnct == 1:  # EMIB
            base_cost_hbm = 1

        if ai2ai_intrcnct_3D == 2:  # SoIC
            base_cost_ai_3D = 6
        elif ai2ai_intrcnct_3D == 3: #FOVEROS
            base_cost_ai_3D = 8

        _, rdl_ai_2p5D, rdl_ai_3D, rdl_hbm = energy_cost(action)
        # print(f'rdl_ai:{rdl_ai}, rdl_hbm:{rdl_hbm}')
        ai2ai_cost_2p5D = cost(base_cost_ai_2p5D, ai2ai_pd_2p5D, rdl_ai_2p5D)
        ai2ai_cost_3D = cost(base_cost_ai_3D, ai2ai_pd_3D, rdl_ai_3D)
        ai2hbm_cost = cost(base_cost_hbm, ai2hbm_pd, rdl_hbm)
        cost_tot_ai = ai2ai_cost_3D + ai2ai_cost_2p5D
        cost_tot = cost_tot_ai + ai2hbm_cost

    return cost_tot

def throughput(action):
    arch_type = action[0]
    num_chiplet = int(action[1])
    chiplet_pairs = num_chiplet // 2 if num_chiplet > 1 else 1
    hbm_loc = action[2]
    ai2ai_intrcnct_2p5D, ai2ai_dr_2p5D, ai2ai_pd_2p5D, ai2ai_tr_2p5D, ai2ai_intrcnct_3D, ai2ai_dr_3D, ai2ai_pd_3D, \
    ai2ai_tr_3D, ai2hbm_intrcnct, ai2hbm_dr, ai2hbm_pd, ai2hbm_tr = action[3:]
    # Initialization
    throughput_achieved = 0
    area_per_chiplet = 0
    ai2ai_lat = 0
    ai2hbm_lat = 0
    cost_tot = 0
    energy_tot = 0
    ai_bw_penalty_2p5D = 0
    area_penalty = 0

    # arch type 2p5D
    if arch_type == 0:
        if ai2ai_dr_2p5D > 20: ai2ai_dr_2p5D = 20    # this is already taken care of in action_refined function. But still having it here, becasue RL does not work with action_refined
        if ai2hbm_dr > 20: ai2hbm_dr = 20

        x, y = factor_pairs(num_chiplet)
        ai2ai_lat, ai2hbm_lat, _ = LATENCY(x, y, hbm_loc)
        scale_factor = 50
        ai2ai_bw_act_2p5D = ai2ai_dr_2p5D * 10**-3 * ai2ai_pd_2p5D * scale_factor
        ai2hbm_bw_act = ai2hbm_dr * 10**-3 * ai2hbm_pd * scale_factor      ## 10^-3 is because giga is converted to tera
        ai_bw_penalty = 1
        hbm_bw_penalty = 1
        area_per_chiplet = 800 / num_chiplet
        ## area penalty based on constraint of 400mm2
        if area_per_chiplet >= 400:
            area_penalty = area_per_chiplet - 400

        area_for_mac = area_per_chiplet * 0.4
        no_of_macs = area_for_mac / 0.003
        thruput_per_chiplet = no_of_macs * 2 * 10 ** -3  # in TOPS (assuming clk freq of 1GHz)
        ai2hbm_bw_req = 4 * 4 * thruput_per_chiplet * 32    # one 4 is for 4 neighboring chiplets and one 4 is for number of operands, it could be 2
        ai2ai_bw_req_2p5D = 4 * 1 * thruput_per_chiplet * 32
        if num_chiplet < 4:
            ai2hbm_bw_req = 4 * num_chiplet * thruput_per_chiplet * 32

        # todo: start from here, mainly reward calculation
        throughput_max = num_chiplet * thruput_per_chiplet
        lat_penalty = ai2ai_lat + ai2hbm_lat
        if ai2ai_bw_act_2p5D < ai2ai_bw_req_2p5D:
            ai_bw_penalty_2p5D = ai2ai_bw_req_2p5D / ai2ai_bw_act_2p5D
        if ai2hbm_bw_act < ai2hbm_bw_req:
            hbm_bw_penalty = ai2hbm_bw_req / ai2hbm_bw_act
        bw_penalty = ai_bw_penalty + hbm_bw_penalty
        cost_tot = package_cost(action)
        energy_tot, _, _, _ = energy_cost(action)
        throughput_achieved = throughput_max - (lat_penalty + bw_penalty + 0.1 * cost_tot + energy_tot + area_penalty)

    # arch type 2p5D
    if arch_type == 1:
        if ai2ai_dr_2p5D > 20: ai2ai_dr_2p5D = 20
        if ai2hbm_dr < 10: ai2hbm_dr = 10

        x, y = factor_pairs(num_chiplet)
        ai2ai_lat, ai2hbm_lat, _ = LATENCY(x, y, hbm_loc)
        scale_factor = 50
        scale_factor_hbm = 100
        ai2ai_bw_act_2p5D = ai2ai_dr_2p5D * 10 ** -3 * ai2ai_pd_2p5D * scale_factor
        ai2hbm_bw_act = ai2hbm_dr * 10 ** -3 * ai2hbm_pd * scale_factor_hbm  ## 10^-3 is because giga is converted to tera
        ai_bw_penalty = 1
        hbm_bw_penalty = 1
        area_per_chiplet = 800 / num_chiplet
        ## area penalty based on constraint of 400mm2
        if area_per_chiplet >= 400:
            area_penalty = area_per_chiplet - 400

        area_for_mac = area_per_chiplet * 0.4
        no_of_macs = area_for_mac / 0.003
        # thruput_per_chiplet = no_of_macs * 2 * 10**12
        thruput_per_chiplet = no_of_macs * 2 * 10 ** -3  # in TOPS (assuming clk freq of 1GHz)
        # print(f'throughput:{thruput_per_chiplet}')
        ai2hbm_bw_req = 4 * 4 * thruput_per_chiplet * 32
        ai2ai_bw_req_2p5D = 4 * 1 * thruput_per_chiplet * 32
        if num_chiplet < 4:
            ai2hbm_bw_req = 4 * num_chiplet * thruput_per_chiplet * 32

        # reward or cost model value calculation
        throughput_max = num_chiplet * thruput_per_chiplet
        lat_penalty = ai2ai_lat + ai2hbm_lat
        if ai2ai_bw_act_2p5D < ai2ai_bw_req_2p5D:
            ai_bw_penalty_2p5D = ai2ai_bw_req_2p5D / ai2ai_bw_act_2p5D
        if ai2hbm_bw_act < ai2hbm_bw_req:
            hbm_bw_penalty = ai2hbm_bw_req / ai2hbm_bw_act
        bw_penalty = ai_bw_penalty_2p5D + hbm_bw_penalty
        cost_tot = package_cost(action)
        energy_tot, _, _, _ = energy_cost(action)
        throughput_achieved = throughput_max - (lat_penalty + bw_penalty + 0.1 * cost_tot + energy_tot + area_penalty)

    # arch type 5p5D_LoL
    if arch_type == 2:
        if ai2ai_dr_2p5D > 20: ai2ai_dr_2p5D = 20
        if ai2hbm_dr > 20: ai2hbm_dr = 20
        if ai2ai_dr_3D < 10: ai2ai_dr_3D = 10

        x, y = factor_pairs(chiplet_pairs)
        ai2ai_lat, ai2hbm_lat, _ = LATENCY(x, y, hbm_loc)
        scale_factor_2p5D = 50
        scale_factor_3D = 100
        ai2ai_bw_act_2p5D = ai2ai_dr_2p5D * 10 ** -3 * ai2ai_pd_2p5D * scale_factor_2p5D
        ai2ai_bw_act_3D = ai2ai_dr_3D * 10 ** -3 * ai2ai_pd_3D * scale_factor_3D
        ai2hbm_bw_act = ai2hbm_dr * 10 ** -3 * ai2hbm_pd * scale_factor_2p5D  ## 10^-3 is because giga is converted to tera
        ai_bw_penalty_2p5D = 1
        ai_bw_penalty_3D = 1
        hbm_bw_penalty = 1
        area_per_chiplet = 800 / chiplet_pairs

        ## area penalty based on constraint of 400mm2
        if area_per_chiplet >= 400:
            area_penalty = area_per_chiplet - 400

        area_for_mac = area_per_chiplet * 0.3
        no_of_macs = area_for_mac / 0.003
        # thruput_per_chiplet = no_of_macs * 2 * 10**12
        thruput_per_chiplet = no_of_macs * 2 * 10 ** -3  # in TOPS (assuming clk freq of 1GHz)
        thruput_per_chiplet_pair = thruput_per_chiplet * 2
        # print(f'throughput:{thruput_per_chiplet}')
        ai2hbm_bw_req = 4 * 4 * thruput_per_chiplet_pair * 32
        ai2ai_bw_req_2p5D = 4 * 1 * thruput_per_chiplet * 32
        ai2ai_bw_req_3D = 4 * 1 * thruput_per_chiplet * 32
        if chiplet_pairs < 4:
            ai2hbm_bw_req = 4 * chiplet_pairs * thruput_per_chiplet_pair * 32

        # todo: start from here, mainly reward calculation
        throughput_max = chiplet_pairs * thruput_per_chiplet_pair
        lat_penalty = ai2ai_lat + ai2hbm_lat
        if ai2ai_bw_act_2p5D < ai2ai_bw_req_2p5D:
            ai_bw_penalty_2p5D = ai2ai_bw_req_2p5D / ai2ai_bw_act_2p5D
        if ai2ai_bw_act_3D < ai2ai_bw_req_3D:
            ai_bw_penalty_3D = ai2ai_bw_req_3D / ai2ai_bw_act_3D
        if ai2hbm_bw_act < ai2hbm_bw_req:
            hbm_bw_penalty = ai2hbm_bw_req / ai2hbm_bw_act
        bw_penalty = ai_bw_penalty_2p5D + ai_bw_penalty_3D + hbm_bw_penalty
        cost_tot = package_cost(action)
        energy_tot, _, _, _ = energy_cost(action)
        throughput_achieved = throughput_max - (lat_penalty + bw_penalty + 0.1 * cost_tot + energy_tot + area_penalty)

    return throughput_achieved, area_per_chiplet, ai2ai_lat, ai2hbm_lat, cost_tot, energy_tot

