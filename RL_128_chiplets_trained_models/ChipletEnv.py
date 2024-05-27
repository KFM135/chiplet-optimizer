import os
from gym import Env
from gym import spaces
from cost_function import *
import datetime

class CustomEnv(Env):
    def __init__(self):
        self.max_package_area = 800      # unit mm^2
        self.max_chiplet_area = 400      # unit mm^2
        self.max_chiplet = 256
        self.max_AI_lat = 50
        self.max_HBM_lat = 50
        self.max_energy = 100
        self.target_throughput = 2048   # TOPS
        self.operations_per_task = 4    # GFLOPS
        self.mac_unit_area = 0.003      # unit um^2
        self.max_cost = 150
        self.ai2ai_lat = []
        self.ai2hbm_lat = []
        self.action_space = spaces.MultiDiscrete([3, 65, 62, 3, 50, 100, 10, 3, 50, 100, 10, 3, 50, 100, 10])
        self.obs_space_low = np.array([self.max_package_area, 1, 0, 0, 0, self.target_throughput, 0, 0, 0, self.operations_per_task])
        self.obs_space_high = np.array([self.max_package_area, self.max_chiplet_area, self.max_chiplet_area, self.max_AI_lat, self.max_HBM_lat,
                                        self.target_throughput, self.target_throughput, self.max_energy, self.max_cost, self.operations_per_task])
        self.observation_space = spaces.Box(low = self.obs_space_low, high = self.obs_space_high)
        self.current_obs = None
        self.time_step = None

    def reset(self):
        self.time_step = 0
        allowed_chiplet_area = random.randint(1, self.max_chiplet_area)
        self.current_obs = np.array(
            [self.max_package_area, allowed_chiplet_area, 0, 0, 0, self.target_throughput, 0,
             0, 0, self.operations_per_task])
        return self.current_obs

    def step(self, action):
        """Takes action and returns next observation, reward done and optionally additional info"""
        done = False
        # print(f'action before:{action}')
        action = action_refined(action)
        # print(f'action after:{action}')
        throughput_achieved, area_per_chiplet, ai2ai_lat, ai2hbm_lat, cost_tot, energy_tot = throughput(action)
        self.current_obs[2] = area_per_chiplet
        self.current_obs[3] = ai2ai_lat
        self.current_obs[4] = ai2hbm_lat
        self.current_obs[6] = throughput_achieved
        self.current_obs[8] = cost_tot
        self.current_obs[7] = energy_tot

        next_obs = self.current_obs
        reward = throughput_achieved
        bonus = 0

        if action[0] == 2:
            bonus = 50
            reward = reward + bonus

        self.time_step += 1
        if self.time_step > 1:
            done = True
        self.current_obs = next_obs


        return self.current_obs, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass
