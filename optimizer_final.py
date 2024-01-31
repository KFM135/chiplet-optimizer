import subprocess
import os
from cost_function import *
import ast
import gym
import numpy as np
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN, HerReplayBuffer

start_time = datetime.datetime.now()

## Running simulated annealing
'''
There is no pretrained version of simulated annealing. So each time we ran the code, the SA will be initialized to a 
random point and will find the optimum parameter from there on.
'''
#
run_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = 'output_64_chiplets'
isExist = os.path.exists(out_dir)
if not isExist:
    os.makedirs(out_dir)

SA_script_path = os.path.join(run_dir, 'SA_64_chiplet.py')

best_cost_model_val_SA = -np.inf
best_parameter_SA = []
for i in range(0, 10):
    print(f'Running script {i} of 10')
    out_file_name = 'SA_3D_64_' + str(i) + '.txt'
    out_file_path = os.path.join(out_dir, out_file_name)
    command = [sys.executable, SA_script_path, f'1>{out_file_path}', str(i)]
    # command = [sys.executable, SA_script_path]
    result = subprocess.run(command, shell = True, capture_output = True, text = True)
    with open(out_file_path, 'r') as r:
        all_vals = r.readlines()
    best_param = [all_vals[i] for i in range(0, len(all_vals)) if 'Best Parameter' in all_vals[i]]
    best_param = best_param[0].split(':')[1]
    action = ast.literal_eval(best_param)       # converting the string literal to actual list
    action = action_refined(action)
    # print(f'best param:{type(best_param)}, action:{type(action)}')
    best_throughput, _, _, _, _, _ = throughput(action)

    if best_throughput > best_cost_model_val_SA:
        best_parameter_SA, best_cost_model_val_SA = action, best_throughput



## RL section
'''
Here the pretrained RL models are used to predict the actions. And the best ever action is taken as the optimum parameter.
However, the RL model can also be trained with this script. todo: add this feature
'''

# defining the environment
best_cost_model_val_RL = -np.inf
best_parameter_RL = []


gym.envs.register(
    id = 'ChipletEnv',
    entry_point = 'gym.envs.classic_control:CustomEnv',
    max_episode_steps = 10,
)
env = gym.make('ChipletEnv')

RL_model_dir = 'RL_64_chiplet'
file_list = os.listdir(RL_model_dir)
trained_RL_models = [file for file in file_list if file.endswith('.zip')]

for i in range(0, len(trained_RL_models)):
    print(f'Using model:{trained_RL_models[i]}')
    model = trained_RL_models[i]
    model = os.path.join(RL_model_dir, model)
    model = PPO.load(model)
    for j in range(0, 20):
        rand_obs = env.observation_space.sample()
        action_after_training, _ = model.predict(rand_obs, deterministic=True)
        action = action_refined(action_after_training)
        best_throughput_RL, _, _, _, _, _ = throughput(action)
        print(f'best param:{action}, best throughput:{best_throughput_RL}')
        if best_throughput_RL > best_cost_model_val_RL:
            best_parameter_RL, best_cost_model_val_RL = action, best_throughput_RL

# write the final output
current_time = str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)
final_output = 'final_output_' + current_time + '.txt'
final_output_path = os.path.join(out_dir, final_output)

if best_cost_model_val_RL > best_cost_model_val_SA:
    best_parameter, best_cost_model_val = best_parameter_RL, best_cost_model_val_RL
else:
    best_parameter, best_cost_model_val = best_parameter_SA, best_cost_model_val_SA

end_time = datetime.datetime.now()


with open(final_output_path, 'w') as fopen:
    fopen = fopen.writelines(f'Best Parameter RL:{best_parameter_RL}, Best Throughput RL:{best_cost_model_val_RL}\n'
                             f'Best Parameter SA:{best_parameter_SA}, Best Throughput SA:{best_cost_model_val_SA}\n'
                             f'Best Parameter:{best_parameter}, Best Throughput:{best_cost_model_val}\n'
                             f'Run time:{end_time - start_time}')
