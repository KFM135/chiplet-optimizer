# chiplet-optimizer
This repository contains the code for this paper: Chiplet-Gym: An RL-based Optimization Framework for Chiplet-based AI Accelerator

# Usage
## Installation of required packages:
`pip install -r requirements.txt`
## Running optimizer with pretrained RL models:
`python optimizer_final.py`
Outputs will be saved in output_64_chiplets/final_output_<time_stamp>.txt
## Training RL agents: 
`python training_multiple_RL_agent.py`
The models will be saved in RL_64_chiplet/ directory
