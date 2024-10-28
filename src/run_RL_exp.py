"""
This file is used to run various experiments in different tmux panes each.
"""
import subprocess
import os
import time

def run_experiment(algorithm, device, config_file):
    command = f'python ./train_stable_baselines.py --algorithm {algorithm} --device {device} --config_file {config_file}'
    subprocess.run(command, shell=True, check=True)

counter = 0
device = 'cuda:0'

algorithms = ['ddpg', 'td3', 'sac', 'a2c', 'ppo', 'tqc', 'trpo', 'ars', 'rppo']
configs = ['V2GProfitMax']
#['V2GProfitMax', 'PublicPST', 'V2GProfitPlusLoads']

for config in configs:
    for algorithm in algorithms:
        config_file = os.path.join(os.getcwd(), 'example_config_files', f'{config}.yaml')
        print(f'Starting training for algorithm: {algorithm} with config: {config_file}')
        run_experiment(algorithm, device, config_file)
        print(f'Completed training for algorithm: {algorithm} with config: {config_file}')
        time.sleep(10)

