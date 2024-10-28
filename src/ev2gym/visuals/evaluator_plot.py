'''
This file is used to plot the comparatigve results of the different algorithms.
'''

import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
from ev2gym.models.ev2gym_env import EV2Gym


marker_list = ['.', 'x', 'o', 'v', 's', 'p',
               'P', '*', 'h', 'H', '+', 'X', 'D', 'd', '|', '_']

# color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e', '#dd4c65', '#93003a']

color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e',
              '#dd4c65', '#93003a', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

algorithm_names = [
    'Charge As Fast As Possible',
    # 'Charge As Late As Possible',
    # 'Round Robin',
    'OCCF V2G',
    'OCCF G2V',
    'eMPC V2G',
    'eMPC G2V',
]


def plot_total_power(results_path, save_path=None, algorithm_names=None):

    # Load the env pickle files
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.figure(figsize=(14, 18))
    plt.rc('font', family='serif')
    light_blue = np.array([0.529, 0.808, 0.922, 1])
    gold = np.array([1, 0.843, 0, 1])

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=10)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.number_of_transformers)))
        dim_y = int(np.ceil(env.number_of_transformers/dim_x))
        for tr in env.transformers:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            if env.config['inflexible_loads']['include']:
                df['inflexible'] = env.tr_inflexible_loads[tr.id, :]
            if env.config['solar_power']['include']:
                df['solar'] = env.tr_solar_power[tr.id, :]

            for cs in tr.cs_ids:
                df[cs] = env.cs_power[cs, :]

            if index == 0:
                # plot the inflexible loads as a fill between
                if env.config['inflexible_loads']['include']:
                    plt.fill_between(df.index,
                                     np.array([0]*len(df.index)),
                                     df['inflexible'],
                                     step='post',
                                     alpha=0.7,
                                     color=light_blue,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Inflexible Loads')

                # plot the solar power as a fill between the inflexible loads and the solar power
                if env.config['solar_power']['include']:
                    plt.fill_between(df.index,
                                     df['inflexible'],
                                     df['solar'] + df['inflexible'],
                                     step='post',
                                     alpha=0.7,
                                     color=gold,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Solar Power')

                if env.config['demand_response']['include']:
                    plt.fill_between(df.index,
                                     np.array([tr.max_power.max()]
                                              * len(df.index)),
                                     tr.max_power,
                                     step='post',
                                     alpha=0.7,
                                     color='r',
                                     hatch='xx',
                                     linestyle='--',
                                     linewidth=2,
                                     label='Demand Response Event')

                plt.step(df.index,
                         #  tr.max_power
                         [tr.max_power.max()] * len(df.index),
                         where='post',
                         color='r',
                         linestyle='--',
                         linewidth=2,
                         label='Transformer Max Power')
                plt.plot([env.sim_starting_date, env.sim_date],
                         [0, 0], 'black')

            df['total'] = df.sum(axis=1)

            # plot total and use different color and linestyle for each algorithm
            plt.step(df.index, df['total'],
                     color=color_list[index],
                     where='post',
                     linestyle='-',
                     linewidth=1,
                     marker=marker_list[index],
                     label=algorithm_names[index])

            counter += 1

    plt.title(f'Transformer {tr.id+1}', fontsize=28)
    plt.xlabel(f'Time', fontsize=28)
    plt.ylabel(f'Power (kW)', fontsize=28)
    plt.xlim([env.sim_starting_date, env.sim_date])
    plt.xticks(ticks=date_range_print,
               labels=[
                   f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
               rotation=45,
               fontsize=28)
    plt.yticks(fontsize=28)
    # put legend under the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=3, fontsize=24)

    plt.grid(True, which='minor', axis='both')
    plt.tight_layout()

    fig_name = f'{save_path}/Transformer_Aggregated_Power.png'
    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')

def plot_total_power_V2G(results_path, save_path=None, algorithm_names=None):

    # Load the env pickle files
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 18))
    plt.grid(True, which='major', axis='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)           
    
    plt.rc('font', family='serif')
   
    light_blue = np.array([0.529, 0.808, 0.922, 1])
    gold = np.array([1, 0.843, 0, 1])
    
    color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
    color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.number_of_transformers)))
        dim_y = int(np.ceil(env.number_of_transformers/dim_x))
        for tr in env.transformers:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            if env.config['inflexible_loads']['include']:
                df['inflexible'] = env.tr_inflexible_loads[tr.id, :]
            if env.config['solar_power']['include']:
                df['solar'] = env.tr_solar_power[tr.id, :]

            for cs in tr.cs_ids:
                df[cs] = env.cs_power[cs, :]

            if index == 0:
                # plot the inflexible loads as a fill between
                if env.config['inflexible_loads']['include']:
                    plt.fill_between(df.index,
                                     np.array([0]*len(df.index)),
                                     df['inflexible'],
                                     step='post',
                                     alpha=0.3,
                                     color=light_blue,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Inflexible Loads')

                # plot the solar power as a fill between the inflexible loads and the solar power
                if env.config['solar_power']['include']:
                    plt.fill_between(df.index,
                                     df['inflexible'],
                                     df['solar'] + df['inflexible'],
                                     step='post',
                                     alpha=0.8,
                                     color=gold,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Solar Power')

                if env.config['demand_response']['include']:
                    plt.fill_between(df.index,
                                     np.array([tr.max_power.max()]
                                              * len(df.index)),
                                     tr.max_power,
                                     step='post',
                                     alpha=0.7,
                                     color='r',
                                     hatch='xx',
                                     linestyle='--',
                                     linewidth=2,
                                     label='Demand Response Event')

                plt.step(df.index,
                         #  tr.max_power
                         [-tr.max_power.max()] * len(df.index),
                         where='post',
                         color='r',
                         linestyle='--',
                         linewidth=2,
                         alpha=0.7,
                        #  label='Transf. Limit'
                         )
                
                plt.step(df.index,
                         #  tr.max_power
                         [tr.max_power.max()] * len(df.index),
                         where='post',
                         color='r',
                         linestyle='--',
                         linewidth=2,
                         alpha=0.7,
                         label='Transf. Limit')
                plt.plot([env.sim_starting_date, env.sim_date],
                         [0, 0], 'black')

            df['total'] = df.sum(axis=1)

            # plot total and use different color and linestyle for each algorithm
            plt.step(df.index, df['total'],
                     color=color_list[index],
                     where='post',
                     linestyle='-',
                     linewidth=1,
                     marker=marker_list[index],
                     label=algorithm_names[index])

            counter += 1
    
    plt.ylabel(f'Power (kW)', fontsize=28)
    plt.xlim([env.sim_starting_date, env.sim_date])
    plt.xticks(ticks=date_range_print,
               labels=[
                   f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
            #    rotation=45,
               fontsize=28)
    plt.yticks(fontsize=28)
    # put legend under the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=3, fontsize=24)    

    fig_name = f'{save_path}/Transformer_Aggregated_Power_Prices_sub.png'
    

    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')

    plt.show()


def plot_comparable_EV_SoC(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    and save the data into a CSV file
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(17, 20))
    plt.rc('font', family='serif')

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=10)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.cs)))
        dim_y = int(np.ceil(env.cs/dim_x))
        for cs in env.charging_stations:            
            plt.subplot(dim_x, dim_y, counter)
            plt.subplots_adjust(hspace=1.5, wspace=0.5)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_energy_level[port, cs.id, :]

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            # Save the SoC data to CSV file
            if save_path:
                csv_filename = f'{save_path}/Charging_Station_{cs.id + 1}_SoC.csv'
                df.to_csv(csv_filename)
                print(f"Saved SoC data for Charging Station {cs.id + 1} to {csv_filename}")

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index,
                             y,
                             where='post',
                             color=color_list[index],
                             marker=marker_list[index],
                             label=algorithm_names[index])

            plt.title(f'Charging Station {cs.id + 1}', fontsize=24)
            plt.ylabel('SoC', fontsize=24)
            plt.ylim([0.1, 1])
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45,
                       fontsize=22)
            counter += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', bbox_to_anchor=(1.1, -0.15),
               fancybox=True, shadow=True, ncol=5, fontsize=24)

    plt.grid(True, which='minor', axis='both')
    plt.tight_layout()

    if save_path:
        fig_name = f'{save_path}/EV_Energy_Level.png'
        plt.savefig(fig_name, format='png',
                    dpi=60, bbox_inches='tight')
        print(f"Saved plot to {fig_name}")

def plot_comparable_EV_SoC_single(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.rc('font', family='serif')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both')

    # CSV 저장을 위한 데이터프레임 초기화
    data_to_save = []

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(minutes=env.timescale),
                                   freq=f'{env.timescale}min')

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

        charge_prices = env.charge_prices[0, :]
        discharge_prices = env.discharge_prices[0, :]

        print(len(charge_prices))
        print(len(discharge_prices))
        
        counter = 1
        for cs in env.charging_stations:   
            if counter != 1:
                counter += 1
                continue
            
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_energy_level[port, cs.id, :]

            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    y = df[port].values.T[t_arr:t_dep]
                    y = np.concatenate([np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    # SoC 및 시간 정보를 data_to_save에 추가
                    for time_index in range(len(df)):
                        timestamp = df.index[time_index]
                        soc_value = y[time_index] * 100  # SoC를 퍼센트로 변환
                        data_to_save.append({
                            "Datetime": timestamp,
                            "Datetime (UTC)": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "Charge_Prices": charge_prices[time_index-1],  # 해당 시점의 충전 가격
                            "Discharge_Prices": discharge_prices[time_index-1],  # 해당 시점의 방전 가격
                            "Battery_Level": soc_value,
                        })

                    plt.step(df.index,
                             y,
                             where='post',
                             color=color_list[index],
                             alpha=0.8,
                             label=algorithm_names[index])

            if counter == 1:
                plt.ylabel('SoC', fontsize=28)
                plt.yticks(np.arange(0, 1.1, 0.2),
                           fontsize=28)
                    
            else:
                plt.yticks(fontsize=28)
                plt.yticks(np.arange(0, 1.1, 0.1),
                            labels=[' ' for d in np.arange(0, 1.1, 0.1)])            
            
            plt.ylim([0.1, 1.09])
            plt.xlim([env.sim_starting_date, env.sim_date])
            counter += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', bbox_to_anchor=(0, -0.15),
               fancybox=True, shadow=True, ncol=3, fontsize=24)

    plt.tight_layout()

    # CSV simulation file
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    csv_file_path = f'{save_path}/Simulation_{current_date}.csv'
    df_to_save = pd.DataFrame(data_to_save)
    df_to_save.to_csv(csv_file_path, index=False)

    # 그래프 저장
    fig_name = f'{save_path}/EV_Energy_Level_single.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')

    #plt.show()

def plot_comparable_CS_Power(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('font', family='serif')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both')
    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))
        
        counter = 1
        for cs in env.charging_stations:   
            if counter != 1:
                counter += 1
                continue
            
            # plt.subplot(1, 2, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_current[port, cs.id, :]
            
            #multiply df[port] by the voltage to get the power
            df = df * cs.voltage * math.sqrt(cs.phases) / 1000
            
            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index,
                             y,
                             where='post',
                             color=color_list[index],
                             marker=marker_list[index],
                             alpha=0.8,
                             label=algorithm_names[index])

            # plt.title(f'Charging Station {cs.id + 1}', fontsize=24)
            
            if counter == 1:
                plt.ylabel('Power (kW)', fontsize=28)
                plt.yticks([-22,-11,0,11,22],
                           fontsize=28)
                    
            else:
                plt.yticks(fontsize=28)
                # plt.yticks(np.arange(0, 1.1, 0.1),
                #             labels=[' ' for d in np.arange(0, 1.1, 0.1)])            
            
            # plt.ylim([0.1, 1.09])
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                    #    rotation=45,
                       fontsize=28)
            counter += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', bbox_to_anchor=(0, -0.15),
               fancybox=True, shadow=True, ncol=3, fontsize=24)

    plt.tight_layout()

    fig_name = f'{save_path}/CS_Power_single.png'
    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')

    plt.show()   

def plot_actual_power_vs_setpoint(results_path, save_path=None, algorithm_names=None):
    
    '''
    This function is used to plot the actual power vs the setpoint power.
    It plots the behavior of each algorithm in subplots vertically.
    '''
    
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(14, 20))
    plt.rc('font', family='serif')    

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        #plot the actual power vs the setpoint power for each algorithm in subplots                
        plt.subplot(len(replay), 1, index+1)
        plt.grid(True, which='major', axis='both')
        
        actual_power = env.current_power_usage        
        setpoints = env.power_setpoints                

        plt.step(date_range, actual_power.T, alpha=0.9, color='#00429d')
        plt.step(date_range, setpoints.T, alpha=1, color='#93003a')
        
        plt.axhline(0, color='black', lw=2)
        plt.title(f'{algorithm_names[index]}', fontsize=22)
        
        plt.yticks(fontsize=22)
        
        if index == len(replay) - 1:
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                    #    rotation=45,
                       fontsize=22)
            # plt.xlabel('Time', fontsize=28)
        else:
            plt.xticks(ticks=date_range_print,
                       labels=[' ' for d in date_range_print])
        
        if index == len(replay) // 2:
            plt.ylabel('Power (kW)', fontsize=22)               
            
        plt.xlim([env.sim_starting_date, env.sim_date])
        plt.ylim([0, 1.1*env.current_power_usage.max()])
        
    # Put the legend under the plot in a separate axis           
    plt.legend(['Actual Power', 'Setpoint'], loc='upper center',
               bbox_to_anchor=(0.5, -0.5),
               fancybox=True, shadow=True, ncol=2, fontsize=22)
        
    plt.tight_layout()
    fig_name = f'{save_path}/Actual_vs_Setpoint_Power_sub.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')    
    
def plot_prices(results_path, save_path=None, algorithm_names=None):
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure()
    plt.rc('font', family='serif')
    
    keys = list(replay.keys())
    env = replay[keys[0]]
    
    date_range = pd.date_range(start=env.sim_starting_date,
                                 end=env.sim_starting_date +
                                 (env.simulation_length - 1) *
                                 datetime.timedelta(
                                      minutes=env.timescale),
                                 freq=f'{env.timescale}min')
    date_range_print = pd.date_range(start=env.sim_starting_date,
                                     end=env.sim_date,
                                     periods=7)
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 10))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.grid(True, which='major', axis='both')    
    plt.rc('font', family='serif')
    
    charge_prices = env.charge_prices[0, :]
    discharge_prices = env.discharge_prices[0, :]

    print(charge_prices)
    print(discharge_prices)
    
    plt.step(date_range, -charge_prices, alpha=0.9, color='#00429d',label='Charge Prices')
    plt.step(date_range, discharge_prices, alpha=1, color='#93003a', label='Discharge Prices')
    
    plt.xlim([env.sim_starting_date, env.sim_date])
    # plt.ylim()
    # plt.axhline(0, color='black', lw=2)    
    y_ticks = np.arange(0.150, 0.351, 0.05)
    plt.yticks(y_ticks,fontsize=28)

    plt.ylim([0.12, 0.35])
    plt.xticks(ticks=date_range_print,
               labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
            #    rotation=45,
               fontsize=28)
    plt.ylabel('Price (€/kWh)', fontsize=28)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), fontsize=28)
    #plt.legend(fontsize=28)
    
    #show grid lines
    
    
    # plt.tight_layout()
    fig_name = f'{save_path}/Prices.png'
    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparable_EV_Prices_SoC_and_Average(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the charging and discharging prices of the EVs along with SoC and average price
    and save the data into a CSV file
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=10)

        dim_x = int(np.ceil(np.sqrt(env.cs)))
        dim_y = int(np.ceil(env.cs / dim_x))

        for cs in env.charging_stations:
            plt.figure(figsize=(17, 10))
            plt.subplots_adjust(hspace=0.5, wspace=0.5)

            charge_prices = env.charge_prices[index]  # Assuming prices are stored in replay
            discharge_prices = env.discharge_prices[index]
            average_price = env.average_price     # Assuming average_price is a float

            # Create an average price array with the same length as charge_prices
            average_price_array = np.full(len(charge_prices), -average_price)  # All values set to average_price

            # Create a primary Y-axis for prices
            ax1 = plt.gca()
            
            # Plot charging and discharging prices
            ax1.plot(date_range, -charge_prices, label='Charging Price', color='blue')
            ax1.plot(date_range, discharge_prices, label='Discharging Price', color='red')
            ax1.plot(date_range, average_price_array, label='Average Price', color='orange', linestyle='-.')
            
            ax1.set_ylabel('Price (€/kWh)', fontsize=24)
            ax1.set_ylim([0, max(max(charge_prices), max(discharge_prices)) * 1.1])
            
            ax1.set_xlim([env.sim_starting_date, env.sim_date])
            ax1.tick_params(axis='y', labelcolor='black')

            # Create a second y-axis for SoC
            ax2 = ax1.twinx()
            
            for port in range(cs.n_ports):
                soc_data = env.port_energy_level[port, cs.id, :]  # Assuming this contains SoC data
                ax2.plot(date_range, soc_data, label=f'SoC Port {port}', linestyle='--')

            ax2.set_ylabel('SoC', fontsize=24)
            ax2.set_ylim([0, 1])  # SoC range from 0 to 1
            ax2.tick_params(axis='y', labelcolor='green')

            plt.title(f'Charging, Discharging Prices, SoC and Average Price for Charging Station {cs.id + 1}', fontsize=24)

            # Adjust x-ticks for better readability
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45,
                       fontsize=22)

            # Combine legends from all axes
            handles, labels = ax1.get_legend_handles_labels()  # Get handles from the first axis
            handles2, labels2 = ax2.get_legend_handles_labels()  # Get handles from the second axis
            
            # Combine all handles and labels
            handles.extend(handles2)
            labels.extend(labels2)

            # Add algorithm name to the legend
            if algorithm_names is not None and index < len(algorithm_names):
                handles.append(plt.Line2D([0], [0], color='black', label=algorithm_names[index]))
                labels.append(algorithm_names[index])

            plt.legend(handles, labels, loc='upper left', fontsize=14)
            plt.grid(True)
            plt.tight_layout()

            if save_path:
                fig_name = f'{save_path}/EV_Prices_SoC_Info_CS_{cs.id + 1}.png'
                plt.savefig(fig_name, format='png', dpi=60, bbox_inches='tight')
                print(f"Saved plot to {fig_name}")

def plot_comparable_EV_SoC_and_Prices(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs along with charging, discharging, and average prices in the same plot.
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(14, 10))
    plt.rc('font', family='serif')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both')

    # CSV 저장을 위한 데이터프레임 초기화
    data_to_save = []

    # 알고리즘별로 이미 표시된 SoC 정보를 저장할 집합
    displayed_algorithms = set()
    displayed_labels = set()  # 레이블 중복 체크

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(minutes=env.timescale),
                                   freq=f'{env.timescale}min')

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

        charge_prices = env.charge_prices[0, :]
        discharge_prices = env.discharge_prices[0, :]
        average_price = env.average_price  # 평균 가격 정보

        counter = 1
        for cs in env.charging_stations:   
            if counter != 1:
                counter += 1
                continue
            
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_energy_level[port, cs.id, :]

            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    y = df[port].values.T[t_arr:t_dep]
                    y = np.concatenate([np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    # SoC 및 시간 정보를 data_to_save에 추가
                    for time_index in range(len(df)):
                        timestamp = df.index[time_index]
                        soc_value = y[time_index] * 100  # SoC를 퍼센트로 변환
                        data_to_save.append({
                            "Datetime": timestamp,
                            "Charge_Prices": charge_prices[time_index-1],  # 해당 시점의 충전 가격
                            "Discharge_Prices": discharge_prices[time_index-1],  # 해당 시점의 방전 가격
                            "Average_Prices": average_price,  # 평균 가격
                            "Battery_Level": soc_value,
                        })

                    # SoC 그래프에 표시 (중복 체크)
                    if algorithm_names[index] not in displayed_algorithms:
                        ax1.step(df.index,
                                  y * 100,  # SoC를 0-100 사이로 변환
                                  where='post',
                                  color=color_list[index],
                                  alpha=0.8,
                                  label=f'SoC {algorithm_names[index]}')  # 중복 방지

                        displayed_algorithms.add(algorithm_names[index])  # 표시된 알고리즘 추가

            if counter == 1:
                ax1.set_ylabel('SoC (%)', fontsize=28)
                ax1.set_ylim([0, 100])
                ax1.set_yticks(np.arange(0, 101, 20))  # y-ticks 설정
                ax1.tick_params(axis='y', labelsize=24)  # y-ticks 폰트 크기 설정
                    
            else:
                ax1.tick_params(axis='y', labelsize=28)

            ax1.set_xlim([env.sim_starting_date, env.sim_date])
            counter += 1

    # 가격 정보를 추가하는 새로운 y축
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price (€/kWh)', fontsize=28)

    # Charging and Discharging Prices (중복 체크)
    if 'Charging Price' not in displayed_labels:
        ax2.plot(date_range, -charge_prices, color='#00429d', label='Charging Price', linestyle='-.')
        displayed_labels.add('Charging Price')

    if 'Discharging Price' not in displayed_labels:
        ax2.plot(date_range, discharge_prices, color='#93003a', label='Discharging Price', linestyle='-.')
        displayed_labels.add('Discharging Price')

    # Average Price (중복 체크)
    if 'Average Price' not in displayed_labels:
        average_prices = np.full(len(charge_prices), average_price)  # 평균 가격을 배열로 생성
        ax2.plot(date_range, -average_prices, label='Average Price', color='orange', linestyle='-.')
        displayed_labels.add('Average Price')

    ax2.set_ylim([0, max(max(charge_prices), max(discharge_prices), average_price) * 1.1])
    ax2.tick_params(axis='y', labelcolor='black')

    # 레전드 통합
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    handles1.extend(handles2)
    labels1.extend(labels2)

    # 레전드를 그래프 하단에 겹치지 않도록 배치
    ax1.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=3, fontsize=14)

    plt.tight_layout()

    # CSV 파일로 저장
    df_to_save = pd.DataFrame(data_to_save)
    csv_file_path = f'{save_path}/EV_Energy_Level_and_Prices.csv'
    df_to_save.to_csv(csv_file_path, index=False)

    # 그래프 저장
    fig_name = f'{save_path}/EV_Energy_Level_and_Prices.png'
    plt.savefig(fig_name, format='png', dpi=60, bbox_inches='tight')

    plt.show()



if __name__ == "__main__":

    pkl_file_path = "../results/eval_20cs_1tr_V2GProfitMax_9_algos_50_exp_2024_10_11_184281/plot_results_dict.pkl"
    result_path = "../results/eval_20cs_1tr_V2GProfitMax_9_algos_50_exp_2024_10_11_184281"

    plot_comparable_EV_SoC_single(results_path=pkl_file_path,
                     save_path=result_path,
                     algorithm_names=['PPO', 'A2C', 'DDPG', 'SAC', 'TD3', 'TQC', 
                                         'TRPO', 'ARS', 'RecurrentPPO'])


    pass