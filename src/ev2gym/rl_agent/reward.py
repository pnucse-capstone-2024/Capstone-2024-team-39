'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''

import math
import numpy as np

def calculate_power_tracker_violation(env):
    violation = 0
    for tr in env.transformers:
        # current power limits for transformer
        power_limits = tr.get_power_limits(step=env.current_step - 1, horizon=1)[0]
        # actual power using
        actual_power = tr.current_power
        
        # violation for gap
        violation += abs(actual_power - power_limits)
    return violation

def calculate_tracking_error(env):
    error = 0
    for tr in env.transformers:
        power_limits = tr.get_power_limits(step=env.current_step - 1, horizon=1)[0]
        actual_power = tr.current_power
        
        # tracking violation error calculation
        error += (actual_power - power_limits) ** 2
    return error

def calculate_average_user_satisfaction(user_satisfaction_list):
    if len(user_satisfaction_list) > 0:
        average_user_satisfaction = sum(user_satisfaction_list) / len(user_satisfaction_list)
    else:
        average_user_satisfaction = 0  # empty user lists

    return average_user_satisfaction

# peak / non-peak (8~11시, 17~21시 피크 시간으로 가정)
def is_peak_hour(current_step):
    peak_hours = [i for i in range(8, 12)] + [i for i in range(17, 22)]
    hour_of_day = (current_step - 1) % 24  # 하루 24시간 기준
    return hour_of_day in peak_hours

def get_current_signal(env, port, cs_id, step):
    # Check if the step is within valid range for the array
    if 0 <= step - 1 < env.port_current_signal.shape[2]:
        return env.port_current_signal[port, cs_id, step - 1]
    else:
        print(f"Warning: Current step {step - 1} is out of bounds for array with shape {env.port_current_signal.shape}")
        return 0  # Default value if out of bounds

def grid_reward(env):
    reward_grid = 0

    for tr in env.transformers:
        # Check if the current step does not exceed the length of the min_power array
        if (env.current_step - 1) < len(tr.min_power):
            min_power_penalty = 1000 * max(0, tr.min_power[env.current_step - 1] - tr.current_power)
            reward_grid -= min_power_penalty
        else:
            print(f"Warning: env.current_step {env.current_step - 1} exceeds min_power array length {len(tr.min_power)}")

        overload_penalty = 500 * tr.get_how_overloaded()
        reward_grid -= overload_penalty
        
        if tr.is_overloaded():
            reward_grid -= 1000  # Additional penalty if overloaded
    
    tracker_violation = calculate_power_tracker_violation(env)
    tracking_error = calculate_tracking_error(env)

    reward_grid -= 100 * tracker_violation  
    reward_grid -= 50 * tracking_error 

    return reward_grid

def provider_reward(env, user_satisfaction_list):
    reward_provider = 0

    # Calculate rewards related to the charging service provider
    total_cost = float(env.get_total_costs())  # Convert to float in case it's an array
    total_energy_charged = float(env.total_energy_charged)  # Handle similarly
    total_energy_discharged = float(env.total_energy_discharged)
    average_user_satisfaction = float(calculate_average_user_satisfaction(user_satisfaction_list))

    reward_provider += -1000 * total_cost  # Minimize costs
    reward_provider += 10 * total_energy_charged  # Reward for total energy charged
    
    efficiency_ratio = total_energy_charged / (total_energy_discharged + 1e-10)
    efficiency_threshold = 0.8
    if efficiency_ratio < efficiency_threshold:
        reward_provider -= 500 * (efficiency_threshold - efficiency_ratio)  # Efficiency penalty
    
    satisfaction_threshold = 0.7
    if average_user_satisfaction > satisfaction_threshold:
        reward_provider += 1000 * average_user_satisfaction  # Reward for user satisfaction
    else:
        reward_provider -= 300 * (satisfaction_threshold - average_user_satisfaction)  # Penalty for user satisfaction

    return reward_provider

def ev_reward(env):
    reward_ev = 0

    soc_values = [ev.get_soc() for cs in env.charging_stations for ev in cs.evs_connected if ev is not None]
    if soc_values:  # Calculate only if not empty
        avg_battery_health = np.mean(soc_values)
    else:
        avg_battery_health = 0
    
    reward_ev += avg_battery_health * 0.1  # Reward for battery health

    # Check for peak hour
    peak_hour = is_peak_hour(env.current_step)

    for cs in env.charging_stations:
        for port in range(cs.n_ports):
            # Get current_signal
            current_signal = env.port_current_signal[port, cs.id, env.current_step - 1]

            if current_signal > 0:  # Charging
                is_charging = True
                is_discharging = False
            elif current_signal < 0:  # Discharging
                is_charging = False
                is_discharging = True
            else:
                is_charging = False
                is_discharging = False

            # Amount of energy exchanged
            energy_exchanged = abs(current_signal)  # Use the absolute value of the current signal

            # Additional reward for discharging during peak hour
            if peak_hour and is_discharging:
                reward_ev += energy_exchanged * 0.1  # Reward for discharging during peak hour

            # Additional reward for charging during off-peak hour
            if not peak_hour and is_charging:
                reward_ev += energy_exchanged * 0.1  # Reward for charging during off-peak hour

    for cs in env.charging_stations:
        for port in range(cs.n_ports):
            ev = cs.evs_connected[port]
            if ev is not None:
                energy_exchanged = ev.total_energy_exchanged
                reward_ev -= energy_exchanged * 0.0005  # Penalty for energy exchanged
                if ev.get_soc() > 0.8:
                    reward_ev += 0.01  # Reward for SOC

    # Get charging and discharging prices based on current step and charging station ID
    for cs in env.charging_stations:
        if env.current_step - 1 < env.charge_prices.shape[1] and env.current_step - 1 < env.discharge_prices.shape[1]:
            current_charge_price = env.charge_prices[cs.id, env.current_step - 1]
            current_discharge_price = env.discharge_prices[cs.id, env.current_step - 1]
        else:
            current_charge_price = 0  
            current_discharge_price = 0 

        reward_ev -= (current_charge_price - current_discharge_price) * 0.02  # Penalty for price difference

    avg_power_output = np.mean([cs.current_power_output for cs in env.charging_stations])
    reward_ev += avg_power_output * 0.01  # Reward for average power output

    return reward_ev


def CombinedRewardFunction_2(env, total_costs, user_satisfaction_list, *args):
    w_grid=1.0
    w_provider=1.0
    w_ev=1.0

    reward_grid =grid_reward(env)
    reward_provider = provider_reward(env, user_satisfaction_list)
    reward_ev = ev_reward(env)  

    # calculate total reward
    total_reward = w_grid * reward_grid + w_provider * reward_provider + w_ev * reward_ev

    return total_reward


def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
        env.current_power_usage[env.current_step-1])**2
        
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    It penalizes transofrmers that are overloaded    
    The reward is negative'''
    
    tr_max_limit = env.transformers[0].max_power[env.current_step-1]
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
        env.current_power_usage[env.current_step-1])**2
            
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
        
    for score in user_satisfaction_list:
        reward -= 1000 * (1 - score)
                    
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        reward -= 100 * math.exp(-10*score)
        
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative
    If the EV is not charging, the reward is penalized
    '''
    if env.current_power_usage[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2 - 100
    else:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    '''This reward function does not consider the charge power potential'''
    
    reward = - (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    ''' This reward function minimizes the tracker surplus and gives a reward for charging '''
    
    reward = 0
    if env.power_setpoints[env.current_step-1] < env.current_power_usage[env.current_step-1]:
            reward -= (env.current_power_usage[env.current_step-1]-env.power_setpoints[env.current_step-1])**2

    reward += env.current_power_usage[env.current_step-1] #/75
    
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    ''' This reward function is used for the profit maximization case '''
    
    reward = total_costs
    
    for score in user_satisfaction_list:
        # reward -= 100 * (1 - score)
        reward -= 100 * math.exp(-10*score)
    
    return reward



# Previous reward functions for testing
#############################################################################################################
# def SquaredError_grid_reward(env):
#     reward_grid = 0

#     # 전력망 운영자 관련 보상 계산
#     for tr in env.transformers:
#         # 현재 스텝이 min_power 배열의 길이를 초과하지 않는지 확인
#         if (env.current_step - 1) < len(tr.min_power):
#             min_power_penalty = 1000 * max(0, tr.min_power[env.current_step - 1] - tr.current_power)
#             reward_grid -= min_power_penalty
#         else:
#             print(f"Warning: env.current_step {env.current_step - 1} exceeds min_power array length {len(tr.min_power)}")

#         overload_penalty = 500 * tr.get_how_overloaded()
#         reward_grid -= overload_penalty
        
#         if tr.is_overloaded():
#             reward_grid -= 1000  # 과부하 발생 시 추가 페널티

#     # Squared error 방식으로 계산
#     tracker_violation = calculate_power_tracker_violation(env)
#     tracking_error = calculate_tracking_error(env)

#     reward_grid -= 100 * (tracker_violation ** 2)  # tracker_violation의 제곱
#     reward_grid -= 50 * (tracking_error ** 2)  # tracking_error의 제곱

#     return reward_grid

# def SquaredError_provider_reward(env, user_satisfaction_list):
#     reward_provider = 0

#     # 충전 서비스 사업자 관련 보상 계산
#     total_cost = float(env.get_total_costs())  # 배열일 수 있는 경우 float로 변환
#     total_energy_charged = float(env.total_energy_charged)  # 동일하게 처리
#     total_energy_discharged = float(env.total_energy_discharged)
#     average_user_satisfaction = float(calculate_average_user_satisfaction(user_satisfaction_list))

#     reward_provider += -1000 * total_cost  # 비용 최소화
#     reward_provider += 10 * total_energy_charged  # 총 에너지 충전 보상
    
#     # 효율성 페널티를 squared error로 계산
#     efficiency_ratio = total_energy_charged / (total_energy_discharged + 1e-10)
#     efficiency_threshold = 0.8
#     if efficiency_ratio < efficiency_threshold:
#         reward_provider -= 500 * ((efficiency_threshold - efficiency_ratio) ** 2)  # 효율성 페널티 (제곱)

#     # 사용자 만족도 보상을 squared error로 계산
#     satisfaction_threshold = 0.8
#     if average_user_satisfaction > satisfaction_threshold:
#         reward_provider += 1000 * ((average_user_satisfaction - satisfaction_threshold) ** 2)  # 만족도 보상 (제곱)
#     else:
#         reward_provider -= 500 * ((satisfaction_threshold - average_user_satisfaction) ** 2)  # 만족도 페널티 (제곱)

#     return reward_provider

# def SquaredError_ev_reward(env):
#     reward_ev = 0

#     # Battery health reward (squared error 적용)
#     soc_values = [ev.get_soc() for cs in env.charging_stations for ev in cs.evs_connected if ev is not None]
#     avg_battery_health = np.mean(soc_values) if soc_values else 0
#     target_soc = 0.8  # 목표 SOC 값
#     reward_ev -= (avg_battery_health - target_soc) ** 2  # SOC에 대해 squared error로 계산

#     # Peak hour determination
#     peak_hour = is_peak_hour(env.current_step)

#     # Charging and discharging rewards
#     for cs in env.charging_stations:
#         for port in range(cs.n_ports):
#             current_signal = get_current_signal(env, port, cs.id, env.current_step)  # <-
#             is_charging = current_signal > 0
#             is_discharging = current_signal < 0

#             energy_exchanged = abs(current_signal)  # Absolute value of current signal

#             if peak_hour and is_discharging:
#                 reward_ev += (energy_exchanged ** 2) * 0.1  # Reward for discharging during peak hours (squared energy)
#             if not peak_hour and is_charging:
#                 reward_ev += (energy_exchanged ** 2) * 0.1  # Reward for charging during non-peak hours (squared energy)

#     # Penalties and rewards based on energy exchanged and SOC (squared error 적용)
#     for cs in env.charging_stations:
#         for port in range(cs.n_ports):
#             ev = cs.evs_connected[port]
#             if ev is not None:
#                 energy_exchanged = ev.total_energy_exchanged
#                 reward_ev -= (energy_exchanged ** 2) * 0.0005  # Penalty for energy exchanged (squared energy)
#                 if ev.get_soc() > 0.8:
#                     reward_ev += ((ev.get_soc() - 0.8) ** 2) * 0.01  # Reward for SOC > 0.8 (squared SOC difference)

#     # Price penalties (squared error 적용)
#     for cs in env.charging_stations:
#         if 0 <= env.current_step - 1 < env.charge_prices.shape[1] and 0 <= env.current_step - 1 < env.discharge_prices.shape[1]:
#             current_charge_price = env.charge_prices[cs.id, env.current_step - 1]
#             current_discharge_price = env.discharge_prices[cs.id, env.current_step - 1]
#         else:
#             current_charge_price = 0
#             current_discharge_price = 0

#         reward_ev -= ((current_charge_price - current_discharge_price) ** 2) * 0.02  # Price difference penalty (squared)

#     # Average power output reward (squared error 적용)
#     avg_power_output = np.mean([cs.current_power_output for cs in env.charging_stations])
#     target_avg_power_output = 0  # 목표 평균 출력 값 (필요에 따라 변경 가능)
#     reward_ev += ((avg_power_output - target_avg_power_output) ** 2) * 0.01  # Reward for average power output (squared)

#     return reward_ev

# def SquaredCombinedRewardFunction(env, total_costs, user_satisfaction_list, *args):
#     w_grid=1.0
#     w_provider=1.0
#     w_ev=1.0

#     reward_grid = SquaredError_grid_reward(env)
#     reward_provider = SquaredError_provider_reward(env, user_satisfaction_list)
#     reward_ev = SquaredError_ev_reward(env)  

#     # 종합 보상 계산
#     total_reward = w_grid * reward_grid + w_provider * reward_provider + w_ev * reward_ev

#     return total_reward

# def grid_reward(env):
#     reward_grid = 0

#     # 전력망 운영자 관련 보상 계산
#     for tr in env.transformers:
#         # 현재 스텝이 min_power 배열의 길이를 초과하지 않는지 확인
#         if (env.current_step - 1) < len(tr.min_power):
#             min_power_penalty = 100 * max(0, tr.min_power[env.current_step - 1] - tr.current_power)
#             reward_grid -= min_power_penalty
#         else:
#             print(f"Warning: env.current_step {env.current_step - 1} exceeds min_power array length {len(tr.min_power)}")

#         overload_penalty = 50 * tr.get_how_overloaded()
#         reward_grid -= overload_penalty
        
#         if tr.is_overloaded():
#             reward_grid -= 100  # 과부하 발생 시 추가 페널티
    
#     tracker_violation = calculate_power_tracker_violation(env)
#     tracking_error = calculate_tracking_error(env)

#     reward_grid -= 10 * tracker_violation  
#     reward_grid -= 5 * tracking_error 

#     return reward_grid


# def provider_reward(env, user_satisfaction_list):
#     reward_provider = 0

#     # 충전 서비스 사업자 관련 보상 계산
#     total_cost = float(env.get_total_costs())  # 배열일 수 있는 경우 float로 변환
#     total_energy_charged = float(env.total_energy_charged)  # 동일하게 처리
#     total_energy_discharged = float(env.total_energy_discharged)
#     average_user_satisfaction = float(calculate_average_user_satisfaction(user_satisfaction_list))

#     reward_provider += -100 * total_cost  # 비용 최소화
#     reward_provider += 10 * total_energy_charged  # 총 에너지 충전 보상
    
#     efficiency_ratio = total_energy_charged / (total_energy_discharged + 1e-10)
#     efficiency_threshold = 0.8
#     if efficiency_ratio < efficiency_threshold:
#         reward_provider -= 50 * (efficiency_threshold - efficiency_ratio)  # 효율성 페널티
    
#     satisfaction_threshold = 0.7
#     if average_user_satisfaction > satisfaction_threshold:
#         reward_provider += 100 * average_user_satisfaction  # 만족도 보상
#     else:
#         reward_provider -= 30 * (satisfaction_threshold - average_user_satisfaction)  # 만족도 페널티

#     return reward_provider

# def ev_reward(env):
#     reward_ev = 0

#     # Battery health reward
#     soc_values = [ev.get_soc() for cs in env.charging_stations for ev in cs.evs_connected if ev is not None]
#     avg_battery_health = np.mean(soc_values) if soc_values else 0
#     reward_ev += avg_battery_health * 0.1  # Battery health reward

#     # Peak hour determination
#     peak_hour = is_peak_hour(env.current_step)

#     # Charging and discharging rewards
#     for cs in env.charging_stations:
#         for port in range(cs.n_ports):
#             #print("current_step : ", env.current_step)
#             current_signal = get_current_signal(env, port, cs.id, env.current_step) ###<-
#             is_charging = current_signal > 0
#             is_discharging = current_signal < 0

#             energy_exchanged = abs(current_signal)  # Absolute value of current signal

#             if peak_hour and is_discharging:
#                 reward_ev += energy_exchanged * 0.1  # Reward for discharging during peak hours
#             if not peak_hour and is_charging:
#                 reward_ev += energy_exchanged * 0.1  # Reward for charging during non-peak hours

#     # Penalties and rewards based on energy exchanged and SOC
#     for cs in env.charging_stations:
#         total_profits = cs.total_profits
#         reward_ev += total_profits * 0.1
#         for port in range(cs.n_ports):
#             ev = cs.evs_connected[port]
#             if ev is not None:
#                 energy_exchanged = ev.total_energy_exchanged
#                 reward_ev -= energy_exchanged * 0.0005  # 배터리 교환량 페널티
                
#                 if ev.get_soc() > 0.8:
#                     reward_ev -= 0.01  # SOC 보상

#     # Price penalties
#     for cs in env.charging_stations:
#         if 0 <= env.current_step - 1 < env.charge_prices.shape[1] and 0 <= env.current_step - 1 < env.discharge_prices.shape[1]:
#             current_charge_price = env.charge_prices[cs.id, env.current_step-1]
#             current_discharge_price = env.discharge_prices[cs.id, env.current_step-1]
#         else:
#             current_charge_price = 0
#             current_discharge_price = 0

#         reward_ev -= (current_charge_price - current_discharge_price) * 0.02  # Price difference penalty

#     # Average power output reward
#     avg_power_output = np.mean([cs.current_power_output for cs in env.charging_stations])
#     reward_ev += avg_power_output * 0.01  # Reward for average power output

#     return reward_ev

# def CombinedRewardFunction(env, total_costs, user_satisfaction_list, *args):
#     w_grid=0.5
#     w_provider=0.5
#     w_ev=0.5

#     reward_grid = grid_reward(env)
#     reward_provider = provider_reward(env, user_satisfaction_list)
#     reward_ev = ev_reward(env)  

#     # 종합 보상 계산
#     total_reward = w_grid * reward_grid + w_provider * reward_provider + w_ev * reward_ev

#     return total_reward