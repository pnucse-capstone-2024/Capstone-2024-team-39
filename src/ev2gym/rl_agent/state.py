'''  This file contains various example state functions for the RL agent '''
import math
import numpy as np


def CombinedStateFunction(env, *args):
    w_grid = 1.0
    w_provider = 1.0
    w_ev = 1.0

    state = []

    # 1. State information related to the grid operator (w_grid)
    state.append(env.current_step * w_grid)
    state.append(env.current_power_usage[env.current_step - 1] * w_grid)

    # Add up to 20 charge prices
    charge_prices = abs(env.charge_prices[0, env.current_step: env.current_step + 20])
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20 - len(charge_prices)))
    state.extend(charge_prices * w_grid)

    # Add transformer-related data
    for tr in env.transformers:
        loads, pv = tr.get_load_pv_forecast(step=env.current_step, horizon=20)
        power_limits = tr.get_power_limits(step=env.current_step, horizon=20)
        state.extend((loads - pv).flatten() * w_grid)
        state.extend(power_limits.flatten() * w_grid)

    # 2. State information related to the charging service provider (w_provider)
    avg_power_output = np.mean([cs.current_power_output for cs in env.charging_stations])
    avg_overload = np.mean([tr.get_how_overloaded() for tr in env.transformers])

    state.extend([
        avg_power_output * w_provider,
        avg_overload * w_provider
    ])

    # Add setpoint
    setpoint = env.power_setpoints[env.current_step - 1] if env.current_step - 1 < env.simulation_length else np.zeros((1,))
    state.extend(setpoint.flatten() * w_provider)

    for cs in env.charging_stations:
        if cs.connected_transformer in [tr.id for tr in env.transformers]:
            for EV in cs.evs_connected:
                if EV is not None:
                    d_cal, d_cyc = EV.get_battery_degradation()
                    state.extend([
                        EV.get_soc() * w_ev,
                        (EV.time_of_departure - env.current_step) * w_ev,
                        EV.total_energy_exchanged * w_ev,
                        (env.current_step - EV.time_of_arrival) * w_ev,
                        d_cal, d_cyc
                    ])
                else:
                    state.extend([0, 0, 0, 0, 0, 0])

    # Adjust the length of the state vector (e.g., to 100)
    state = np.array(state, dtype=np.float32)
    if len(state) > 100:
        state = state[:100]
    elif len(state) < 100:
        state = np.pad(state, (0, 100 - len(state)), 'constant')

    return state


def PublicPST(env, *args):
    '''This state function is the public power setpoints
    The state is the public power setpoints
    The state is a vector '''

    state = [
        (env.current_step/env.simulation_length),
        # env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        # math.sin(env.sim_date.hour/24*2*math.pi),
        # math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    # the final state of each simulation
    # if env.current_step < env.simulation_length:        
    #     setpoint = min(env.power_setpoints[env.current_step], env.charge_power_potential[env.current_step])        
    # else:
    #     setpoint = 0       
    if env.current_step < env.simulation_length:  
        # setpoint = env.power_setpoints[env.current_step:env.current_step+10]
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = np.zeros((1))
        
    # if len(setpoint) < 10:
    #     setpoint = np.append(setpoint, np.zeros(10-len(setpoint)))
    
    state.append(setpoint)
    state.append(env.current_power_usage[env.current_step-1])

    # For every transformer
    for tr in env.transformers:
        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            1 if EV.get_soc() == 1 else 0.5,  # we know if the EV is full
                            EV.total_energy_exchanged,
                            # EV.max_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            # EV.min_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            (env.current_step-EV.time_of_arrival)
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state

def V2G_profit_max(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario.
    '''
    
    state = [
        (env.current_step),        
    ]

    state.append(env.current_power_usage[env.current_step-1])

    charge_prices = abs(env.charge_prices[0, env.current_step:
        env.current_step+20])
    
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20-len(charge_prices)))
    
    state.append(charge_prices)
    
    # For every transformer
    for tr in env.transformers:

        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state

def V2G_profit_max_loads(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario with loads
    '''
    
    state = [
        (env.current_step),        
    ]

    state.append(env.current_power_usage[env.current_step-1])

    charge_prices = abs(env.charge_prices[0, env.current_step:
        env.current_step+20])
    
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20-len(charge_prices)))
    
    state.append(charge_prices)
    
    # For every transformer
    for tr in env.transformers:
        loads, pv = tr.get_load_pv_forecast(step = env.current_step,
                                            horizon = 20)
        power_limits = tr.get_power_limits(step = env.current_step,
                                           horizon = 20)
        state.append(loads-pv)
        state.append(power_limits)
        
        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state
    
def BusinessPSTwithMoreKnowledge(env, *args):
    '''
    This state function is used for the business case scenario that requires more knowledge such as SoC and time of departure for each EV present.
    '''

    state = [
        (env.current_step) / env.simulation_length,
        #env.sim_date.weekday() / 5,
        # turn hour and minutes in sin and cos
        #math.sin(env.sim_date.hour/12*2*math.pi),
        #math.cos(env.sim_date.hour/12*2*math.pi),
    ]

    # the final state of each simulation
    if env.current_step < env.simulation_length:
        state.append(env.power_setpoints[env.current_step]) #/100
        state.append(env.charge_power_potential[env.current_step]) #/100
    else:
        state.append(env.power_setpoints[env.current_step-1]) #/100
        state.append(env.charge_power_potential[env.current_step-1]) #/100   

    for tr in env.transformers:
        state.append(tr.max_current/100)
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([#EV.total_energy_exchanged / EV.battery_capacity, #how much soc we charge
                                      #EV.max_ac_charge_power*1000 /            same EVs, no need right now
                                      #(cs.voltage*math.sqrt(cs.phases)),
                                      #EV.min_ac_charge_power*1000 /
                                      #(cs.voltage*math.sqrt(cs.phases)),
                                      EV.time_of_arrival / env.simulation_length,  # time of arrival
                                      EV.etime_of_departure / env.simulation_length,  # time of departure
                                      EV.get_soc(),  # soc
                                      #(EV.etime_of_departure - env.current_step) \
                                      #  / env.simulation_length, #remaining time
                                      #(env.current_step-EV.time_of_arrival) \
                                      #  / env.simulation_length,  # time stayed
                                      #(EV.etime_of_departure - \
                                      # EV.time_of_arrival) / env.simulation_length, # total staying time
                                      #(((EV.battery_capacity - EV.battery_capacity_at_arrival) /
                                      #  (EV.etime_of_departure - EV.time_of_arrival)) / EV.max_ac_charge_power),  # average charging speed
                                      #(((EV.battery_capacity - EV.battery_capacity_at_arrival) / EV.battery_capacity)) \
                                      #  / ((EV.etime_of_departure - env.current_step + 1) / env.simulation_length),   #charging priority
                                      #EV.required_power / EV.battery_capacity,  # required energy
                                      ])
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state