import pickle
import pandas as pd
import datetime
import os

def Simulation_csv(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the charging and discharging prices of the EVs along with SoC and average price
    and save the data into a CSV file.
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    # data list
    data = []

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(minutes=env.timescale),
                                   freq=f'{env.timescale}min')

        for cs in env.charging_stations:
            charge_prices = env.charge_prices[index]  # Assuming prices are stored in replay
            discharge_prices = env.discharge_prices[index]
            average_price = env.average_price     # Assuming average_price is a float
            
            # data for each port
            for port in range(cs.n_ports):
                soc_data = env.port_energy_level[port, cs.id, :]  # Assuming this contains SoC data

                for time_index in range(len(date_range)):
                    timestamp = date_range[time_index]

                    data.append([
                        date_range[time_index],                  # 시간
                        timestamp.date(),                        # 날짜 (YYYY-MM-DD)
                        timestamp.time(),                        # 시간 (HH:MM)
                        charge_prices[time_index],               # 충전가
                        discharge_prices[time_index],            # 방전가
                        average_price,                           # 평균가격
                        cs.id,                                   # cs.id
                        algorithm_names[index],                  # 알고리즘 이름
                        soc_data[time_index]                     # 해당 포트에서 해당 알고리즘 SoC 정보
                    ])

    # convert DataFrame
    df = pd.DataFrame(data, columns=[
        'Timestamp', 'Date', 'Time', 'Charging Price', 'Discharging Price', 'Average Price', 'CS Id', 'Algorithm Name', 'SoC'
    ])

    # save CSV file
    if save_path:
        first_date = df['Date'].iloc[0]  
        simulation_date = first_date.strftime('%Y%m%d')  # YYYYMMDD format
        csv_file_path = os.path.join(save_path, f'Simulation_{simulation_date}.csv')
        df.to_csv(csv_file_path, index=False)
        print(f"Saved data to {csv_file_path}")


pkl_file_path = "./results/231009_eval_20cs_1tr_V2GProfitMax_9_algos_50_exp_2024_10_27_894053/plot_results_dict.pkl"
result_path = "./results/231009_eval_20cs_1tr_V2GProfitMax_9_algos_50_exp_2024_10_27_894053"

Simulation_csv(pkl_file_path, result_path)
