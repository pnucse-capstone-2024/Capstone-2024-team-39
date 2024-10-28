from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # select charging station HTML

@app.route('/load_data', methods=['POST'])
def load_data():
    date_input = request.form['date']
    file_name = f"simulation/Simulation_{date_input.replace('-', '')}.csv"
    
    if not os.path.exists(file_name):
        return jsonify({"error": "File not found"}), 404

    df = pd.read_csv(file_name)
    charging_stations = df['CS Id'].unique().tolist()
    return jsonify({"charging_stations": charging_stations})

@app.route('/get_soc', methods=['POST'])
def get_soc():
    date_input = request.form['date']
    cs_id = request.form['cs_id']
    file_name = f"simulation/Simulation_{date_input.replace('-', '')}.csv"
    
    df = pd.read_csv(file_name)
    
    # convert datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # charging station data filtering
    filtered_data = df[df['CS Id'] == int(cs_id)]

    # convert data type to float
    filtered_data.loc[:, 'Charging Price'] = pd.to_numeric(filtered_data['Charging Price'], errors='coerce')
    filtered_data.loc[:, 'Discharging Price'] = pd.to_numeric(filtered_data['Discharging Price'], errors='coerce')
    filtered_data.loc[:, 'SoC'] = pd.to_numeric(filtered_data['SoC'], errors='coerce')

    # maximum profit calculation 
    algorithms = filtered_data['Algorithm Name'].unique()
    max_profit_data = None
    max_profit = float('-inf')

    for algorithm in algorithms:
        algo_data = filtered_data[filtered_data['Algorithm Name'] == algorithm]
        chargingProfit = 0
        dischargingProfit = 0

        for index in range(1, len(algo_data)):
            soc_change = algo_data['SoC'].iloc[index] - algo_data['SoC'].iloc[index - 1]
            if soc_change > 0:  # charging
                chargingProfit += soc_change * algo_data['Charging Price'].iloc[index]
            else:  # discharging
                dischargingProfit += abs(soc_change) * algo_data['Discharging Price'].iloc[index]

        total_profit = chargingProfit - dischargingProfit

        if total_profit > max_profit:
            max_profit = total_profit
            max_profit_data = algo_data

    # return result list
    time_series = max_profit_data['Timestamp'].dt.strftime('%H:%M:%S').tolist()
    soc_series = max_profit_data['SoC'].tolist()
    charging_price_series = max_profit_data['Charging Price'].tolist() 
    discharging_price_series = max_profit_data['Discharging Price'].tolist() 

    return jsonify({
        "time_series": time_series,
        "soc_series": soc_series,
        "charging_price": charging_price_series, 
        "discharging_price": discharging_price_series, 
    })


@app.route('/result')
def result():
    return render_template('result.html')  # result HTML

if __name__ == '__main__':
    app.run(debug=True)
