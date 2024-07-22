import os
import re
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from All_sensor_names_v2 import SPFC_receiver_spf2_sensor_names

# Base directory containing the test data
base_directory = r'C:\Users\agould\Alphawave Services\EA Production - SARAO - SARAO\DocumentControl\Test data\317-022000'

# Function to get the latest revision directory for a given serial number
def get_latest_revision_dir(serial_number, intermediate_dir):
    serial_path = os.path.join(base_directory, serial_number, intermediate_dir)
    if os.path.isdir(serial_path):
        revision_dirs = [d for d in os.listdir(serial_path) if re.match(r"Rev_\d+", d)]
        if revision_dirs:
            latest_revision_dir = max(revision_dirs, key=lambda x: int(re.search(r'\d+', x).group()))
            return os.path.join(serial_path, latest_revision_dir)
    return None

# Extract and process the .log file from a .tgz archive
def extract_and_process_log_file(tgz_file):
    with tarfile.open(tgz_file, 'r:gz') as tar:
        log_file_name = [m for m in tar.getnames() if m.endswith('MASTER_LOG.log')][0]
        tar.extract(log_file_name, path='extracted_logs')
        log_file_path = os.path.join('extracted_logs', log_file_name)
        return process_log_file(log_file_path)

# Process a single .log file
def process_log_file(file_path):
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            time = parts[0]
            sensor_data = parts[1:]
            parsed_data = {'Time': pd.to_numeric(time)}
            for sd in sensor_data:
                num, value = sd.split(':')
                sensor_name = SPFC_receiver_spf2_sensor_names.get(bytes(num, 'utf-8'), f'unknown_{num}')
                parsed_data[sensor_name] = float(value)
            data.append(parsed_data)

    data = pd.DataFrame(data)
    return data

# Main loop to process and plot all .log files
serial_numbers = [f"SN4A10{str(i).zfill(2)}" for i in range(1, 21)]
intermediate_dir = "317-022000-010"

all_data = []
stats = []

for serial in serial_numbers:
    latest_revision_dir = get_latest_revision_dir(serial, intermediate_dir)
    if latest_revision_dir:
        tgz_file_path = os.path.join(latest_revision_dir, 'MASTER_LOG.tgz')
        if os.path.isfile(tgz_file_path):
            print(f"Processing {tgz_file_path}...")
            data = extract_and_process_log_file(tgz_file_path)
            if not data.empty:
                all_data.append((serial, data))
            else:
                print(f"No data found in {tgz_file_path}")

# Ensure we have data to plot
if all_data:
    plt.figure(figsize=(15, 10))

    for serial, data in all_data:
        # Convert 'Time' column to datetime format
        data['Time'] = pd.to_datetime(data['Time'], unit='s', utc=True)

        # Ensure no duplicate time values
        data = data.drop_duplicates(subset='Time')

        if not data.empty:
            # Ensure necessary columns are present
            required_columns = ['b2ColdheadStage1Temp', 'b2LnaTemp', 'b2LnaHDrainVoltage1', 'b2LnaHDrainCurrent1',
                                'b2LnaHDrainVoltage2', 'b2LnaHDrainCurrent2', 'b2LnaHDrainVoltage3', 'b2LnaHDrainCurrent3',
                                'b2LnaVDrainVoltage1', 'b2LnaVDrainCurrent1', 'b2LnaVDrainVoltage2', 'b2LnaVDrainCurrent2',
                                'b2LnaVDrainVoltage3', 'b2LnaVDrainCurrent3', 'b2CryoMotorSpeed']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                print(f"Missing columns in data for {serial}: {missing_columns}")
                continue  # Skip this dataset if required columns are missing

            # Align data based on the first time the temperature reaches 273K
            stage2_temp = data['b2LnaTemp'].values
            first_above_273_idx = np.argmax(stage2_temp <= 273)

            if first_above_273_idx == 0 and stage2_temp[0] < 273:
                print(f"Stage 2 temperature never reaches 273K for {serial}")
                continue  # Skip this dataset if it never reaches 273K

            start_time = data['Time'].iloc[first_above_273_idx] - timedelta(seconds=3600)
            data = data[data['Time'] >= start_time]

            # Extract the relevant columns as arrays
            time = data['Time'].values
            stage1_temp = data['b2ColdheadStage1Temp'].values
            stage2_temp = data['b2LnaTemp'].values
            heater2_power = (
                data['b2LnaHDrainVoltage1'].values * data['b2LnaHDrainCurrent1'].values +
                data['b2LnaHDrainVoltage2'].values * data['b2LnaHDrainCurrent2'].values +
                data['b2LnaHDrainVoltage3'].values * data['b2LnaHDrainCurrent3'].values +
                data['b2LnaVDrainVoltage1'].values * data['b2LnaVDrainCurrent1'].values +
                data['b2LnaVDrainVoltage2'].values * data['b2LnaVDrainCurrent2'].values +
                data['b2LnaVDrainVoltage3'].values * data['b2LnaVDrainCurrent3'].values
            )
            RPM_condition = data['b2CryoMotorSpeed'].values  # Placeholder for actual RPM condition

            # Calculate gradients of the RMS signal
            time_seconds = (data['Time'] - data['Time'].iloc[0]).astype('timedelta64[s]').values.astype(np.float64)

            # Calculate gradients in K/s using np.gradient
            window_size = 41
            gradients_stage2 = np.gradient(stage2_temp, time_seconds)
            gradients_stage2 *= 60
            gradients_stage2Smoothed = pd.Series(gradients_stage2).rolling(window=window_size, min_periods=1).mean().values
            gradients_stage1 = np.gradient(stage1_temp, time_seconds)
            gradients_stage1 *= 60
            gradients_stage1Smoothed = pd.Series(gradients_stage1).rolling(window=window_size, min_periods=1).mean().values

            # Calculate gradient difference for stability plot
            def gradient_difference(windowSize, Data):
                diff = []
                for i in range(len(Data)):
                    first_max = np.amax(Data[i:i + windowSize])
                    last_min = np.amin(Data[i:i + windowSize])
                    diff.append(first_max - last_min)
                return np.array(diff)

            stability_grad_diffStage1 = gradient_difference(60, stage1_temp)
            stability_grad_diffStage2 = gradient_difference(60, stage2_temp)

           # Calculate statistics
            ultimate_temp_stage2 = 18
            ultimate_temp_stage1 = 50
            
            time_to_ultimate_stage2 = time_seconds[np.argmax(stage2_temp <= ultimate_temp_stage2)]
            time_to_ultimate_stage1 = time_seconds[np.argmax(stage1_temp <= ultimate_temp_stage1)]
            time_to_273K_stage2 = time_seconds[np.argmax(stage2_temp <= 273)]
            time_to_273K_stage1 = time_seconds[np.argmax(stage1_temp <= 273)]
            
            cooldown_gradient_stage2 = (ultimate_temp_stage2 - 273) / (time_to_ultimate_stage2 - time_to_273K_stage2)
            cooldown_gradient_stage1 = (ultimate_temp_stage1 - 273) / (time_to_ultimate_stage1 - time_to_273K_stage1)

            specline_start_idx = np.argmax(stage1_temp <= ultimate_temp_stage1)
            specline_intersections = np.sum(np.abs(gradients_stage2[specline_start_idx:]) > 0.3)

            stats.append({
                'Serial': serial,
                'Time to Ultimate Temperature (s)': time_seconds[np.argmax(stage2_temp <= ultimate_temp_stage2)],
                'Ultimate Temperature Reached Stage 2 (K)': ultimate_temp_stage2,
                'Ultimate Temperature Reached Stage 1 (K)': ultimate_temp_stage1,
                'Cooldown Gradient Stage 2 (K/s)': cooldown_gradient_stage2,
                'Cooldown Gradient Stage 1 (K/s)': cooldown_gradient_stage1,
                'Number of Specline Intersections': specline_intersections
            })

            # Plot Temperature
            plt.figure(1)
            plt.plot(time_seconds, stage1_temp, label=f'Stage 1 Temp - {serial}')
            plt.plot(time_seconds, stage2_temp, label=f'Stage 2 Temp - {serial}')

            # Plot Gradients
            plt.figure(2)
            plt.plot(time_seconds, gradients_stage1Smoothed, label=f'Stage 1 Grad - {serial}')
            plt.plot(time_seconds, gradients_stage2Smoothed, label=f'Stage 2 Grad - {serial}')

            # Plot Grad range analysis
            plt.figure(3)
            plt.plot(time_seconds[:len(stability_grad_diffStage1)], stability_grad_diffStage1, label=f'Stage 1 Grad Diff - {serial}', color='blue')
            plt.plot(time_seconds[:len(stability_grad_diffStage2)], stability_grad_diffStage2, label=f'Stage 2 Grad Diff - {serial}', color='red')

    # Finalize and show all plots
    plt.figure(1)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature over Time')
    plt.legend()
    plt.grid(True)

    plt.figure(2)
    plt.xlabel('Time (s)')
    plt.ylabel('Gradient (K/min)')
    plt.title('Temperature Gradients over Time')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, label='+0.3 K/min line')
    plt.axhline(y=-0.3, color='orange', linestyle='--', linewidth=1, label='-0.3 K/min line')

    plt.figure(3)
    plt.xlabel('Time (s)')
    plt.ylabel('Gradient (K/min)')
    plt.title('Temperature Gradients range')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, label='+0.3 K/min line')

    plt.show()

    # Save statistics to a CSV file
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(r'C:\Users\agould\OneDrive - Alphawave Services\code\Coldhead stability\B2 stats\statistics.csv', index=False)
    

    # Print summary statistics
    summary = stats_df.describe()
    summary.to_csv(r'C:\Users\agould\OneDrive - Alphawave Services\code\Coldhead stability\B2 stats\summarystatistics.csv')
else:
    print("No valid log files found.")