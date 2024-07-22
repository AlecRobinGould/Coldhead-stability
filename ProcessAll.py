import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def test(filepath):
    file_path = filepath
    # Read the data from the file
    data = pd.read_csv(file_path, sep='\t', skiprows=1)

    # Drop unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Convert the 'Time' column to datetime format
    data['Time'] = pd.to_datetime(data['Time'], format='%b %d %Y, %H:%M:%S')

    # # Filter out temperatures exceeding 400K
    # data = data[data['Temp2'] <= 400]

    # # Filter out data where 'Temp2' is less than or equal to 300
    # data = data[data['Temp2'] > 300]

    # Extract the relevant columns as numpy arrays
    time = data['Time'].values
    stage1_temp = data['Temp1'].values
    stage2_temp = data['Temp2'].values
    RPM_condition = data['Cryo Speed'].values
    heater1_power = data['Load1'].values
    heater2_power = data['Load2'].values

    # Identify the first index where RPM goes from 72 to zero
    first_zero_index = None
    for i in range(1, len(RPM_condition)):
        if RPM_condition[i] == 0 and RPM_condition[i - 1] == 72:
            first_zero_index = i
            break

    # If such an index is found, filter the data up to that point
    if first_zero_index is not None:
        time = time[:first_zero_index]
        stage1_temp = stage1_temp[:first_zero_index]
        stage2_temp = stage2_temp[:first_zero_index]
        RPM_condition = RPM_condition[:first_zero_index]
        heater1_power = heater1_power[:first_zero_index] * 10
        heater2_power = heater2_power[:first_zero_index] * 10

    # Calculate the difference between Temp1 and Temp2 temperatures
    temp_diff = stage1_temp - stage2_temp

    # Find where the difference changes sign (i.e., where the temperatures intersect)
    sign_change_indices = np.where(np.diff(np.sign(temp_diff)))[0]

    # Extract the intersection points
    intersection_times = time[sign_change_indices]
    intersection_temps = stage1_temp[sign_change_indices]

    # Define block size and time limit for 3 hours
    block_size = 1
    time_limit = 3600 * 2

    # Perform RMS on the temperature data
    def moving_rms(signal, window_size):
        squared_signal = pd.Series(signal) ** 2
        mean_squared = squared_signal.rolling(window=window_size, min_periods=1, center=True).mean()
        rms = np.sqrt(mean_squared)
        return rms

    rmsSignal = moving_rms(stage2_temp, 41).values
    # __________________________REMOVE THE NAN & INTERPOLATE_______________________________
    rmsSignal = pd.Series(rmsSignal).interpolate().bfill().ffill().values

    # Ensure no duplicate time values
    data = data.drop_duplicates(subset='Time')

    # Calculate gradients of the RMS signal
    time_seconds = (data['Time'] - data['Time'].iloc[0]).dt.total_seconds().values

    # Ensure lengths of time_seconds and other relevant arrays match
    min_length = min(len(time_seconds), len(rmsSignal), len(stage2_temp), len(stage1_temp), len(RPM_condition), len(heater1_power), len(heater2_power))
    time_seconds = time_seconds[:min_length]
    rmsSignal = rmsSignal[:min_length]
    stage2_temp = stage2_temp[:min_length]
    stage1_temp = stage1_temp[:min_length]
    RPM_condition = RPM_condition[:min_length]
    heater1_power = heater1_power[:min_length]
    heater2_power = heater2_power[:min_length]

    # Calculate gradients in K/s using np.gradient
    gradients_stage2 = np.gradient(stage2_temp, time_seconds)
    gradients_stage2 *= 60
    gradients_stage1 = np.gradient(stage1_temp, time_seconds)
    gradients_stage1 *= 60

    # Update min_length after computing gradients
    min_length = min(len(time_seconds), len(gradients_stage2))

    # Ensure lengths of time_seconds and gradients match
    time_seconds = time_seconds[:min_length]
    gradients_stage2 = gradients_stage2[:min_length]
    gradients_stage1 = gradients_stage1[:min_length]

    window_size = 60
    gradients_moving_avg = pd.Series(gradients_stage2).rolling(window=window_size, min_periods=1).mean().values
    gradients_stage1_moving_avg = pd.Series(gradients_stage1).rolling(window=window_size, min_periods=1).mean().values

    # Find the minimum gradient within the first 6 hours
    if len(gradients_moving_avg) > time_limit:
        I = np.argmin(gradients_moving_avg[:time_limit])
    else:
        I = np.argmin(gradients_moving_avg)

    # Find the first instance where the gradient is greater than or equal to -0.002 K/s after index I
    for q in range(I, (I + time_limit)):
        if abs(gradients_moving_avg[q]) <= 0.002 * 60:
            I2 = q
            break

    # Factor of how long it should take to Apex. B2 is 2    
    apexFactor = 2
    # Calculate the apex index
    apex_index = I + int((I2 / apexFactor))

    # Find ranges in which RPM_condition values change
    rpm_ranges = []
    start_index = 0
    current_value = RPM_condition[0]

    for i in range(1, len(RPM_condition)):
        if RPM_condition[i] != current_value:
            rpm_ranges.append((start_index, i - 1, current_value))
            start_index = i
            current_value = RPM_condition[i]

    # Add the last range
    rpm_ranges.append((start_index, len(RPM_condition) - 1, current_value))

    # Define a 1-hour window for stability check (3600 seconds)
    stability_period = 3600 * 2  # 1 hour

    def longGradient(windowSize, Data, period):
        # Calculate the number of samples corresponding to the given period
        num_samples = int(period / np.mean(np.diff(time_seconds)))
        
        # Ensure the windowSize is not larger than num_samples
        windowSize = min(windowSize, num_samples // 2)
        
        # Calculate the average of the first windowSize samples
        first_avg = np.mean(Data[:windowSize])
        
        # Calculate the average of the last windowSize samples
        last_avg = np.mean(Data[-windowSize:])
        
        # Calculate the difference between the averages
        gradient = last_avg - first_avg
        
        return gradient

    # Calculate the average cooldown gradient for the 2nd stage
    start_2nd_stage = np.where(stage2_temp <= 273)[0][0]
    end_2nd_stage = np.where(stage2_temp <= 15)[0][0]
    avg_cooldown_gradient_2nd_stage = 60 * (15 - 273) / (end_2nd_stage - start_2nd_stage)
    # Calculate the average cooldown gradient for the 1st stage
    start_1st_stage = np.where(stage1_temp <= 273)[0][0]
    end_1st_stage = np.where(stage1_temp <= 50)[0][0]
    avg_cooldown_gradient_1st_stage = 60 * (50 - 273) / (end_1st_stage - start_1st_stage)

    # Calculate gradient difference for stability plot
    def gradient_difference(windowSize, Data):
        diff = []
        for i in range(len(Data)):
            first_max = np.amax(Data[i:i + windowSize])
            last_min = np.amin(Data[i:i + windowSize])
            diff.append(first_max - last_min)
        return np.array(diff)

    # Stability plot data
    stability_grad_diffStage2 = gradient_difference(60, stage2_temp)
    stability_grad_diffStage1 = gradient_difference(60, stage1_temp)
    stability_time = time_seconds[:len(stability_grad_diffStage2)]

    specLineStage1 = 0.3
    specLineStage2 = 0.8

    # Function to calculate intersections based on the given logic
    def count_intersections(signal, spec_line):
        count = 0
        above = False
        for i in range(1, len(signal)):
            if signal[i] > spec_line+0.05 and not above:
                above = True
            elif signal[i] < spec_line-0.05 and above:
                count += 1
                above = False
        return count

    # Function to calculate ultimate temperature
    def calculate_ultimate_temperature(temperature_data, time_data, min_samples=10, spec = 45):
        sorted_indices = np.argsort(temperature_data)
        sorted_temps = temperature_data[sorted_indices]
        sorted_times = time_data[sorted_indices]
        
        temp_counts = pd.Series(sorted_temps).value_counts()
        
        for temp in temp_counts.index:
            if temp < spec:
                if temp_counts[temp] >= min_samples:
                    ultimate_temp = temp
                    ultimate_time = sorted_times[np.where(sorted_temps == temp)[0][0]]
                    return ultimate_temp, ultimate_time
        return np.nan, np.nan
    # Calculate ultimate temperatures and times
    UltimateTempStage1, TimeUltimateTempStage1 = calculate_ultimate_temperature(stage1_temp, time, min_samples=50, spec = 45)
    UltimateTempStage2, TimeUltimateTempStage2 = calculate_ultimate_temperature(stage2_temp, time, min_samples=50, spec = 15)

    TimeUltimateTempStage1 = np.where(stage1_temp <= UltimateTempStage1)[0][0]
    TimeUltimateTempStage2 = np.where(stage2_temp <= UltimateTempStage2)[0][0]


    # Calculate the standard deviation for both stages of stability_grad_diff after 30 minutes (1800 seconds)
    start_time_30min = 3600  # 30 minutes in seconds

    # Filter data for the condition where heaters' power is greater than zero
    valid_indices = np.where((heater1_power > 0) & (heater2_power > 0))[0]
    valid_times = time_seconds[valid_indices]

    # Filter data for 30 minutes after valid times
    filtered_stage1 = []
    filtered_stage2 = []

    for t in valid_times:
        start_idx = np.where(time_seconds >= t + start_time_30min)[0]
        if len(start_idx) > 0:
            start_idx = start_idx[0]
            if start_idx < len(stability_grad_diffStage1):
                filtered_stage1.append(stability_grad_diffStage1[start_idx])
            if start_idx < len(stability_grad_diffStage2):
                filtered_stage2.append(stability_grad_diffStage2[start_idx])

    filtered_stage1 = np.array(filtered_stage1)
    filtered_stage2 = np.array(filtered_stage2)

    meanStage1 = np.mean(filtered_stage1)
    meanStage2 = np.mean(filtered_stage2)

    std_stage1 = np.std(filtered_stage1)
    std_stage2 = np.std(filtered_stage2)

    # Filter data for 30 minutes after valid times
    start_idx = np.where(time_seconds >= valid_times[0] + start_time_30min)[0][0]

    # Calculate number of intersections after the start index
    num_intersections_Stage2 = count_intersections(stability_grad_diffStage2[start_idx:], std_stage2 * 3)
    num_intersections_Stage1 = count_intersections(stability_grad_diffStage1[start_idx:], std_stage1 * 3)


    # Plotting each figure and saving them
    figures_dir = os.path.dirname(filepath)
    # ______________________________________________________________________________________________
    # Main cooldown plot
    # ______________________________________________________________________________________________

    # Plot Stage 1 temp, Stage 2 temp, RMS smoothed Stage 2 temp, heater power, and RPM vs Time
    plt.figure(figsize=(14, 7))

    # Plot Stage 1 temp
    plt.plot(time[:min_length], stage1_temp[:min_length], label='Stage 1 Temp', color='blue')
    # Plot Stage 2 temp
    plt.plot(time[:min_length], stage2_temp[:min_length], label='Stage 2 Temp', color='red')
    # Plot RMS signal
    plt.plot(time[:min_length], rmsSignal[:min_length], label='RMS Signal', color='black')

    # Plot heater power
    plt.plot(time[:min_length], heater1_power[:min_length], label='Heater 1 Power / 1000', color='purple', linestyle='--')
    plt.plot(time[:min_length], heater2_power[:min_length], label='Heater 2 Power / 1000', color='orange', linestyle='--')

    # Mark the apex start, end, and the apex point
    plt.axvline(x=time[I], color='r', linestyle='--', linewidth=1, label='Apex Start')
    plt.axvline(x=time[I + I2], color='g', linestyle='--', linewidth=1, label='Apex End')
    plt.axvline(x=time[I2], color='grey', linestyle='--', linewidth=1, label='I2')
    plt.axvline(x=time[apex_index], color='b', linestyle='--', linewidth=1, label='Apex')
    # Mark the intersection points
    plt.scatter(intersection_times, intersection_temps, color='orange', zorder=5, label='Intersections')
    # Mark the apex point
    plt.scatter(time[apex_index], stage2_temp[apex_index], color='purple', zorder=6, label='Apex', s=100)
    # Plot RPM
    plt.plot(time[:min_length], RPM_condition[:min_length], label='RPM', color='brown', linestyle='--')

    # Adding titles and labels
    plt.title('Stage 1 and Stage 2 Temperatures Over Time with Heater Power and RPM')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    # Display the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'figure1.png'))
    plt.close()

    # ______________________________________________________________________________________________
    # Gradient minmax
    # ______________________________________________________________________________________________
    plt.figure(figsize=(14, 7))
    # Plot gradient difference
    plt.plot(stability_time, stability_grad_diffStage2, label='Gradient Difference Stage 2', color='blue')
    # Overlay horizontal lines at +1K/min and -1K/min

    plt.axvline(x=valid_times[0]+start_time_30min, color='k', linestyle='--', label='Post heat stable')

    # plt.axhline(y=specLineStage2, color='g', linestyle='--', label='Spec Line Stage 2')
    plt.axhline(y=std_stage2 +meanStage2, color='b', linestyle=':', label='1 Std Dev Stage 2')
    plt.axhline(y=(3 * std_stage2) + meanStage2, color='r', linestyle=':', label='3 Std Dev Stage 2')
    # Adding titles and labels
    plt.title('Gradient Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Gradient (K/min)')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'figure2.png'))
    plt.close()

    plt.figure(figsize=(14, 7))
    # Plot gradient difference
    plt.plot(stability_time, stability_grad_diffStage1, label='Gradient Difference Stage 1', color='blue')
    # Overlay horizontal lines at +1K/min and -1K/min
    plt.axvline(x=valid_times[0]+start_time_30min, color='k', linestyle='--', label='Post heat stable')
    # plt.axhline(y=specLineStage1, color='g', linestyle='--', label='Spec Line Stage 1')
    plt.axhline(y=std_stage1+meanStage1, color='r', linestyle=':', label='1 Std Dev Stage 1')
    plt.axhline(y=(3 * std_stage1) + meanStage1, color='b', linestyle=':', label='3 Std Dev Stage 1')

    # Adding titles and labels
    plt.title('Gradient Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Gradient (K/min)')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'figure3.png'))
    plt.close()

    # ______________________________________________________________________________________________
    # Gradient (Stability)
    # ______________________________________________________________________________________________
    plt.figure(figsize=(14, 7))

    # Plot gradients (Stage 2)
    plt.plot(time_seconds[:len(gradients_stage2)], gradients_moving_avg, label='Gradient (Stage 2)', color='green')
    # Overlay horizontal lines at +1K/min and -1K/min
    plt.axhline(y=1.0 , color='r', linestyle='--', linewidth=1, label='+1 K/min')
    plt.axhline(y=-1.0, color='r', linestyle='--', linewidth=1, label='-1 K/min')
    # Adding titles and labels
    plt.title('Stability Plot Stage 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Gradient (K/min)')
    plt.legend()
    # Display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'figure4.png'))
    plt.close()

    plt.figure(figsize=(14, 7))

    # Plot gradients (Stage 2)
    plt.plot(time_seconds[:len(gradients_stage1)], gradients_stage1_moving_avg, label='Gradient (Stage 1)', color='green')
    # Overlay horizontal lines at +1K/min and -1K/min
    plt.axhline(y=0.3 , color='r', linestyle='--', linewidth=1, label='+0.3 K/min')
    plt.axhline(y=-0.3, color='r', linestyle='--', linewidth=1, label='-0.3 K/min')
    # Adding titles and labels
    plt.title('Stability Plot Stage 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Gradient (K/min)')
    plt.legend()
    # Display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'figure5.png'))
    plt.close()

    # Define the log file path
    log_file_path = os.path.join(os.path.dirname(filepath), 'TestResults.log')

    # Collect results to log
    results = []

    results.append(f"Apex point (Time, Temperature): {time[apex_index]}: {stage2_temp[apex_index]}°C")
    results.append(f"Average cooldown gradient for the 1st stage: {avg_cooldown_gradient_1st_stage} K/s")
    results.append(f"Average cooldown gradient for the 2nd stage: {avg_cooldown_gradient_2nd_stage} K/s")
    results.append(f"1st stage ultimate temperature: {UltimateTempStage1}")
    results.append(f"2nd stage ultimate temperature: {UltimateTempStage2}")
    results.append(f"Number of intersections (Stage 1) at {specLineStage1} K/min: {num_intersections_Stage1}")
    results.append(f"Number of intersections (Stage 2) at {specLineStage2} K/min: {num_intersections_Stage2}")
    results.append(f"1st stage 3 SD: {3 * std_stage1}")
    results.append(f"2nd stage 3 SD: {3 * std_stage2}")

    # Print results to console
    for result in results:
        print(result)

    # Log results to file
    log_results(log_file_path, *results)

def log_results(log_file_path, *args):
    with open(log_file_path, 'a') as log_file:
        for arg in args:
            log_file.write(f"{arg}\n")

def get_files_in_folders(directory, extension=".txt"):
    """
    Get a list of all files in the given directory and its subdirectories
    with the specified extension (e.g., .txt).
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Use the function to process each text file in the folder
def process_files_in_directory(directory):
    # Get the list of files in the directory
    files = get_files_in_folders(directory)

    # Process each file using the 'test' function
    for file in files:
        print(file)
        test(file)

# Call the function with the appropriate directory path
directory_path = "C:/Users/agould/OneDrive - Alphawave Services/code/Coldhead stability/29 coldheads/GM Coldhead test document"
process_files_in_directory(directory_path)