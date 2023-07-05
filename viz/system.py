from absl import app
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import numpy as np

from absl import flags

_SMOOTHING_WINDOW_SIZE = flags.DEFINE_integer('smoothing_window_size', 1, 'Size of the window for the moving average')
_BASE_DIR = flags.DEFINE_string('base_dir', '../logs/circuit_training/debug/64_cores_3_nodes_1_gpu_test/',
                                'Base directory where CSV files are located')
_SUBTRACT_BASELINE = flags.DEFINE_boolean('subtract_baseline', False,
                                          'Whether or not to subtract the baseline from the data')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', 'output_plots', 'Directory to save the plots to')


def main(argv):
    del argv  # Unused.

    units = {
        # 'duration': 's',
        'cpu_power': 'Watts',
        'gpu_power': 'Watts',
        'ram_power': 'Watts',
        'cpu_energy': 'kWh',
        'gpu_energy': 'kWh',
        'ram_energy': 'kWh',
        'energy_consumed': 'kWh',
        'ram_process': 'GB'
    }

    dir_list = glob.glob(_BASE_DIR.value + 'circuit_training_num_collect_jobs_16_seed_*')

    # Initialize an empty DataFrame to hold all of the CSV data
    all_data = pd.DataFrame()
    all_data_minus_baseline = pd.DataFrame()
    duration_vals = []
    # Loop over the directories
    for dir in dir_list:
        # Define the CSV file path
        csv_file_path = dir + '/metrics/train_emissions.csv'
        baseline_csv_file_path = dir + '/metrics/baseline_emissions.csv'
        seed = dir.split('seed_')[1]
        seed = seed.split('/')[0]

        if os.path.exists(csv_file_path):
            # Load the data from the CSV file
            df = pd.read_csv(csv_file_path)

            df['steps'] = range(0, len(df['duration']))
            df['seed'] = int(seed)
            duration_vals.append(df['duration'].max())
            # check the earliest and latest timestamp for the df
            print('Logging data for seed: ' + seed)
            print(f'Earliest timestamp: {df["timestamp"].min()}')
            print(f'Latest timestamp: {df["timestamp"].max()}')

            # convert these to timestamps and subtract to get the duration
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f'Duration: {df["timestamp"].max() - df["timestamp"].min()}')

            # Fix the energy columns by creating diff columns
            df['cpu_energy_total'] = df['cpu_energy']
            df['gpu_energy_total'] = df['gpu_energy']
            df['ram_energy_total'] = df['ram_energy']

            df['cpu_energy'] = df['cpu_energy_total'].diff()
            df['gpu_energy'] = df['gpu_energy_total'].diff()
            df['ram_energy'] = df['ram_energy_total'].diff()

            df['computed_cpu_energy'] = df['cpu_power'] * df['duration']
            df['computed_gpu_energy'] = df['gpu_power'] * df['duration']
            df['computed_ram_energy'] = df['ram_power'] * df['duration']

            all_data = pd.concat([all_data, df], ignore_index=True, copy=False)
            if _SUBTRACT_BASELINE.value and os.path.exists(baseline_csv_file_path):
                baseline_df = pd.read_csv(baseline_csv_file_path)
                baseline_df['steps'] = range(0, len(baseline_df['duration']))
                avgs = baseline_df[units.keys()].mean()
                df_minus_baseline = df.subtract(avgs, axis='columns')
                all_data_minus_baseline = pd.concat([all_data_minus_baseline, df_minus_baseline], ignore_index=True,
                                                    copy=False)
            else:
                print(f'Baseline CSV file not found at {baseline_csv_file_path}')
        else:
            raise Exception(f'CSV file not found at {csv_file_path}')

    # trim the dataframe so that all seeds have the same number of steps
    max_step_count = all_data.groupby('seed')['steps'].max().min()
    all_data = all_data[all_data['steps'] <= max_step_count]

    # Considering 'units.keys()' as the list of columns you want to calculate mean and std
    statistics = all_data[list(units.keys())].agg([np.mean, np.std])

    # Print the statistics in the format "mean ± std"
    for col in statistics.columns:
        mean = statistics.loc['mean', col]
        std = statistics.loc['std', col]
        unit = units[col]

        # Convert the column name to CamelCase
        col_camel_case = ''.join(word.title() for word in col.split('_'))

        print(f"{col_camel_case} ({unit}): {mean:.10f} ± {std:.10f}")
    # print duration with mean std
    print(f"Duration (m): {np.mean(duration_vals) / 60:.2f} ± {np.std(duration_vals) / 60:.2f}")

    return
    to_plot = list(units.keys())  # Get the keys (metric names) from the units dictionary

    if not os.path.exists(_OUTPUT_DIR.value):
        os.makedirs(_OUTPUT_DIR.value)

    window_size = _SMOOTHING_WINDOW_SIZE.value
    for column in to_plot:
        # Calculate the moving average for the current column
        all_data[f'{column}_smoothed'] = all_data.groupby('seed')[column].rolling(window_size).mean().reset_index(0,
                                                                                                                  drop=True)

        # Create a display name by replacing underscores with spaces and capitalizing each word
        display_name = column.replace('_', ' ').title()

        sns.lineplot(data=all_data, x='steps', y=f'{column}', errorbar='sd', color='blue')
        plt.title(display_name)
        plt.xlabel('Measurement Step (1 step = 1 second)')
        plt.ylabel(f'{display_name} ({units[column]})')  # Add units to the y-axis label
        plt.savefig(f'{_OUTPUT_DIR.value}/{column}.png')  # Save the plot to the output directory
        plt.clf()  # Clear the current plot before generating the next one


if __name__ == '__main__':
    app.run(main)
