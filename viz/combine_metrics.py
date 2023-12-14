# Example:  python3 combine_metrics.py --base_dir=/home/ikechukwuu/workspace/gcs/a2perf/web_navigation/difficulty_1/0048/difficulty_level=1_env_batch_size=16_seed=37_total_env_steps=1000000/ --pattern='**/*.json'
# Locomotion: python3.9 combine_metrics.py
from absl import app
from absl import flags
import glob
import json
import os
import numpy as np
from collections import defaultdict

_BASE_DIR = flags.DEFINE_string(
    'base_dir',
    'train',
    'Root directory for train logs from all of the experiments that reliability'
    ' metrics will be calculated on',
)

# _PATTERN = flags.DEFINE_string(
#     'pattern',
#     '**',
#     'Pattern to match in the root directory',
# )

_PAT

def main(_):
    # glob for metrics_results.json in each directory
    glob_path = os.path.join(_BASE_DIR.value, _PATTERN.value, )
    paths = glob.glob(glob_path, recursive=True)

    # print(paths)
    # We will keep a mapping between metrics and their corresponding values
    metric_values = defaultdict(list)
    metric_units = {}

    for path in paths:
        print(f'Processing: {path}')
        with open(path) as f:
            json_data = json.load(f)
            # print(json_data)
            for configuration, data in json_data.items():
                for metric, metric_data in data.items():
                    # print(metric)
                    unit = metric_data.get('units', None)
                    values = metric_data.get('values', [])

                    # make sure values is a list
                    if not isinstance(values, list):
                        values = [values]

                    # flatten the list if the values are 2-dimensional
                    if values and isinstance(values[0], list):
                        values = [item for sublist in values for item in sublist]

                    metric_values[metric].extend(values)
                    metric_units[metric] = unit

    # If no values were found, terminate early
    if not metric_values:
        print("No values were found in any metrics_results.json files.")
        return

    for metric, values in metric_values.items():
        if values:
            values_array = np.array(values)
            non_nan_values = values_array[~np.isnan(values_array)]
            overall_mean = np.mean(non_nan_values)
            overall_std = np.std(non_nan_values)
            print(f'{metric}: {overall_mean:.2f} ± {overall_std:.2f} {metric_units[metric]}')


if __name__ == '__main__':
    app.run(main)
