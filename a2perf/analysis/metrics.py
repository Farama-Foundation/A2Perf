import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from a2perf import analysis
from a2perf.analysis.metrics_lib import correct_energy_measurements
from a2perf.analysis.metrics_lib import load_generalization_metric_data
from a2perf.analysis.metrics_lib import load_inference_metric_data
from a2perf.analysis.metrics_lib import load_inference_system_data
from a2perf.analysis.metrics_lib import load_training_reward_data
from a2perf.analysis.metrics_lib import load_training_system_data

_SEED = flags.DEFINE_integer("seed", 0, "Random seed.")
_BASE_DIR = flags.DEFINE_string(
    "base-dir",
    "/home/ikechukwuu/workspace/rl-perf/logs",
    "Base directory for logs.",
)
_EXPERIMENT_IDS = flags.DEFINE_list(
    "experiment-ids", [94408569], "Experiment IDs to process."
)
NUM_COLLECT_JOB_TO_CPU_RATIO = dict(
    quadruped_locomotion=44 / 96,
    web_navigation=36 / 96,
    circuit_training=25 / 96,
)

DOMAIN_COLLECT_CPU_USAGE_FRACTION = {
    "circuit_training": 0.85,
    "quadruped_locomotion": 0.46,
    "web_navigation": 0.57,
}
TRAIN_SERVER_CPU_USAGE_FRACTION = 0.05


def _initialize_plotting():
    base_font_size = 25
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "font.size": base_font_size - 2,
            "axes.labelsize": base_font_size - 2,
            "axes.titlesize": base_font_size,
            "axes.labelweight": "bold",  # Bold font for the axes labels
            "legend.fontsize": base_font_size - 4,
            "xtick.labelsize": base_font_size - 4,
            "ytick.labelsize": base_font_size - 4,
            "figure.titlesize": base_font_size,
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "savefig.format": "png",
            "savefig.bbox": "tight",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,  # Lighter grid lines
        }
    )


def main(_):
    tf.compat.v1.enable_eager_execution()
    np.random.seed(_SEED.value)
    tf.random.set_seed(_SEED.value)
    _initialize_plotting()

    base_dir = os.path.expanduser(_BASE_DIR.value)
    training_reward_data_df = load_training_reward_data(
        base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
    )

    # plot_training_reward_data(training_reward_data_df,
    #                           event_file_tags=['Metrics/AverageReturn'])

    training_reward_metrics = analysis.reliability.get_training_metrics(
        data_df=training_reward_data_df, tag="Metrics/AverageReturn", index="Step"
    )
    training_system_metrics_df = load_training_system_data(
        base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
    )

    # DEBUG: See how many collect jobs there are. To do this, group by domain/task/algo/seed, and count the number of unique
    # experiment fields. This should be 1 + the number of collect jobs.
    total_jobs = training_system_metrics_df.groupby(
        ["domain", "task", "algo", "seed", "experiment"]
    ).run_id.nunique()
    num_collect_jobs = total_jobs - 1
    # Only keep groups with more than one run_id
    num_collect_jobs = num_collect_jobs[num_collect_jobs > 0]
    env_batch_size = 512
    domain = training_system_metrics_df["domain"].iloc[0]
    collect_cpu_ratio = NUM_COLLECT_JOB_TO_CPU_RATIO[domain]

    # Make sure the number of collect jobs matches the expected number
    expected_num_collect_jobs = np.ceil(env_batch_size / collect_cpu_ratio / 96)
    if all(num_collect_jobs == expected_num_collect_jobs):
        # 96 vCPU case
        cpus_per_collect_job = 96
        logging.info("Experiments were run on 96 vCPUs")
    else:
        # 32 vCPU case
        cpus_per_collect_job = 32
    cpus_per_train_Job = 48
    total_cpus_on_collect_machine = (
        128  # https://cloud.google.com/compute/docs/general-purpose-machines#n2_series
    )
    total_cpus_on_train_machine = (
        48  # https://cloud.google.com/compute/docs/gpus#a100-gpus
    )
    true_collect_cpu_tdp = (
        300  # https://www.cpu-world.com/CPUs/Xeon/Intel-Xeon%208373C.html
    )
    true_train_cpu_tdp = (
        165  # https://www.cpu-world.com/CPUs/Xeon/Intel-Xeon%208273CL.html
    )
    training_system_metrics_df = correct_energy_measurements(
        training_system_metrics_df,
        cpus_per_train_job=cpus_per_train_Job,
        total_cpus_on_train_machine=total_cpus_on_train_machine,
        percent_train_cpu_usage=TRAIN_SERVER_CPU_USAGE_FRACTION,
        cpus_per_collect_job=cpus_per_collect_job,
        total_cpus_on_collect_machine=total_cpus_on_collect_machine,
        percent_collect_cpu_usage=DOMAIN_COLLECT_CPU_USAGE_FRACTION[domain],
        true_train_cpu_tdp=true_train_cpu_tdp,
        true_collect_cpu_tdp=true_collect_cpu_tdp,
    )

    training_system_metrics = analysis.system.get_training_metrics(
        data_df=training_system_metrics_df
    )
    print(training_system_metrics)
    training_metrics = dict(**training_reward_metrics, **training_system_metrics)

    inference_reward_metrics, inference_reward_metrics_df = load_inference_metric_data(
        base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
    )
    inference_reward_metrics.update(
        analysis.reliability.get_inference_metrics(data_df=inference_reward_metrics_df)
    )
    inference_system_metrics_df = load_inference_system_data(
        base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
    )
    inference_system_metrics = analysis.system.get_inference_metrics(
        data_df=inference_system_metrics_df
    )
    inference_metrics = dict(**inference_reward_metrics, **inference_system_metrics)
    print(inference_metrics)

    generalization_reward_metrics = load_generalization_metric_data(
        base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
    )
    print(generalization_reward_metrics)

    # Take the rollout_returns from generalization_metrics and add it to training_metrics
    training_metrics["generalization_rollout_returns"] = generalization_reward_metrics[
        "generalization_rollout_returns"
    ]
    del generalization_reward_metrics["generalization_rollout_returns"]

    # Take rollout_returns from inference_metrics and add it to training_metrics
    training_metrics["rollout_returns"] = inference_metrics["rollout_returns"]
    del inference_metrics["rollout_returns"]

    training_metrics_df = analysis.results.metrics_dict_to_pandas_df(training_metrics)
    inference_metrics_df = analysis.results.metrics_dict_to_pandas_df(inference_metrics)

    print(analysis.results.df_as_latex(training_metrics_df, mode="train"))
    print(analysis.results.df_as_latex(inference_metrics_df, mode="inference"))


if __name__ == "__main__":
    app.run(main)
