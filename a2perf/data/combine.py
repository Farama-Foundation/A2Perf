import multiprocessing
import os
import subprocess
import time

import pandas as pd
from absl import app, flags, logging

from a2perf.data.minari_dataset.data_utils import (
    combine_minari_datasets,
    delete_dataset,
    load_dataset,
)

_ROOT_DIR = flags.DEFINE_string(
    "root_dir",
    None,
    "Root directory of the environment. If not set, the ROOT_DIR environment "
    "variable is used.",
)
_NUM_EPISODES = flags.DEFINE_integer(
    "num_episodes", None, "Number of episodes to evaluate the policy."
)
_NUM_PROCESSES = flags.DEFINE_integer(
    "num_processes", None, "Number of processes to use."
)
_REPLICA_ID = flags.DEFINE_integer(
    "replica_id",
    0,
    "Replica ID of the current process. This is used to distribute the "
    "evaluation across multiple machines.",
)
_TASK_NAME = flags.DEFINE_string(
    "task_name",
    "evaluation",
    "Name of the task to perform. This is used to name the dataset.",
)
_NUM_MACHINES = flags.DEFINE_integer(
    "num_machines",
    1,
    "Number of machines used to generate the dataset. This is used to "
    "distribute the dataset generation across multiple machines.",
)
_SEED = flags.DEFINE_integer("seed", None, "Seed to use.")
_DATASETS_PATH = flags.DEFINE_string(
    "datasets_path",
    "/mnt/gcs/a2perf/datasets/quadruped_locomotion",
    "Path to save the dataset to.",
)
_AUTHOR = flags.DEFINE_string("author", "Ikechukwu Uchendu", "Author name.")
_AUTHOR_EMAIL = flags.DEFINE_string(
    "author_email", "iuchendu@g.harvard.edu", "Author email."
)
_CODE_PERMALINK = flags.DEFINE_string("code_permalink", "", "Code permalink.")
_SKILL_LEVEL = flags.DEFINE_enum(
    "skill_level",
    "novice",
    ["novice", "intermediate", "expert"],
    "Skill level of the expert.",
)

_ENV_NAME = flags.DEFINE_string("env_name", None, "Name of the environment.")
_POLICY_NAME = flags.DEFINE_string("policy_name", None, "Name of the policy.")


def main(_):
    env_name = _ENV_NAME.value[:-3]
    root_dir = os.path.expanduser(_ROOT_DIR.value)
    if _DATASETS_PATH.value is not None:
        base_path = _DATASETS_PATH.value
    else:
        base_path = _ROOT_DIR.value

    # For policies of the target skill level, get the average cost
    evaluation_data_path = os.path.join(
        root_dir,
        _TASK_NAME.value,
        _SKILL_LEVEL.value,
        "evaluation_data_with_skill_levels.csv",
    )
    evaluation_df = pd.read_csv(evaluation_data_path)
    evaluation_df = evaluation_df[evaluation_df.skill_level == _SKILL_LEVEL.value]
    if evaluation_df.empty:
        logging.warning("No policies found for skill level: %s", _SKILL_LEVEL.value)
        return

    average_energy_kwh = evaluation_df["training_energy_kwh"].mean()
    logging.info(
        "Average energy cost of policies at skill level: %s", average_energy_kwh
    )
    replica_dataset_paths = []
    replica_dataset_ids = []

    for i in range(_NUM_MACHINES.value):
        replica_dataset_id = (
            f"{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-{i:03d}-v0"
        )
        replica_dataset_paths.append(
            os.path.join(
                os.path.expanduser(base_path),
                _TASK_NAME.value,
                _SKILL_LEVEL.value,
                f"{i:03d}",
            )
        )
        replica_dataset_ids.append(replica_dataset_id)

    for path in replica_dataset_paths:
        while not os.path.exists(path):
            logging.info("Waiting for dataset %s to be available.", path)
            time.sleep(10)
    time.sleep(30)
    logging.info("All datasets are now available for combination.")

    # To use the replica datasets, we'll have to copy them to our local machine
    local_replica_dataset_paths = [
        os.path.join(
            "/tmp", _TASK_NAME.value, _SKILL_LEVEL.value, "replicas", f"{i:03d}"
        )
        for i in range(_NUM_MACHINES.value)
    ]
    logging.info("Copying replica datasets to local machine...")
    for i, (replica_dataset_path, local_replica_dataset_path) in enumerate(
        zip(replica_dataset_paths, local_replica_dataset_paths)
    ):
        os.makedirs(local_replica_dataset_path, exist_ok=True)
        subprocess.run(
            [
                "cp",
                "-r",
                os.path.join(replica_dataset_path, replica_dataset_ids[i]),
                os.path.join(local_replica_dataset_path, replica_dataset_ids[i]),
            ],
            check=True,
        )
        logging.info("Finished copying replica dataset %s.", i)

    # Wait after copying the datasets
    time.sleep(10)

    # Multiprocessing is unreliable with h5, so sequentially load
    replica_datasets = []
    for local_replica_dataset_path, replica_dataset_id in zip(
        local_replica_dataset_paths, replica_dataset_ids
    ):
        replica_datasets.append(
            load_dataset(local_replica_dataset_path, replica_dataset_id)
        )
    logging.info("Finished loading all datasets.")

    logging.info("Combining datasets...")
    final_dataset_path = os.path.join(
        os.path.expanduser(base_path),
        _TASK_NAME.value,
        _SKILL_LEVEL.value,
    )
    final_dataset_id = f"{env_name}-{_TASK_NAME.value}-{_SKILL_LEVEL.value}-v0"

    replica_datasets_combine_path = os.path.join(
        "/tmp", _TASK_NAME.value, _SKILL_LEVEL.value, "replicas"
    )
    # Use a tmp path for combining the replica datasets
    os.environ["MINARI_DATASETS_PATH"] = replica_datasets_combine_path

    # Multiprocessing version
    datasets_to_combine = replica_datasets[:]
    all_combined_dataset_ids = []
    j = 0
    while len(datasets_to_combine) > 2:
        dataset_a_list = datasets_to_combine[::2]
        dataset_b_list = datasets_to_combine[1::2]
        combined_dataset_ids = []
        for i in range(len(dataset_a_list)):
            comb_dataset_id = (
                f"{env_name}-{_TASK_NAME.value}-"
                f"{_SKILL_LEVEL.value}-final-merge-{j:03d}-v0"
            )
            combined_dataset_ids.append(comb_dataset_id)
            all_combined_dataset_ids.append(comb_dataset_id)
            j += 1
        tasks = zip(dataset_a_list, dataset_b_list, combined_dataset_ids)
        with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
            datasets_to_combine = pool.starmap(combine_minari_datasets, tasks)
            pool.close()
            pool.join()
            logging.info("Combined datasets %s.", combined_dataset_ids)

    # Combine the final two datasets with the proper name
    os.environ["MINARI_DATASETS_PATH"] = final_dataset_path
    full_dataset = combine_minari_datasets(
        datasets_to_combine[0],
        datasets_to_combine[1],
        final_dataset_id,
    )

    logging.info("Successfully combined datasets from each replica.")
    logging.info("\tTotal steps: %s", full_dataset.total_steps)
    logging.info("\tTotal episodes: %s", full_dataset.total_episodes)

    logging.info("Cleaning up temporary datasets.")
    with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
        tasks = zip(replica_dataset_paths, replica_dataset_ids)
        pool.starmap(delete_dataset, tasks)
        pool.close()
        pool.join()
    logging.info("Finished cleaning up temporary replica datasets.")

    logging.info("Cleaning up combined datasets...")
    for combined_dataset_id in all_combined_dataset_ids:
        subprocess.run(
            [
                "rm",
                "-r",
                os.path.join(replica_datasets_combine_path, combined_dataset_id),
            ],
            check=True,
        )
    logging.info("Finished cleaning up combined datasets.")

    logging.info("Cleaning up replica datasets on network drive...")
    for replica_dataset_path in replica_dataset_paths:
        subprocess.run(["rm", "-r", replica_dataset_path], check=True)
    logging.info("Finished cleaning up replica datasets on network drive.")

    # Leader should save the training sample cost
    with open(
        os.path.join(
            os.path.expanduser(base_path),
            _TASK_NAME.value,
            _SKILL_LEVEL.value,
            "training_sample_cost.txt",
        ),
        "w",
    ) as f:
        f.write(str(average_energy_kwh))


if __name__ == "__main__":
    app.run(main)
