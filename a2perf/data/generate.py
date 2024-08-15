import itertools
import multiprocessing
import os
import subprocess
import time

import minari
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags, logging

from a2perf.data.minari_dataset.data_utils import collect_dataset, delete_dataset

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
_NUM_TASKS = flags.DEFINE_integer(
    "num_tasks",
    1,
    "Number of tasks to perform. This is used to distribute the dataset "
    "generation across multiple tasks.",
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
    multiprocessing.set_start_method("spawn", force=True)
    np.random.seed(_SEED.value)
    tf.random.set_seed(_SEED.value)

    job_completion_index = _REPLICA_ID.value
    logging.info("Job completion index: %s", job_completion_index)

    if _DATASETS_PATH.value is not None:
        base_path = _DATASETS_PATH.value
    else:
        base_path = _ROOT_DIR.value

    minari_datasets_path = os.path.join(
        os.path.expanduser(base_path),
        _TASK_NAME.value,
        _SKILL_LEVEL.value,
        f"{job_completion_index:03d}",
    )

    tmp_minari_datasets_path = os.path.join(
        "/tmp",
        _TASK_NAME.value,
        _SKILL_LEVEL.value,
        f"{job_completion_index:03d}",
    )
    root_dir = os.path.expanduser(_ROOT_DIR.value)

    os.environ["MINARI_DATASETS_PATH"] = tmp_minari_datasets_path
    logging.info("Set MINARI_DATASETS_PATH to %s", minari_datasets_path)

    evaluation_data_path = os.path.join(
        root_dir,
        _TASK_NAME.value,
        _SKILL_LEVEL.value,
        "evaluation_data_with_skill_levels.csv",
    )

    while True:
        if (
            os.path.exists(evaluation_data_path)
            and os.path.getsize(evaluation_data_path) > 0
        ):
            try:
                evaluation_df = pd.read_csv(evaluation_data_path)
                break
            except pd.errors.EmptyDataError:
                logging.warning("Evaluation File is empty, waiting to retry...")
        else:
            logging.info(
                "Waiting for evaluation data to be available at %s",
                evaluation_data_path,
            )

        time.sleep(10)

    evaluation_df = evaluation_df[evaluation_df.skill_level == _SKILL_LEVEL.value]
    if evaluation_df.empty:
        logging.warning("No policies found for skill level: %s", _SKILL_LEVEL.value)
        return

    logging.info(
        "After filtering by skill level, %s policies found.", len(evaluation_df)
    )

    num_episodes_to_generate = _NUM_EPISODES.value // _NUM_MACHINES.value
    remainder = _NUM_EPISODES.value % _NUM_MACHINES.value
    if job_completion_index < remainder:
        num_episodes_to_generate += 1

    evaluation_df = evaluation_df.sample(
        random_state=_SEED.value, n=_NUM_TASKS.value, replace=True
    )

    if num_episodes_to_generate == 0 or evaluation_df.empty:
        logging.warning("No episodes to generate.")
        return

    episodes_per_checkpoint = num_episodes_to_generate // len(evaluation_df)
    logging.info("Episodes per checkpoint: %s", episodes_per_checkpoint)
    remainder = num_episodes_to_generate % len(evaluation_df)
    logging.info(
        "After distributing episodes, %s episodes to generate.",
        num_episodes_to_generate,
    )
    num_episodes_list = [
        episodes_per_checkpoint + 1 if i < remainder else episodes_per_checkpoint
        for i in range(len(evaluation_df))
    ]

    dataset_paths = []
    dataset_ids = []
    env_name = _ENV_NAME.value[:-3]
    for i in range(_NUM_TASKS.value):
        tmp_dataset_id = (
            f"{env_name}-{_TASK_NAME.value}-"
            f"{_SKILL_LEVEL.value}-{job_completion_index:03d}-{i:03d}-v0"
        )
        dataset_paths.append(tmp_minari_datasets_path)
        dataset_ids.append(tmp_dataset_id)

    with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
        tasks = zip(
            itertools.cycle([_ENV_NAME.value]),
            itertools.cycle([root_dir]),
            evaluation_df.checkpoint_path,
            dataset_paths,
            dataset_ids,
            num_episodes_list,
            itertools.cycle([_SEED.value]),
            itertools.cycle([_POLICY_NAME.value]),
        )
        datasets = pool.starmap(collect_dataset, tasks)
        pool.close()
        pool.join()
    logging.info("Finished collecting all episodes.")

    logging.info("Combining local datasets...")
    dataset_id = (
        f"{env_name}-{_TASK_NAME.value}-"
        f"{_SKILL_LEVEL.value}-{job_completion_index:03d}-v0"
    )
    dataset = minari.combine_datasets(
        datasets_to_combine=datasets, new_dataset_id=dataset_id
    )

    logging.info("Successfully combined datasets")
    logging.info("\tTotal steps: %s", dataset.total_steps)
    logging.info("\tTotal episodes: %s", dataset.total_episodes)

    logging.info("Moving dataset to final location %s", minari_datasets_path)
    os.makedirs(minari_datasets_path, exist_ok=True)
    subprocess.run(
        [
            "cp",
            "-r",
            os.path.join(tmp_minari_datasets_path, dataset_id),
            os.path.join(minari_datasets_path, dataset_id),
        ],
        check=True,
    )
    logging.info("Successfully moved dataset to %s ", minari_datasets_path)

    logging.info("Cleaning up temporary datasets.")
    with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
        tasks = zip(dataset_paths, dataset_ids)
        pool.starmap(delete_dataset, tasks)
        pool.close()
        pool.join()
    logging.info("Finished cleaning up temporary datasets.")


if __name__ == "__main__":
    app.run(main)
