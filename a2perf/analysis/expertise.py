import glob
import json
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_EXPERIMENT_IDS = flags.DEFINE_list(
    "experiment_ids", [], "List of experiment IDs to load the evaluation data."
)
_ROOT_DIR = flags.DEFINE_string(
    "root_dir", None, "Root directory to load the evaluation data."
)

_AVERAGE_MEASURE = flags.DEFINE_enum(
    "average_measure",
    None,
    ["mean", "median"],
    "Measure to use for averaging the episode rewards.",
)

_SKILL_LEVEL = flags.DEFINE_enum(
    "skill_level",
    None,
    ["novice", "intermediate", "expert"],
    "Skill level of the expert.",
)

_TASK_NAME = flags.DEFINE_string(
    "task_name",
    None,
    "Name of the task to perform. This is used to name the dataset.",
)


def assign_skill_level(row, col_name, bounds):
    """Assign skill level to a row based on episode_reward and predefined bounds."""
    reward = row[col_name]

    # Check against the bounds and assign the level
    if reward <= bounds["novice"][1]:
        return "novice"
    elif bounds["intermediate"][0] <= reward <= bounds["intermediate"][1]:
        return "intermediate"
    elif reward >= bounds["expert"][0]:
        return "expert"
    else:
        # If it is not in any bounds, we place a pd.NA and drop it later
        return pd.NA


def plot_skill_levels(data_df, col_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    data_df["skill_level"] = data_df["skill_level"].astype("category")

    sns.histplot(
        data=data_df,
        x=col_name,
        hue="skill_level",
        kde=False,
        stat="count",
        legend=True,
        linewidth=0,
        ax=ax,
    )

    ax.set_title("Episode Reward Distribution by Skill Level")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    fig.legend()
    fig.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)


def glob_path(path):
    return glob.glob(path, recursive=True)


def load_evaluation_json_data(base_dir, experiment_ids):
    with multiprocessing.Pool() as pool:
        json_files = pool.map(
            glob_path,
            [
                os.path.join(base_dir, f"{exp_id}/**/evaluation.json")
                for exp_id in experiment_ids
            ],
        )
        pool.close()
        pool.join()
    json_files_paths = [item for sublist in json_files for item in sublist]
    json_files_paths = set(json_files_paths)

    all_data = []
    for file_path in json_files_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            data_df = pd.DataFrame.from_dict(data, orient="index").reset_index()
            data_df = data_df.rename(columns={"index": "checkpoint_path"})
            all_data.append(data_df)
    all_data = pd.concat(all_data, ignore_index=True)
    return all_data


def main(_):
    root_dir = os.path.expanduser(_ROOT_DIR.value)
    evaluation_data_df = load_evaluation_json_data(
        base_dir=root_dir, experiment_ids=_EXPERIMENT_IDS.value
    )

    logging.info("Loaded evaluation data")

    logging.info("Using average measure: %s", _AVERAGE_MEASURE.value)
    average_average_return = evaluation_data_df[_AVERAGE_MEASURE.value].mean()
    logging.info("Average average return: %s", average_average_return)

    std_average_return = evaluation_data_df[_AVERAGE_MEASURE.value].std()
    logging.info("Standard deviation of average return: %s", std_average_return)

    novice_cutoff = (-np.inf, average_average_return - (2 * std_average_return))
    intermediate_cutoff = (
        average_average_return - std_average_return,
        average_average_return + std_average_return,
    )
    expert_cutoff = (average_average_return + (2 * std_average_return), np.inf)

    logging.info("Novice cutoff: %s", novice_cutoff)
    logging.info("Intermediate cutoff: %s", intermediate_cutoff)
    logging.info("Expert cutoff: %s", expert_cutoff)

    # Add a column to the evaluation_data_df for the skill level
    evaluation_data_df["skill_level"] = evaluation_data_df.apply(
        assign_skill_level,
        args=(
            _AVERAGE_MEASURE.value,
            {
                "novice": novice_cutoff,
                "intermediate": intermediate_cutoff,
                "expert": expert_cutoff,
            },
        ),
        axis=1,
    )

    # Drop rows with pd.NA since they do not fall into any skill level
    evaluation_data_df = evaluation_data_df.dropna(subset=["skill_level"])

    plot_skill_levels(
        evaluation_data_df,
        _AVERAGE_MEASURE.value,
        save_path=os.path.join(
            root_dir,
            _TASK_NAME.value,
            _SKILL_LEVEL.value,
            "skill_level_distribution.png",
        ),
    )

    # Save the data with skill levels so we can load to generate datasets
    evaluation_data_df.to_csv(
        os.path.join(
            root_dir,
            _TASK_NAME.value,
            _SKILL_LEVEL.value,
            "evaluation_data_with_skill_levels.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    app.run(main)
