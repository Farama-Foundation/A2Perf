import os
import subprocess

from absl import app, flags, logging

_ROOT_DIR = flags.DEFINE_string("root-dir", None, "Root directory.")
_DATASETS_PATH = flags.DEFINE_string(
    "datasets-path", None, "Path to save the dataset to."
)
_SUBMISSION_GIN_CONFIG_PATH = flags.DEFINE_string(
    "submission-gin-config-path", None, "Path to the gin configuration file."
)
_PARTICIPANT_MODULE_PATH = flags.DEFINE_string(
    "participant-module-path",
    None,
    "Path to the participant training and inference Python modules",
)
_PARTICIPANT_ARGS = flags.DEFINE_string(
    "participant-args",
    None,
    "Additional arguments to pass to the participant's train function",
)


def main(_):
    # os.environ["MINARI_DATASETS_PATH"] = _DATASETS_PATH.value
    os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_ON_CSV_WRITE"] = "append"
    # os.environ["CODECARBON_PUE"] = "1.0"

    root_dir = os.path.expanduser(_ROOT_DIR.value)
    os.environ["ROOT_DIR"] = root_dir
    command = [
        "python",
        "-m",
        "a2perf.submission.main_submission",
        f"--verbosity={logging.get_verbosity()}",
        f"--submission-gin-config-path={_SUBMISSION_GIN_CONFIG_PATH.value}",
        f"--root-dir={root_dir}",
        f"--metric-values-dir={os.path.join(root_dir, 'metrics')}",
        f"--participant-module-path={_PARTICIPANT_MODULE_PATH.value}",
        f"--participant-args={_PARTICIPANT_ARGS.value}",
    ]

    process = subprocess.Popen(command, env=os.environ.copy(), text=True)

    process.wait()
    if process.returncode != 0:
        raise ValueError(f"Error running the command: {command}")
    else:
        print("Finished running the command successfully.")


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            _SUBMISSION_GIN_CONFIG_PATH.name,
            _ROOT_DIR.name,
        ],
    )
    app.run(main)
