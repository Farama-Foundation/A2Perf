import os
import subprocess

from absl import app
from absl import flags
from absl import logging

_ROOT_DIR = flags.DEFINE_string(
    "root-dir", None, "Root directory to save metrics and logs from training"
)
_DATASETS_PATH = flags.DEFINE_string(
    "datasets-path",
    None,
    "Path to load the Minari datasets from. "
    "This is usually not needed since datasets will be downloaded from the internet",
)
_SUBMISSION_GIN_CONFIG_PATH = flags.DEFINE_string(
    "submission-gin-config-path",
    None,
    "Path to the gin configuration file for running the A2Perf submission.",
)
_PARTICIPANT_ARGS = flags.DEFINE_string(
    "participant-args",
    None,
    "Additional keyword arguments to pass to the participant's train and inference functions.",
)


def _usage():
    return (
        "Usage: a2perf <participant_module_path> --root-dir=<root_dir> --submission-gin-config-path=<path> [--participant-args=<k1=v1,k2=v2,...>]\n"
        "\n"
        "Options:\n"
        "\t--root-dir\tRoot directory for the experiment\n"
        "\t--submission-gin-config-path\tPath to the gin configuration file\n"
        "\t--participant-args\tAdditional arguments for the participant's train function"
    )


def main(argv):
    if len(argv) < 2:
        print(_usage())
        return 1

    participant_module_path = argv[1]

    if not os.path.exists(participant_module_path):
        print(f"Error: Participant module {participant_module_path} does not exist.")
        return 1

    try:
        app.parse_flags_with_usage(argv[1:])
    except flags.Error as e:
        print(f"Error: {e}\n")
        print(_usage())
        return 1

    # Check for required flags
    if _ROOT_DIR.value is None:
        print("Error: --root-dir is required.\n")
        print(_usage())
        return 1

    if _SUBMISSION_GIN_CONFIG_PATH.value is None:
        print("Error: --submission-gin-config-path is required.\n")
        print(_usage())
        return 1

    if not os.path.exists(_SUBMISSION_GIN_CONFIG_PATH.value):
        print(f"Error: {_SUBMISSION_GIN_CONFIG_PATH.value} does not exist.")
        return 1

    root_dir = os.path.expanduser(_ROOT_DIR.value)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        logging.info("Created root directory %s", root_dir)
    os.environ["ROOT_DIR"] = root_dir

    # os.environ["MINARI_DATASETS_PATH"] = _DATASETS_PATH.value
    os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_ON_CSV_WRITE"] = "append"
    # os.environ["CODECARBON_PUE"] = "1.0"

    command = [
        "python",
        "-m",
        "a2perf.submission.main_submission",
        f"--verbosity={logging.get_verbosity()}",
        f"--submission-gin-config-path={_SUBMISSION_GIN_CONFIG_PATH.value}",
        f"--root-dir={root_dir}",
        f"--metric-values-dir={os.path.join(root_dir, 'metrics')}",
        f"--participant-module-path={participant_module_path}",
    ]

    if _PARTICIPANT_ARGS.value:
        command.append(f"--participant-args={_PARTICIPANT_ARGS.value}")

    try:
        process = subprocess.Popen(command, env=os.environ.copy(), text=True)
        process.wait()

        if process.returncode != 0:
            print(f"Error: Command failed with return code {process.returncode}")
            return process.returncode
        else:
            print("Finished running the command successfully.")
            return 0
    except Exception as e:
        print(f"Error running the command: {e}")
        return 1


def run_main():
    app.run(main)


if __name__ == "__main__":
    run_main()
