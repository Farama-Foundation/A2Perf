import multiprocessing
import os

import gin
from absl import app
from absl import flags
from absl import logging

from a2perf.constants import BenchmarkMode
from a2perf.submission import submission_util

_GIN_CONFIG = flags.DEFINE_string("gin-config", None, "Path to the gin-config file.")
_PARTICIPANT_MODULE_PATH = flags.DEFINE_string(
    "participant-module-path", None, "Path to participant module."
)
_ROOT_DIR = flags.DEFINE_string(
    "root-dir", "/tmp/xm_local", "Base directory for logs and results."
)
_METRIC_VALUES_DIR = flags.DEFINE_string(
    "metric-values-dir", None, "Directory to save metrics values."
)
_EXTRA_GIN_BINDINGS = flags.DEFINE_multi_string(
    "extra-gin-bindings",
    [],
    "Extra gin bindings to add configurations on the fly.",
)
_RUN_OFFLINE_METRICS_ONLY = flags.DEFINE_bool(
    "run-offline-metrics-only", False, "Whether to run offline metrics only."
)
_MODE = flags.DEFINE_enum(
    "mode",
    "train",
    ["train", "inference", "generalization"],
    "Mode of the submission. train, inference, or generalization.",
)


def main(_):
    # Set the working directory to the submission directory.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    multiprocessing.set_start_method("spawn", force=False)

    logging.info("Gin config for the submission: %s", _GIN_CONFIG.value)
    logging.info("Participant module path: %s", _PARTICIPANT_MODULE_PATH.value)

    gin.parse_config_file(_GIN_CONFIG.value)
    for binding in _EXTRA_GIN_BINDINGS.value:
        gin.parse_config(binding)
        logging.info("Adding extra gin binding: %s", binding)

    submission = submission_util.Submission(
        mode=BenchmarkMode(_MODE.value),
        root_dir=_ROOT_DIR.value,
        metric_values_dir=_METRIC_VALUES_DIR.value,
        participant_module_path=_PARTICIPANT_MODULE_PATH.value,
        run_offline_metrics_only=_RUN_OFFLINE_METRICS_ONLY.value,
    )

    submission.run_benchmark()

    # multiprocessing make sure all processes are terminated
    for p in multiprocessing.active_children():
        p.terminate()
        p.join()


if __name__ == "__main__":
    app.run(main)
