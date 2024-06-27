import os
import shutil

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

from a2perf.constants import BenchmarkDomain
from a2perf.launch.docker_utils import get_docker_instructions
from a2perf.launch.docker_utils import get_entrypoint
from typing import Any, Dict

_NUM_GPUS = flags.DEFINE_integer("num-gpus", 1, "Number of GPUs to use")
_CPU_BASE_IMAGE = flags.DEFINE_string(
    "cpu-base-image",
    "gcr.io/deeplearning-platform-release/base-cpu:latest",
    "Base image for CPU jobs",
)
_GPU_BASE_IMAGE = flags.DEFINE_string(
    "gpu-base-image",
    "gcr.io/deeplearning-platform-release/base-gpu:latest",
    "Base image for GPU jobs",
)
_DOMAIN = flags.DEFINE_enum(
    "domain",
    None,
    [
        BenchmarkDomain.QUADRUPED_LOCOMOTION.value,
        BenchmarkDomain.WEB_NAVIGATION.value,
        BenchmarkDomain.CIRCUIT_TRAINING.value,
    ],
    "Domain to run",
)
_USER_ID = flags.DEFINE_integer("user_id", 1000, "User ID")
_USER = flags.DEFINE_string("user", os.getlogin(), "User")
_EXPERIMENT_ID = flags.DEFINE_string("experiment-id", None, "Experiment number")
_EXPERIMENT_NAME = flags.DEFINE_string("experiment-name", None, "Experiment name")
_INTERACTIVE = flags.DEFINE_bool(
    "interactive", False, "Whether to run in interactive mode"
)
_SUBMISSION_GIN_CONFIG_PATH = flags.DEFINE_string(
    "submission-gin-config-path",
    None,
    "Path to the gin configuration file",
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
_ROOT_DIR = flags.DEFINE_string(
    "root-dir",
    None,
    "Root directory for the experiment",
)

GENERIC_GIN_CONFIG_NAME = "submission_config.gin"
DOCKER_EXPERIMENT_DIR = "/experiment_dir"
DOCKER_PARTICIPANT_DIR = "/participant_code"
DOCKER_DATASETS_PATH = "/workdir/datasets"


def main(_):
    """Main function to set up and run the experiment."""
    create_experiment = xm_local.create_experiment

    with create_experiment(experiment_title=_EXPERIMENT_NAME.value) as experiment:
        experiment_id = _EXPERIMENT_ID.value or experiment.experiment_id
        base_root_dir = os.path.join(
            os.path.expanduser(_ROOT_DIR.value),
            str(experiment_id),
            _EXPERIMENT_NAME.value,
        )

        async def make_job(work_unit: xm.WorkUnit, **hparams: Dict[str, Any]) -> None:
            work_unit_id = work_unit.work_unit_id
            full_root_dir = os.path.join(base_root_dir, str(work_unit_id))
            os.makedirs(full_root_dir, exist_ok=True)

            participant_module_path = _PARTICIPANT_MODULE_PATH.value

            docker_gin_config_path = os.path.join(
                full_root_dir, GENERIC_GIN_CONFIG_NAME
            )
            try:
                shutil.copy(_SUBMISSION_GIN_CONFIG_PATH.value, docker_gin_config_path)
            except IOError as e:
                raise IOError(f"Error copying gin config file: {e}")

            executor = xm_local.Local(
                requirements=xm.JobRequirements(
                    resources={xm.ResourceType.LOCAL_GPU: _NUM_GPUS.value},
                ),
                docker_options=xm_local.DockerOptions(
                    ports={},
                    volumes={
                        full_root_dir: DOCKER_EXPERIMENT_DIR,
                        participant_module_path: DOCKER_PARTICIPANT_DIR,
                    },
                    interactive=_INTERACTIVE.value,
                ),
                experimental_stream_output=True,
            )
            docker_instructions = get_docker_instructions(
                uid=_USER_ID.value, env_name=_DOMAIN.value, user=_USER.value
            )

            base_image = (
                _GPU_BASE_IMAGE.value if _NUM_GPUS.value > 0 else _CPU_BASE_IMAGE.value
            )

            [executable] = experiment.package(
                [
                    xm.python_container(
                        executor_spec=executor.Spec(),
                        path=".",
                        use_deep_module=True,
                        base_image=base_image,
                        docker_instructions=docker_instructions,
                        entrypoint=get_entrypoint(_DOMAIN.value, _USER.value),
                    )
                ]
            )

            hparams.update(
                {
                    "root-dir": DOCKER_EXPERIMENT_DIR,
                    "gin-config": os.path.join(
                        DOCKER_EXPERIMENT_DIR, GENERIC_GIN_CONFIG_NAME
                    ),
                    "participant-module-path": DOCKER_PARTICIPANT_DIR,
                    "participant-args": _PARTICIPANT_ARGS.value,
                }
            )

            job = xm.Job(executable, args=hparams, executor=executor)
            work_unit.add(job)

        hparams = {"datasets-path": DOCKER_DATASETS_PATH}

        experiment.add(make_job, args=hparams)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            _DOMAIN.name,
            _EXPERIMENT_NAME.name,
            _ROOT_DIR.name,
            _SUBMISSION_GIN_CONFIG_PATH.name,
        ]
    )
    app.run(main)
