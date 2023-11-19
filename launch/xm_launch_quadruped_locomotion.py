import os
import numpy as np
from xmanager import xm
from xmanager import xm_local

import itertools
from absl import app
from absl import flags

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name', 'quadruped_locomotion', 'Name of experiment'
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir', '/tmp/xm_local', 'Base directory for logs and results'
)
_TRAIN_LOGS_DIRS = flags.DEFINE_multi_string(
    'train_logs_dirs',
    ['train'],
    'Directory patterns fr train logs that will be used to calculate reliability metrics. Should be glob patterns',
)
_LOCAL = flags.DEFINE_bool('local', False, 'Run locally or on cluster')
_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode')
_TRAIN_EXP_ID = flags.DEFINE_string(
    'train_exp_id',
    None,
    'Experiment where the training logs are stored. This must be present for'
    ' inference or running offline metrics',
)
_INFERENCE = flags.DEFINE_bool(
    'inference', False, 'Whether to run train or inference.'
)
_RUN_OFFLINE_METRICS_ONLY = flags.DEFINE_bool(
    'run_offline_metrics_only', False, 'Whether to run train or inference.'
)
_PARTICIPANT_MODULE_PATH = flags.DEFINE_string(
    'participant_module_path', None, 'Path to participant module'
)
_GIN_CONFIG = flags.DEFINE_string(
    'gin_config',
    None,
    'Path to gin config file that determines which experiment to run',
)
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed')
_EXPERIMENT_NUMBER = flags.DEFINE_string('experiment_number', None, 'Experiment number')
_MOTION_FILE_PATH = flags.DEFINE_string('motion_file_path',
                                        '/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt',
                                        'Motion file')
FLAGS = flags.FLAGS


def main(_):
    # set directory of this script as working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # If experiment number is defined, replace the last part of root_dir with experiment number
    if _EXPERIMENT_NUMBER.value is not None:
        root_dir_flag = os.path.join(os.path.dirname(_ROOT_DIR.value), _EXPERIMENT_NUMBER.value)
    else:
        root_dir_flag = _ROOT_DIR.value
    quadruped_locomotion_dir = os.path.join(os.getcwd(), '../rl_perf/domains/quadruped_locomotion')
    if _LOCAL.value:
        executable_path = '/usr/bin/bash'
        binary_path = './local/quadruped_locomotion/launch.sh'
        additional_args = []
        env_vars = dict(
            quadruped_locomotion_DIR=quadruped_locomotion_dir,
            TF_FORCE_GPU_ALLOW_GROWTH='true',

        )

    else:
        # Create log dirs since singularity needs them to exist
        executable_path = '/usr/bin/sbatch'
        binary_path = './singularity/quadruped_locomotion/launch.slurm'
        additional_args = []
        env_vars = dict(
            TF_FORCE_GPU_ALLOW_GROWTH='true',
            # TF_GPU_ALLOCATOR='cuda_malloc_async' # doesn't work on some of the FASRC machines???
        )

    with xm_local.create_experiment(experiment_title=_EXPERIMENT_NAME.value) as experiment:
        if _DEBUG.value:
            quadruped_locomotion_seeds = [
                _SEED.value,
            ]
            num_parallel_cores = [170]
            total_env_steps = [500000, ]
            int_save_freqs = [100000]
            int_eval_freqs = [10000]
        else:
            quadruped_locomotion_seeds = [
                _SEED.value,
            ]
            total_env_steps = [200000000, ]
            num_parallel_cores = [170]
            int_save_freqs = [1000000]
            int_eval_freqs = [1000000]

        quadruped_locomotion_hparam_sweeps = list(
            dict([
                ('seed', seed),
                ('total_env_steps', env_steps),
                ('parallel_cores', parallel_cores),
                ('int_save_freq', int_save_freq),
                ('int_eval_freq', int_eval_freq),
            ])
            for seed, env_steps, parallel_cores, int_save_freq, int_eval_freq in
            itertools.product(quadruped_locomotion_seeds,
                              total_env_steps, num_parallel_cores,
                              int_save_freqs,
                              int_eval_freqs
                              )
        )

        # Define Executable
        [executable] = experiment.package([
            xm.binary(
                path=executable_path,
                args=[binary_path] + additional_args,
                executor_spec=xm_local.LocalSpec(),
                env_vars=env_vars,
            )
        ])

        for hparam_config in quadruped_locomotion_hparam_sweeps:
            experiment_name = _EXPERIMENT_NAME.value + '_' + '_'.join(
                f"{key}_{hparam_config[key]}" for key in sorted(hparam_config.keys()))

            root_dir = os.path.abspath(root_dir_flag)
            root_dir = os.path.join(root_dir, experiment_name)
            participant_module_path = os.path.join(_PARTICIPANT_MODULE_PATH.value)
            run_offline_metrics_only = str(_RUN_OFFLINE_METRICS_ONLY.value)

            # Add additional arguments that are constant across all runs
            hparam_config.update(dict(
                root_dir=root_dir,
                gin_config=_GIN_CONFIG.value,
                participant_module_path=participant_module_path,
                quad_loco_dir=quadruped_locomotion_dir,
                train_logs_dirs=','.join(_TRAIN_LOGS_DIRS.value),
                motion_file_path=_MOTION_FILE_PATH.value,
                run_offline_metrics_only=run_offline_metrics_only,
                mode='train',
            ))

            print(hparam_config)
            experiment.add(xm.Job(
                executable=executable,
                executor=xm_local.Local(),
                args=hparam_config,
                env_vars=env_vars,
            ))


if __name__ == '__main__':
    app.run(main)
