import os
from xmanager import xm
from xmanager import xm_local

import itertools
from absl import app
from absl import flags

flags.DEFINE_string('experiment_name', 'quadruped_locomotion', 'Name of experiment')
flags.DEFINE_string('root_dir', '/tmp/xm_local', 'Base directory for logs and results')
flags.DEFINE_string('train_logs_dirs', 'train',
                    'Directory for train logs from all of the experiments that reliability metrics will be calculated on')
flags.DEFINE_bool('local', False, 'Run locally or on cluster')
flags.DEFINE_bool('debug', False, 'Debug mode')
flags.DEFINE_bool('run_offline_metrics_only', False, 'Whether to run offline metrics only.')
flags.DEFINE_string('participant_module_path', None, 'Path to participant module')
flags.DEFINE_string('gin_config', None, 'Path to gin config file that determines which experiment to run')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('experiment_number', None, 'Experiment number')
flags.DEFINE_string('motion_file_path',
                    '/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt',
                    'Motion file')
FLAGS = flags.FLAGS


def main(_):
    # set directory of this script as working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # If experiment number is defined, replace the last part of root_dir with experiment number
    if FLAGS.experiment_number is not None:
        FLAGS.root_dir = os.path.join(os.path.dirname(FLAGS.root_dir), FLAGS.experiment_number)

    quadruped_locomotion_dir = os.path.join(os.getcwd(), '../rl_perf/domains/quadruped_locomotion')
    if FLAGS.local:
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

    with xm_local.create_experiment(experiment_title=FLAGS.experiment_name) as experiment:

        if FLAGS.debug:
            quadruped_locomotion_seeds = [
                # 37,
                FLAGS.seed,
                # 14,
                # 65,
                # 23,
                # 98,
                # 51,
                # 19,
                # 77,
                # 43
            ]
            num_parallel_cores = [4]
            total_env_steps = [1000, ]
        else:
            quadruped_locomotion_seeds = [
                FLAGS.seed,
                # 82,
                # 14,
                # 65,
                # 23,
                # 98,
                # 51,
                # 19,
                # 77,
                # 43
            ]
            total_env_steps = [
                # 2000000,
                200000000,
            ]
            num_parallel_cores = [40]

        quadruped_locomotion_hparam_sweeps = list(
            dict([
                ('seed', seed),
                ('total_env_steps', env_steps),
                ('parallel_cores', parallel_cores),

            ])
            for seed, env_steps, parallel_cores, in itertools.product(quadruped_locomotion_seeds,
                                                                      total_env_steps, num_parallel_cores,
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
            experiment_name = FLAGS.experiment_name + '_' + '_'.join(
                f"{key}_{hparam_config[key]}" for key in sorted(hparam_config.keys()))

            root_dir = os.path.abspath(FLAGS.root_dir)
            root_dir = os.path.join(root_dir, experiment_name)
            train_logs_dirs = root_dir
            participant_module_path = os.path.join(FLAGS.participant_module_path)
            run_offline_metrics_only = str(FLAGS.run_offline_metrics_only)

            # Add additional arguments that are constant across all runs
            hparam_config.update(dict(
                root_dir=root_dir,
                gin_config=FLAGS.gin_config,
                participant_module_path=participant_module_path,
                quad_loco_dir=quadruped_locomotion_dir,
                train_logs_dirs=train_logs_dirs,
                motion_file_path=FLAGS.motion_file_path,
                run_offline_metrics_only=run_offline_metrics_only,
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
