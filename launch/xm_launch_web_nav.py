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
_DIFFICULTY_LEVEL = flags.DEFINE_integer('difficulty_level', 1, 'Difficulty level')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed')
FLAGS = flags.FLAGS


def main(_):
    # set directory of this script as working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    web_nav_dir = os.path.join(os.getcwd(), '../rl_perf/domains/web_nav')
    if FLAGS.local:
        executable_path = '/usr/bin/bash'
        binary_path = './local/web_nav/launch.sh'
        additional_args = []
        env_vars = dict(
            WEB_NAV_DIR=web_nav_dir,
            TF_FORCE_GPU_ALLOW_GROWTH='true',
            TF_GPU_ALLOCATOR='cuda_malloc_async'
        )

    else:
        # Create log dirs since singularity needs them to exist
        executable_path = '/usr/bin/sbatch'
        binary_path = './singularity/web_nav/launch.slurm'
        additional_args = []
        env_vars = dict(
            TF_FORCE_GPU_ALLOW_GROWTH='true',
            # TF_GPU_ALLOCATOR='cuda_malloc_async' # doesn't work on some of the FASRC machines???
        )

    with xm_local.create_experiment(experiment_title=FLAGS.experiment_name) as experiment:

        if FLAGS.debug:
            web_nav_seeds = [_SEED.value]
            env_batch_sizes = [8]
            total_env_steps = [50000]
            difficulty_levels = [_DIFFICULTY_LEVEL.value]
            learning_rates = [1e-4]  # Example values; adjust as needed.
            eval_intervals = [5000]
            train_checkpoint_intervals = [10000]
            policy_checkpoint_intervals = [10000]

        else:
            web_nav_seeds = [_SEED.value]
            env_batch_sizes = [16]
            total_env_steps = [1000000]
            difficulty_levels = [_DIFFICULTY_LEVEL.value]

            learning_rates = [1e-4]  # Example values; adjust as needed.
            eval_intervals = [50000]
            train_checkpoint_intervals = [100000]
            policy_checkpoint_intervals = [100000]

        # Combine all hyperparameters for the sweep
        web_nav_hparam_sweeps = [
            {
                'seed': seed,
                'env_batch_size': env_batch_size,
                'total_env_steps': env_steps,
                'difficulty_level': difficulty_level,
                'learning_rate': lr,
                'eval_interval': ei,
                'train_checkpoint_interval': tci,
                'policy_checkpoint_interval': pci,
            }
            for (seed, env_batch_size, env_steps, difficulty_level, lr, ei, tci, pci) in itertools.product(
                web_nav_seeds,
                env_batch_sizes,
                total_env_steps,
                difficulty_levels,
                learning_rates,
                eval_intervals,
                train_checkpoint_intervals,
                policy_checkpoint_intervals,
            )
        ]

        # Define Executable
        [executable] = experiment.package([
            xm.binary(
                path=executable_path,
                args=[binary_path] + additional_args,
                executor_spec=xm_local.LocalSpec(),
                env_vars=env_vars,
            )
        ])

        for hparam_config in web_nav_hparam_sweeps:
            experiment_name = FLAGS.experiment_name + '_' + '_'.join(
                f"{key}_{hparam_config[key]}" for key in sorted(hparam_config.keys()))

            # Add additional arguments that are constant across all runs
            root_dir = os.path.abspath(FLAGS.root_dir)
            root_dir = os.path.join(root_dir, experiment_name)
            train_logs_dirs = root_dir
            participant_module_path = os.path.join(FLAGS.participant_module_path)
            run_offline_metrics_only = str(FLAGS.run_offline_metrics_only)
            hparam_config.update(dict(root_dir=root_dir,
                                      gin_config=FLAGS.gin_config,
                                      participant_module_path=participant_module_path,
                                      web_nav_dir=web_nav_dir,
                                      train_logs_dirs=train_logs_dirs,
                                      run_offline_metrics_only=run_offline_metrics_only, ))

            print(hparam_config)
            experiment.add(xm.Job(
                executable=executable,
                executor=xm_local.Local(),
                args=hparam_config,
                env_vars=env_vars,
            ))


if __name__ == '__main__':
    app.run(main)
