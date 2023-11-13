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
        )

    with xm_local.create_experiment(experiment_title=FLAGS.experiment_name) as experiment:

        if FLAGS.debug:
            # Debug mode hyperparameters
            summary_intervals = [5000]
            log_intervals = [5000]
            rb_capacity_values = [10000, ]
            rb_checkpoint_intervals = [5000]  # Assuming a default value for debug mode
            batch_size_values = [32, ]
            timesteps_per_actorbatch_values = [8]
            web_nav_seeds = [_SEED.value]
            env_batch_sizes = [2]
            total_env_steps = [50000]
            difficulty_levels = [_DIFFICULTY_LEVEL.value]
            learning_rates = [1e-4]
            eval_intervals = [5000]
            train_checkpoint_intervals = [10000]
            policy_checkpoint_intervals = [10000]
        else:
            # Non-debug mode hyperparameters
            summary_intervals = [50000]  # Adjusted to match the non-debug scale
            log_intervals = [50000]  # Adjusted to match the non-debug scale
            rb_capacity_values = [100000, 200000]  # Hypothetical values for non-debug mode
            rb_checkpoint_intervals = [5000]  # Assuming a default value for debug mode
            batch_size_values = [64, 128, 256]  # Hypothetical larger batch sizes for non-debug mode
            timesteps_per_actorbatch_values = [20]  # Hypothetical value for non-debug mode
            web_nav_seeds = [_SEED.value]
            env_batch_sizes = [16]
            total_env_steps = [1000000]
            difficulty_levels = [_DIFFICULTY_LEVEL.value]
            learning_rates = [1e-4]
            eval_intervals = [50000]
            train_checkpoint_intervals = [100000]
            policy_checkpoint_intervals = [100000]
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
                'summary_interval': si,
                'log_interval': li,
                'rb_capacity': rb,
                'rb_checkpoint_interval': rci,  # Added rb_checkpoint_interval
                'batch_size': bs,
                'timesteps_per_actorbatch': tpab,
            }
            for seed, env_batch_size, env_steps, difficulty_level, lr, ei, tci, pci, si, li, rb, rci, bs, tpab in
            itertools.product(
                web_nav_seeds,
                env_batch_sizes,
                total_env_steps,
                difficulty_levels,
                learning_rates,
                eval_intervals,
                train_checkpoint_intervals,
                policy_checkpoint_intervals,
                summary_intervals,
                log_intervals,
                rb_capacity_values,
                rb_checkpoint_intervals,  # Added rb_checkpoint_intervals here
                batch_size_values,
                timesteps_per_actorbatch_values,
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

        for i, hparam_config in enumerate(web_nav_hparam_sweeps):
            # Add additional arguments that are constant across all runs
            root_dir = os.path.abspath(FLAGS.root_dir)
            root_dir = os.path.join(root_dir, f'web_nav_{i}')
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
