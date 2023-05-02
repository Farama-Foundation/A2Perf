import os
from xmanager import xm
from xmanager import xm_local

import itertools
from absl import app
from absl import flags

flags.DEFINE_string('experiment_name', 'web_nav', 'Name of experiment')
flags.DEFINE_string('root_dir', '/tmp/xm_local', 'Base directory for logs and results')
flags.DEFINE_bool('local', False, 'Run locally or on cluster')
flags.DEFINE_bool('train', False, 'Whether to run training or inference')
flags.DEFINE_string('participant_module_path', None, 'Path to participant module')
FLAGS = flags.FLAGS


def main(_):
    # set directory of this script as working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if FLAGS.local:
        executable_path = '/usr/bin/bash'
        binary_path = './local/web_nav/launch.sh'
        additional_args = []
        env_vars = dict(
            WEB_NAV_DIR=os.path.join(os.getcwd(), '../rl_perf/domains/web_nav'),
            TF_FORCE_GPU_ALLOW_GROWTH='true',

        )

    else:
        # Create log dirs since singularity needs them to exist
        executable_path = '/usr/bin/sbatch'
        binary_path = './singularity/web_nav/launch.slurm'
        additional_args = []
        env_vars = dict(
            TF_FORCE_GPU_ALLOW_GROWTH='true',
            TF_GPU_ALLOCATOR='cuda_malloc_async'
        )

    mode = 'train' if FLAGS.train else 'inference'

    with xm_local.create_experiment(experiment_title=FLAGS.experiment_name) as experiment:
        web_nav_seeds = [
            # 37,
            #              82, 14, 65, 23,
            98,
            # 51, 19, 77, 43
        ]
        env_batch_sizes = [4, ]
        web_nav_hparam_sweeps = list(
            dict([
                ('seed', seed),
                ('env_batch_size', env_batch_size),
            ])
            for (seed, env_batch_size) in itertools.product(
                web_nav_seeds,
                env_batch_sizes,
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

        # Get the full path of our FLAGS.root_dir since it is relative to this script
        root_dir = os.path.abspath(FLAGS.root_dir)

        for hparam_config in web_nav_hparam_sweeps:
            experiment_name = f"{FLAGS.experiment_name}_seed_{hparam_config['seed']}_env_batch_size_{hparam_config['env_batch_size']}"

            # Add additional arguments that are constant across all runs
            root_dir = os.path.join(root_dir, experiment_name)
            participant_module_path = os.path.join(FLAGS.participant_module_path, 'web_nav', f'{mode}.py')

            gin_file = os.path.join(f'configs/web_nav_{mode}.gin')
            hparam_config.update(dict(root_dir=root_dir, gin_config=gin_file,
                                      participant_module_path=participant_module_path))

            print(hparam_config)
            experiment.add(xm.Job(
                executable=executable,
                executor=xm_local.Local(),
                args=hparam_config,
                env_vars=env_vars,
            ))


if __name__ == '__main__':
    app.run(main)
