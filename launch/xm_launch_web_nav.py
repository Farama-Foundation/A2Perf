import os
from xmanager import xm
from xmanager import xm_local

import itertools
from absl import app
from absl import flags

# create flag for experiment name
flags.DEFINE_string('experiment_name', 'experiment_name', 'Name of experiment')
flags.DEFINE_bool('local', True, 'Run locally or on cluster')

FLAGS = flags.FLAGS


def main(_):
    # set directory of this script as working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if FLAGS.local:
        executable_path = '/usr/bin/bash'
        binary_path = './local/web_nav/launch.sh'
        additional_args = []
    else:
        # Create log dirs since singularity needs them to exist

        executable_path = '/usr/bin/sbatch'
        binary_path = './singularity/web_nav/launch.slurm'
        additional_args = []

    with xm_local.create_experiment(experiment_title=FLAGS.experiment_name) as experiment:
        web_nav_seeds = [0, ]
        web_nav_hparam_sweeps = list(
                dict([
                        ('seed', seed),
                        ])
                for (seed,) in
                itertools.product(
                        web_nav_seeds,
                        )
                )

        # Define Executable
        [executable] = experiment.package([
                xm.binary(
                        path=executable_path,
                        args=[binary_path] + additional_args,
                        executor_spec=xm_local.LocalSpec()

                        )
                ])

        # Define resource requirements for the job
        for hparam_config in web_nav_hparam_sweeps:
            print(hparam_config)
            experiment.add(xm.Job(
                    executable=executable,
                    executor=xm_local.Local(),
                    args=dict(**hparam_config),
                    env_vars=dict(),
                    ))


if __name__ == '__main__':
    app.run(main)
