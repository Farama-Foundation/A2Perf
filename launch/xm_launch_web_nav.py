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
    else:
        executable_path = '/usr/bin/sbatch'
        binary_path = 'sbatch ./singularity/web_nav/launch.sh'

    with xm_local.create_experiment(experiment_title=FLAGS.experiment_name) as experiment:
        web_nav_seeds = [0, ]
        web_nav_hparam_sweeps = list(
                dict([
                        ('seed', seed),
                        ])
                for (
                        seed,
                        ) in
                itertools.product(
                        web_nav_seeds,
                        )
                )

        # Define Executable
        [executable] = experiment.package([
                xm.binary(
                        path=executable_path,
                        args=[binary_path],
                        executor_spec=xm_local.LocalSpec()

                        )
                ])

        # Define resource requirements for the job
        requirements = xm.JobRequirements(resources={xm.ResourceType.CPU: 1,
                                                     xm.ResourceType.MEMORY: 1,  # in bytes
                                                     xm.ResourceType.P100: 1,
                                                     })
        for hparam_config in web_nav_hparam_sweeps:
            experiment.add(xm.Job(
                    executable=executable,
                    executor=xm_local.Local(
                            requirements=requirements,
                            ),
                    args=dict(**hparam_config),
                    env_vars=dict(),
                    ))


if __name__ == '__main__':
    app.run(main)
