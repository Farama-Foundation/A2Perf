import os
from xmanager import xm
from xmanager import xm_local

import itertools
from absl import app
from absl import flags

flags.DEFINE_string('experiment_name', 'circuit_training', 'Name of experiment')
flags.DEFINE_string('root_dir', '/tmp/xm_local',
                    'Base directory for logs and results')
flags.DEFINE_string('train_logs_dirs', 'train',
                    'Directory for train logs from all of the experiments that reliability metrics will be calculated on')
flags.DEFINE_bool('local', False, 'Run locally or on cluster')
flags.DEFINE_bool('debug', False, 'Debug mode')
flags.DEFINE_bool('run_offline_metrics_only', False,
                  'Whether to run offline metrics only.')
flags.DEFINE_string('participant_module_path', None,
                    'Path to participant module')
flags.DEFINE_string('gin_config', None,
                    'Path to gin config file that determines which experiment to run')
FLAGS = flags.FLAGS


def main(_):
  # set directory of this script as working directory
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  circuit_training_dir = os.path.join(os.getcwd(),
                                      '../rl_perf/domains/circuit_training')

  executable_path = '/usr/bin/bash'
  binary_path = './xgcp/circuit_training/launch.sh'
  additional_args = []
  env_vars = dict(
      CIRCUIT_TRAINING_DIR=circuit_training_dir,
      TF_FORCE_GPU_ALLOW_GROWTH='true',

  )
  repo_root = os.path.abspath(os.path.join(os.getcwd(), '../'))

  with xm_local.create_experiment(
      experiment_title=FLAGS.experiment_name) as experiment:
    if FLAGS.debug:
      num_collect_job_params = [4, ]
      netlist_file = os.path.join(repo_root,
                                  'rl_perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt'),
      init_placement = os.path.join(repo_root,
                                    'rl_perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/initial.plc'),
    else:

      num_collect_job_params = [4]
      netlist_file = os.path.join(repo_root,
                                  'rl_perf/domains/circuit_training/circuit_training/environment/test_data/ariane/netlist.pb.txt'),
      init_placement = os.path.join(repo_root,
                                    'rl_perf/domains/circuit_training/circuit_training/environment/test_data/ariane/initial.plc'),
    circuit_training_seeds = [
        37,
    ]
    circuit_training_hparam_sweeps = list(
        dict([
            ('seed', seed),
            ('num_collect_jobs', num_collect_jobs),
        ])
        for seed, num_collect_jobs in
        itertools.product(circuit_training_seeds, num_collect_job_params)
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

    for hparam_config in circuit_training_hparam_sweeps:
      experiment_name = FLAGS.experiment_name + '_' + '_'.join(
          f"{key}_{hparam_config[key]}" for key in sorted(hparam_config.keys()))

      root_dir = os.path.abspath(FLAGS.root_dir)
      root_dir = os.path.join(root_dir, experiment_name)
      train_logs_dirs = root_dir
      participant_module_path = os.path.join(FLAGS.participant_module_path)
      run_offline_metrics_only = str(FLAGS.run_offline_metrics_only)

      train_logs_dirs = os.path.join(train_logs_dirs,
                                     str(hparam_config['seed']), 'train',
                                     '0'),  # add more train logs dirs here if needed
      # Add additional arguments that are constant across all runs
      hparam_config.update(dict(
          root_dir=root_dir,
          gin_config=FLAGS.gin_config,
          participant_module_path=participant_module_path,
          circuit_training_dir=circuit_training_dir,
          train_logs_dirs=train_logs_dirs,
          run_offline_metrics_only=run_offline_metrics_only,
          reverb_port='8000',
          reverb_server_ip='ikechukwuu@reverb-eval-server',
          netlist_file=netlist_file,
          init_placement=init_placement
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
