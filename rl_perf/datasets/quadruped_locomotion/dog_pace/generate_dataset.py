import multiprocessing
import os
import shutil

import gymnasium as gym
import minari
import numpy as np
from absl import app
from absl import flags
from absl import logging
from minari import DataCollectorV0

from rl_perf.domains.quadruped_locomotion.motion_imitation.learning import \
  ppo_imitation
from rl_perf.rlperf_benchmark_submission.quadruped_locomotion.ddpg import \
  ddpg_imitation

FLAGS = flags.FLAGS

_NUM_SAMPLES = flags.DEFINE_integer('num_samples', 100,
                                    'Number of samples to generate.')
_NUM_PROCESSES = flags.DEFINE_integer('num_processes', 1,
                                      'Number of processes to use.')
_SKILL_LEVEL = flags.DEFINE_enum('skill_level', 'novice',
                                 ['novice', 'intermediate', 'expert'],
                                 'Skill level of the expert.')
_DATASET_ID = flags.DEFINE_string('dataset_id', 'QuadrupedLocomotion',
                                  'Dataset ID.')
_SEED = flags.DEFINE_integer('seed', 0, 'Seed to use.')
_DATASETS_PATH = flags.DEFINE_string('datasets_path',
                                     '/mnt/gcs/a2perf/datasets/quadruped_locomotion',
                                     'Path to save the dataset to.')
_AUTHOR = flags.DEFINE_string('author', 'Ikechukwu Uchendu', 'Author name.')
_AUTHOR_EMAIL = flags.DEFINE_string('author_email', 'iuchendu@g.harvard.edu',
                                    'Author email.')
_CODE_PERMALINK = flags.DEFINE_string('code_permalink', '', 'Code permalink.')
_TASK = flags.DEFINE_enum('task', 'dog_pace',
                          ['dog_pace', 'dog_trot', 'dog_spin'],
                          'Task to run.')
SKILL_LEVEL_DICT = {'novice': [
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_4250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_4250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_4250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_4250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_20092500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_3013875_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_1004625_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_1004625_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_15069375_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_3013875_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_14064750_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_11050875_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_60277500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_12055500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_6027750_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_12055500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_11050875_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_14064750_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_10046250_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_12055500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_9041625_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_4018500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_17078625_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_9041625_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_6027750_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_9041625_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_4018500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_7032375_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_11050875_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_13060125_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_12055500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_6027750_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_4018500_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_8037000_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_16074000_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_17078625_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_14064750_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_14064750_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_3013875_steps.zip',
    '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_28129500_steps.zip'],
    'intermediate': [
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_177781750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_69916750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_32963000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_150815500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_86895500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_72913000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_49941750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_136833000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_70915500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_130840500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_85896750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_87894250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_182775500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_134835500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_108868000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_25971750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_93886750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_100878000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_84898000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_19979250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_47944250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_38955500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_29966750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_176783000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_127844250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_29966750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_70915500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_185771750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_51939250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_81901750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_61926750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_33152625_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_64296000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_105485625_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_51235875_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_176814000_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_87402375_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_55254375_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_39180375_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_117541125_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_61282125_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_172795500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_50231250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_91420875_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_100462500_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_130601250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_106490250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_50231250_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_142656750_steps.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_36166500_steps.zip'],
    'expert': [
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/final_ppo_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ppo_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ppo_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/final_ppo_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ppo_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/final_ppo_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ppo_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/final_ddpg_policy.zip',
        '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/final_ddpg_policy.zip']}


def load_model(policy_path, env):
  if 'ppo' in policy_path:
    model = ppo_imitation.PPOImitation.load(policy_path, env=env)
  elif 'ddpg' in policy_path:
    model = ddpg_imitation.DDPGImitation.load(policy_path, env=env)
  else:
    raise ValueError('Unknown policy path: {}'.format(policy_path))
  return model


def infer_once(model, observation):
  action, _states = model.predict(observation)
  return action


def preprocess_observation(observation):
  return observation


def set_seed(seed):
  np.random.seed(seed)


def collect_samples(policy_path, unique_id, samples_to_collect, seed):
  if _DATASETS_PATH.value is not None:
    os.environ['MINARI_DATASETS_PATH'] = _DATASETS_PATH.value

  # Need to set random seed again since we are in a new process.
  set_seed(seed)
  env = gym.make('QuadrupedLocomotion-v0', mode='test', motion_files=[
      '/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt'],
                 enable_rendering=False)
  env = DataCollectorV0(env=env, observation_space=gym.spaces.Box(low=-np.inf,
                                                                  high=np.inf,
                                                                  shape=(
                                                                      env.observation_space.shape[
                                                                        0],),
                                                                  dtype=np.float64),
                        action_space=env.action_space, max_buffer_steps=None,
                        record_infos=True, )
  # env.seed(seed)
  model = load_model(policy_path, env)
  logging.info(f'Loaded model from {policy_path}')
  algorithm_name = 'ppo' if 'ppo' in policy_path else 'ddpg'
  dataset_id = f'{_DATASET_ID.value}-{_TASK.value}-{_SKILL_LEVEL.value}-{unique_id}-v0'

  logging.info(f'Collecting samples for {dataset_id}.')

  while samples_to_collect > 0:
    logging.info(f'{samples_to_collect} samples left to collect.')
    observation, info = env.reset(seed=seed)
    observation = preprocess_observation(observation)

    terminated, truncated = False, False
    episode_reward = 0
    while not terminated and not truncated:
      action = infer_once(model, observation)
      observation, reward, terminated, truncated, info = env.step(action)
      episode_reward += reward
      observation = preprocess_observation(observation)
      samples_to_collect -= 1
    logging.info(f'Episode reward: {episode_reward}')

  logging.info(f'Finished collecting samples for {dataset_id}.')
  minari.create_dataset_from_collector_env(
      dataset_id=dataset_id,
      collector_env=env,
      algorithm_name=algorithm_name,
      author=_AUTHOR.value,
      author_email=_AUTHOR_EMAIL.value,
      code_permalink=_CODE_PERMALINK.value,
      ref_max_score=None,
      ref_min_score=None,
      expert_policy=None,
  )


def main(_):
  if _DATASETS_PATH.value is not None:
    os.environ['MINARI_DATASETS_PATH'] = _DATASETS_PATH.value

  set_seed(_SEED.value)
  policies_to_use = SKILL_LEVEL_DICT[_SKILL_LEVEL.value]

  policies_to_use = np.array(policies_to_use)
  np.random.shuffle(policies_to_use)
  num_repetitions = np.ceil(_NUM_PROCESSES.value / len(policies_to_use))
  policies_to_use = np.repeat(policies_to_use,
                              repeats=num_repetitions)
  policies_to_use = policies_to_use[:_NUM_PROCESSES.value]

  samples_per_process = _NUM_SAMPLES.value // len(policies_to_use)
  logging.info(f'Collecting {samples_per_process} samples per process.')
  assert samples_per_process > 0, 'Not enough samples to collect.'

  processes = []
  for i, policy_path in enumerate(policies_to_use):
    process = multiprocessing.Process(target=collect_samples,
                                      args=(policy_path, i,
                                            samples_per_process,
                                            _SEED.value + i))
    process.start()
    processes.append(process)

  for p in processes:
    p.join()

  logging.info('Finished collecting all samples.')
  # Now we simply need to load all of the datasets and merge them into one.
  dataset_ids = [
      f'{_DATASET_ID.value}-{_TASK.value}-{_SKILL_LEVEL.value}-{i}-v0' for i in
      range(len(policies_to_use))]

  logging.info(f'Combining datasets: {dataset_ids}')

  with multiprocessing.Pool(processes=len(policies_to_use)) as pool:
    datasets = pool.map(minari.load_dataset, dataset_ids)

  dataset = minari.combine_datasets(datasets_to_combine=datasets,
                                    copy=True,
                                    new_dataset_id=f'{_DATASET_ID.value}-{_TASK.value}-{_SKILL_LEVEL.value}-v0',
                                    )
  # log the datasets total steps and total episodes
  logging.info(f'Successfully combined datasets')
  logging.info(f'\tTotal steps: {dataset.total_steps}')
  logging.info(f'\tTotal episodes: {dataset.total_episodes}')

  logging.info(f'Cleaning up temporary datasets.')
  with multiprocessing.Pool(processes=len(policies_to_use)) as pool:
    pool.map(minari.delete_dataset, dataset_ids)
  logging.info(f'Finished cleaning up temporary datasets.')


if __name__ == '__main__':
  app.run(main)
