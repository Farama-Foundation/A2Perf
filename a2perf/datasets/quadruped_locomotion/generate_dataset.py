import multiprocessing
import os
from collections import namedtuple

import gymnasium as gym
import minari
import numpy as np
from absl import app
from absl import flags
from absl import logging
from minari import DataCollectorV0

from a2perf.a2perf_benchmark_submission.quadruped_locomotion.ddpg import \
  ddpg_imitation
from a2perf.domains.quadruped_locomotion.motion_imitation.learning import \
  ppo_imitation

FLAGS = flags.FLAGS

_NUM_SAMPLES = flags.DEFINE_integer('num_samples', 100,
                                    'Number of samples to generate.')
_NUM_PROCESSES = flags.DEFINE_integer('num_processes', 1,
                                      'Number of processes to use.')

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
_SKILL_LEVEL = flags.DEFINE_enum('skill_level', 'novice',
                                 ['novice', 'intermediate', 'expert'],
                                 'Skill level of the expert.')
_TASK = flags.DEFINE_enum('task', 'dog_pace',
                          ['dog_pace', 'dog_trot', 'dog_spin'],
                          'Task to run.')
SKILL_LEVEL_DICT = {
    "dog_pace": {
        "expert": [
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_112518000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_165763125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_150693750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_156721500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_59272875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_176814000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_103476375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_95439375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_172795500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_173800125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_117541125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_159735375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_151698375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_110508750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_186860250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_196906500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_165763125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_157726125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_166767750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_179827875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_185855625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_96444000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_174804750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_171790875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_186860250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_151698375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_182841750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_194897250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_97448625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_195901875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_147679875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_137633625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_190878750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_186860250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_177818625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_182841750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_184851000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_120555000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_68314500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_199920375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_90416250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_199920375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_161744625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_192888000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_166767750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_105485625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_113522625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_150693750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_144666000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_132610500_steps.zip"
        ],
        "intermediate": [
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_139829250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_122564250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_39180375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_151698375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_131839250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_193892625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_60277500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_60928000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_78905500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_177818625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_129841750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_5996750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_182775500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_160740000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_91889250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_127587375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_143824250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_62925500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_64923000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_71914250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_72333000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_49226625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_9991750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_199754250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_174785500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_139829250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_9991750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_194760500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_156721500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_169781625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_62286750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_176814000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_170790500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_86895500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_199754250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_78905500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_187864875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_136833000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_154810500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_74342250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_66920500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_20978000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_77356125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_74342250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_137633625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_53936750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_132610500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_179827875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_105871750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_145670625_steps.zip"
        ],
        "novice": [
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_7032375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_3013875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_11050875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_36166500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_6027750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_3013875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_11050875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_18083250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_8037000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_20092500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_13060125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_18083250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_3013875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_11050875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_14064750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_15069375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_4018500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_14064750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_9041625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_12055500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_11050875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_1004625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_18083250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_10046250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_16074000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_8037000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_1004625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_6027750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_4250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_21097125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_15069375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_1004625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_4018500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_10046250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_9041625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_8037000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_8037000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_4250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_4018500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_15069375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_8037000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_6027750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_8037000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_1004625_steps.zip"
        ]
    },
    "dog_spin": {
        "expert": [
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_136833000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_156808000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_136833000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_147819250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_95884250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_199754250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_132838000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_88893000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_151814250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_160803000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_164798000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_139829250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_193761750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_199754250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_137831750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_144823000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_124848000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_148818000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_185771750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_115859250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_147819250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_159804250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_141826750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_186770500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_118855500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_135834250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_157806750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_141826750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_140828000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_131839250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_193761750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_166795500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_102875500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_125846750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_146820500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_137831750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_161801750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_176783000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_165796750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_198755500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_156808000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_134835500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_199754250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_165796750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_113861750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_106870500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_133836750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_117856750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_176783000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_147819250_steps.zip"
        ],
        "intermediate": [
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_115531875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_182841750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_45946750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_18980500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_107494875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_157726125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_102471750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_2009250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_142656750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_134619750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_83383875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_83899250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_46945500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_3999250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_89891750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_181837125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_52240500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_86895500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_10046250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_43949250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_111513375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_51939250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_10046250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_43198875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_63924250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_60277500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_60277500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_169781625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_40185000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_172795500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_52938000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_31143375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_42950500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_57263625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_25115625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_183846375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_97448625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_165763125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_44948000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_87402375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_42194250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_125578125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_131839250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_81901750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_116536500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_135624375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_38955500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_110865500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_65300625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_spin/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_106490250_steps.zip"
        ]
    },
    "dog_trot": {
        "expert": [
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_133836750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_172788000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_163799250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_145821750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_146820500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_149816750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_103874250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_177781750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_79904250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_162800500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_184773000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_191764250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_171789250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_199754250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_178780500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_148818000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_168793000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_154810500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_174785500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_147819250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_151698375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_96444000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_192763000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_185771750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_196758000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_192763000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_43949250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_182775500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_165796750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_123849250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_196758000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_158805500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_129596625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_179779250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_169791750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_164798000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_175809375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_165796750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_172788000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_133836750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_97881750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_150693750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_63924250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_171789250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_154810500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_101876750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_95884250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_126582750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_197756750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_175784250_steps.zip"
        ],
        "intermediate": [
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_3000500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_16983000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_127587375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_85896750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_69319125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_119854250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_68314500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_83383875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_1004625_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_74342250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_10046250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_10046250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_58930500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_35959250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_190878750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_130601250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_107494875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_34157250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_336_total_env_steps_200000000/policies/rl_policy_168777000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_163753875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_38175750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_184851000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_134835500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_117856750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_106490250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_95884250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_43949250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_88407000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_57931750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_55254375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_138638250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_54249750_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_170786250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_293_total_env_steps_200000000/policies/rl_policy_35161875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_186860250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_31964250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_59929250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_43198875_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_65_total_env_steps_200000000/policies/rl_policy_109504125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_215_total_env_steps_200000000/policies/rl_policy_14985500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_318_total_env_steps_200000000/policies/rl_policy_95884250_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_5023125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_48222000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_20978000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_303_total_env_steps_200000000/policies/rl_policy_62925500_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_159735375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_345_total_env_steps_200000000/policies/rl_policy_151698375_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ppo/2223/quadruped_locomotion_algo_ppo_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_98_total_env_steps_200000000/policies/rl_policy_72913000_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_491_total_env_steps_200000000/policies/rl_policy_133615125_steps.zip",
            "/mnt/gcs/a2perf/quadruped_locomotion/dog_trot/ddpg/2223/quadruped_locomotion_algo_ddpg_int_eval_freq_100000_int_save_freq_1000000_parallel_cores_170_seed_182_total_env_steps_200000000/policies/rl_policy_22101750_steps.zip"
        ]
    }
}


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


DummyModel = namedtuple('DummyModel', 'predict')


def collect_samples(policy_path, unique_id, samples_to_collect, seed,
    use_random_policy=False):
  os.environ['MINARI_DATASETS_PATH'] = os.path.join(_DATASETS_PATH.value,
                                                    unique_id)

  # Need to set random seed again since we are in a new process.
  set_seed(seed)
  env = gym.make('QuadrupedLocomotion-v0', mode='test', motion_files=[
      f'/rl-perf/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/{_TASK.value}.txt'],
                 enable_rendering=False)
  env = DataCollectorV0(env=env, observation_space=gym.spaces.Box(low=-np.inf,
                                                                  high=np.inf,
                                                                  shape=(
                                                                      env.observation_space.shape[
                                                                        0],),
                                                                  dtype=np.float64),
                        action_space=env.action_space, max_buffer_steps=None,
                        record_infos=True, )

  if use_random_policy:

    assert _SKILL_LEVEL.value == 'novice', 'Random policy is only for novice skill level.'

    # Define the predict function
    def predict_function(observation):
      return env.action_space.sample(), None

    # Create the namedtuple instance with the predict function
    model = DummyModel(predict=predict_function)
    algorithm_name = 'random'
    dataset_id = f'{_DATASET_ID.value}-{_TASK.value}-novice-{unique_id}-v0'  # Random policies are of skill level "novice".
  else:
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


def load_dataset_wrapper(dataset_id, unique_id):
  unique_id = f'{unique_id:03d}'
  os.environ['MINARI_DATASETS_PATH'] = f'{_DATASETS_PATH.value}/{unique_id}'
  return minari.load_dataset(dataset_id)


def delete_dataset_wrapper(dataset_id, unique_id):
  unique_id = f'{unique_id:03d}'
  os.environ['MINARI_DATASETS_PATH'] = f'{_DATASETS_PATH.value}/{unique_id}'
  minari.delete_dataset(dataset_id)


def main(_):
  if _DATASETS_PATH.value is not None:
    os.environ['MINARI_DATASETS_PATH'] = _DATASETS_PATH.value

  set_seed(_SEED.value)

  # Some dicts may not exist depending on how difficult the task is.
  task_skill_level_dict = SKILL_LEVEL_DICT[_TASK.value]
  policies_to_use = task_skill_level_dict.get(_SKILL_LEVEL.value, None)

  if policies_to_use is None:
    logging.info(f'No policies found for task: {_TASK.value} and skill level: '
                 f'{_SKILL_LEVEL.value}.')

    with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
      samples_per_process = _NUM_SAMPLES.value // _NUM_PROCESSES.value
      tasks = [(None, f'{i:03d}', samples_per_process, _SEED.value + i, True)
               for i in range(_NUM_PROCESSES.value)]
      pool.starmap(collect_samples, tasks)
  else:
    policies_to_use = np.array(policies_to_use)
    np.random.shuffle(policies_to_use)
    num_repetitions = np.ceil(_NUM_PROCESSES.value / len(policies_to_use))
    policies_to_use = np.repeat(policies_to_use,
                                repeats=num_repetitions)
    policies_to_use = policies_to_use[:_NUM_PROCESSES.value]

    samples_per_process = _NUM_SAMPLES.value // len(policies_to_use)
    logging.info(f'Collecting {samples_per_process} samples per process.')
    assert samples_per_process > 0, 'Not enough samples to collect.'

    # Create a pool of workers
    with  multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
      tasks = [(policy_path, f'{i:03d}', samples_per_process, _SEED.value + i)
               for i, policy_path in enumerate(policies_to_use)]

      # Map collect_samples function to the tasks
      pool.starmap(collect_samples, tasks)

    logging.info('Finished collecting all samples.')

  logging.info('Finished collecting all samples.')
  # Now we simply need to load all of the datasets and merge them into one.
  dataset_ids = [
      f'{_DATASET_ID.value}-{_TASK.value}-{_SKILL_LEVEL.value}-{i:03d}-v0' for i
      in
      range(_NUM_PROCESSES.value)]

  logging.info(f'Combining datasets: {dataset_ids}')

  with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
    datasets = pool.starmap(load_dataset_wrapper,
                            zip(dataset_ids, range(len(dataset_ids))))

  dataset = minari.combine_datasets(datasets_to_combine=datasets,
                                    copy=True,
                                    new_dataset_id=f'{_DATASET_ID.value}-{_TASK.value}-{_SKILL_LEVEL.value}-v0',
                                    )
  # log the datasets total steps and total episodes
  logging.info(f'Successfully combined datasets')
  logging.info(f'\tTotal steps: {dataset.total_steps}')
  logging.info(f'\tTotal episodes: {dataset.total_episodes}')

  logging.info(f'Cleaning up temporary datasets.')
  with multiprocessing.Pool(processes=_NUM_PROCESSES.value) as pool:
    pool.starmap(delete_dataset_wrapper,
                 zip(dataset_ids, range(len(dataset_ids))))
  logging.info(f'Finished cleaning up temporary datasets.')


if __name__ == '__main__':
  app.run(main)
