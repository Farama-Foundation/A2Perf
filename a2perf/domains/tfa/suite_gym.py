# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Suite for loading Gym Environments.

Note we use gym.spec(env_id).make() on gym envs to avoid getting a TimeLimit
wrapper on the environment. OpenAI's TimeLimit wrappers terminate episodes
without indicating if the failure is due to the time limit, or due to negative
agent behaviour. This prevents us from setting the appropriate discount value
for the final step of an episode. To prevent that we extract the step limit
from the environment specs and utilize our TimeLimit wrapper.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Text

import gin
import gymnasium as gym
import numpy as np
from absl import logging
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.typing import types

from a2perf.domains import circuit_training  # noqa: F401
from a2perf.domains import quadruped_locomotion  # noqa: F401
from a2perf.domains import web_navigation  # noqa: F401
from a2perf.domains.tfa import gym_wrapper

TimeLimitWrapperType = Callable[
    [py_environment.PyEnvironment, int], py_environment.PyEnvironment
]

WEB_NAVIGATION_ENVS = ("WebNavigation-DifficultyLevel-01-v0",)
CIRCUIT_TRAINING_ENVS = ("CircuitTraining-Ariane-v0", "CircuitTraining-ToyMacro-v0")
QUADRUPED_LOCOMOTION_ENVS = (
    "QuadrupedLocomotion-DogPace-v0",
    "QuadrupedLocomotion-DogTrot-v0",
    "QuadrupedLocomotion-DogSpin-v0",
)


@gin.configurable
def load(
    environment_name: Text,
    discount: types.Float = 1.0,
    max_episode_steps: Optional[types.Int] = None,
    gym_env_wrappers: Sequence[types.GymEnvWrapper] = (),
    env_wrappers: Sequence[types.PyEnvWrapper] = (),
    spec_dtype_map: Optional[Dict[gym.Space, np.dtype]] = None,
    gym_kwargs: Optional[Dict[str, Any]] = None,
    render_kwargs: Optional[Dict[str, Any]] = None,
) -> py_environment.PyEnvironment:
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
      environment_name: Name for the environment to load.
      discount: Discount to use for the environment.
      max_episode_steps: If None the max_episode_steps will be set to the default
        step limit defined in the environment's spec. No limit is applied if set
        to 0 or if there is no max_episode_steps set in the environment's spec.
      gym_env_wrappers: Iterable with references to wrapper classes to use
        directly on the gym environment.
      env_wrappers: Iterable with references to wrapper classes to use on the
        gym_wrapped environment.
      spec_dtype_map: A dict that maps gym spaces to np dtypes to use as the
        default dtype for the arrays. An easy way how to configure a custom
        mapping through Gin is to define a gin-configurable function that returns
        desired mapping and call it in your Gin congif file, for example:
        `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.
      gym_kwargs: Optional kwargs to pass to the Gym environment class.
      render_kwargs: Optional kwargs for rendering to pass to `render()` of the
        gym_wrapped environment.

    Returns:
      A PyEnvironment instance.
    """
    gym_kwargs = gym_kwargs if gym_kwargs else {}
    gym_spec = gym.spec(environment_name)
    gym_env = gym_spec.make(**gym_kwargs)

    if max_episode_steps is None and gym_spec.max_episode_steps is not None:
        max_episode_steps = gym_spec.max_episode_steps

    return wrap_env(
        gym_env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map,
        render_kwargs=render_kwargs,
    )


@gin.configurable
def wrap_env(
    gym_env: gym.Env,
    discount: types.Float = 1.0,
    max_episode_steps: Optional[types.Int] = None,
    gym_env_wrappers: Sequence[types.GymEnvWrapper] = (),
    time_limit_wrapper: TimeLimitWrapperType = wrappers.TimeLimit,
    env_wrappers: Sequence[types.PyEnvWrapper] = (),
    spec_dtype_map: Optional[Dict[gym.Space, np.dtype]] = None,
    auto_reset: bool = True,
    render_kwargs: Optional[Dict[str, Any]] = None,
) -> py_environment.PyEnvironment:
    """Wraps given gym environment with TF Agent's GymWrapper.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
      gym_env: An instance of OpenAI gym environment.
      discount: Discount to use for the environment.
      max_episode_steps: Used to create a TimeLimitWrapper. No limit is applied if
        set to None or 0. Usually set to `gym_spec.max_episode_steps` in `load`.
      gym_env_wrappers: Iterable with references to wrapper classes to use
        directly on the gym environment.
      time_limit_wrapper: Wrapper that accepts (env, max_episode_steps) params to
        enforce a TimeLimit. Usuaully this should be left as the default,
        wrappers.TimeLimit.
      env_wrappers: Iterable with references to wrapper classes to use on the
        gym_wrapped environment.
      spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
        default dtype for the tensors. An easy way how to configure a custom
        mapping through Gin is to define a gin-configurable function that returns
        desired mapping and call it in your Gin config file, for example:
        `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.
      auto_reset: If True (default), reset the environment automatically after a
        terminal state is reached.
      render_kwargs: Optional `dict` of keywoard arguments for rendering.

    Returns:
      A PyEnvironment instance.
    """

    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)
    env = gym_wrapper.GymWrapper(
        gym_env,
        discount=discount,
        spec_dtype_map=spec_dtype_map,
        auto_reset=auto_reset,
        render_kwargs=render_kwargs,
    )

    if max_episode_steps is not None and max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in env_wrappers:
        env = wrapper(env)

    return env


@gin.configurable
def create_domain(
    env_name, root_dir=None, env_wrappers=(), gym_env_wrappers=(), **env_kwargs
):
    if env_name in WEB_NAVIGATION_ENVS:
        # noinspection PyUnresolvedReferences
        from a2perf.domains import web_navigation  # noqa: F401
        from a2perf.domains.web_navigation.gwob.CoDE import vocabulary_node

        save_vocab_dir = os.path.join(root_dir, "vocabulary")
        reload_vocab = env_kwargs.pop("reload_vocab", True)
        vocab_type = env_kwargs.pop("vocab_type", "threaded")
        if vocab_type == "threaded":
            global_vocab = vocabulary_node.LockedThreadedVocabulary()
        elif vocab_type == "unlocked":
            global_vocab = vocabulary_node.UnlockedVocabulary()
        elif vocab_type == "multiprocessing":
            global_vocab = vocabulary_node.LockedMultiprocessingVocabulary()
        else:
            raise ValueError(f"Unknown vocabulary type: {vocab_type}")

        if os.path.exists(save_vocab_dir) and reload_vocab:
            vocab_files = os.listdir(save_vocab_dir)
            if vocab_files:
                vocab_files.sort()
                latest_vocab_file = vocab_files[-1]
                with open(os.path.join(save_vocab_dir, latest_vocab_file), "r") as f:
                    global_vocab_dict = json.load(f)
                    global_vocab.restore(state=global_vocab_dict)
        seed = int(os.environ.get("SEED", None))
        num_websites = int(os.environ.get("NUM_WEBSITES", None))
        difficulty = int(os.environ.get("DIFFICULTY_LEVEL", None))

        env_kwargs.update(
            {
                "global_vocabulary": global_vocab,
                "seed": seed,
                "num_websites": num_websites,
                "difficulty": difficulty,
                "browser_args": dict(
                    threading=False,
                    chrome_options={
                        "--headless",
                        "--no-sandbox",
                        "--disable-gpu",
                        # '--disable-dev-shm-usage',
                    },
                ),
            }
        )
        env_wrappers = [wrappers.ActionClipWrapper] + list(env_wrappers)
    elif env_name in CIRCUIT_TRAINING_ENVS:
        # noinspection PyUnresolvedReferences
        from a2perf.domains import circuit_training  # noqa: F401

        env_kwargs.pop("netlist", None)
        netlist_file_path = os.environ.get("NETLIST_PATH", None)
        seed = int(os.environ.get("SEED", None))
        init_placement_file_path = os.environ.get("INIT_PLACEMENT_PATH", None)
        std_cell_placer_mode = os.environ.get("STD_CELL_PLACER_MODE", None)
        env_kwargs.update(
            {
                "global_seed": seed,
                "netlist_file": netlist_file_path,
                "init_placement": init_placement_file_path,
                "output_plc_file": os.path.join(root_dir, "output.plc"),
                "std_cell_placer_mode": std_cell_placer_mode,
            }
        )
        env_wrappers = [wrappers.ActionClipWrapper] + list(env_wrappers)
    elif env_name in QUADRUPED_LOCOMOTION_ENVS:
        # noinspection PyUnresolvedReferences
        from a2perf.domains import quadruped_locomotion  # noqa: F401

        motion_file_path = os.environ.get("MOTION_FILE_PATH", None)
        env_kwargs["motion_files"] = [motion_file_path]
        env_wrappers = [wrappers.ActionClipWrapper] + list(env_wrappers)
    else:
        raise NotImplementedError(f"Unknown environment: {env_name}")

    logging.info("Creating domain %s with kwargs %s", env_name, env_kwargs)
    return load(
        environment_name=env_name,
        env_wrappers=env_wrappers,
        gym_env_wrappers=gym_env_wrappers,
        gym_kwargs=env_kwargs,
    )
