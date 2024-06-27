import collections
import functools
import importlib
import json
import multiprocessing
import os
import sys
import timeit
import typing
from contextlib import contextmanager

import codecarbon
import gin
import numpy as np
import pkg_resources
from absl import flags
from absl import logging
from tf_agents.metrics import py_metrics
from tf_agents.train import actor

from a2perf.constants import BenchmarkDomain
from a2perf.constants import BenchmarkMode
from a2perf.constants import ReliabilityMetrics
from a2perf.domains.tfa import suite_gym


def parse_participant_args(args_string):
    if not args_string:
        return {}
    try:
        return dict(arg.split("=", 1) for arg in args_string.split(","))
    except ValueError:
        raise ValueError(
            "Invalid format in participant arguments. "
            "Please use the format 'key1=value1,key2=value2'"
        )


@contextmanager
def working_directory(path):
    """Context manager for temporarily changing the working directory."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        sys.path.remove(path)


@gin.configurable
def track_emissions_decorator(
    project_name: typing.Optional[str] = None,
    measure_power_secs: typing.Optional[int] = None,
    api_call_interval: typing.Optional[int] = None,
    api_endpoint: typing.Optional[str] = None,
    api_key: typing.Optional[str] = None,
    output_dir: typing.Optional[str] = None,
    output_file: typing.Optional[str] = None,
    save_to_file: typing.Optional[bool] = None,
    save_to_api: typing.Optional[bool] = None,
    save_to_logger: typing.Optional[bool] = None,
    save_to_prometheus: typing.Optional[bool] = None,
    prometheus_url: typing.Optional[str] = None,
    logging_logger: typing.Optional[
        codecarbon.output_methods.logger.LoggerOutput
    ] = None,
    offline: typing.Optional[bool] = None,
    emissions_endpoint: typing.Optional[str] = None,
    experiment_id: typing.Optional[str] = None,
    country_iso_code: typing.Optional[str] = None,
    region: typing.Optional[str] = None,
    cloud_provider: typing.Optional[str] = None,
    cloud_region: typing.Optional[str] = None,
    gpu_ids: typing.Optional[typing.List] = None,
    co2_signal_api_token: typing.Optional[str] = None,
    log_level: typing.Optional[typing.Union[int, str]] = None,
    default_cpu_power: typing.Optional[int] = None,
    pue: typing.Optional[float] = 1.0,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            decorator_instance = codecarbon.track_emissions(
                project_name=project_name,
                measure_power_secs=measure_power_secs,
                api_call_interval=api_call_interval,
                api_endpoint=api_endpoint,
                api_key=api_key,
                output_dir=output_dir,
                output_file=output_file,
                save_to_file=save_to_file,
                save_to_api=save_to_api,
                save_to_logger=save_to_logger,
                save_to_prometheus=save_to_prometheus,
                prometheus_url=prometheus_url,
                logging_logger=logging_logger,
                offline=offline,
                emissions_endpoint=emissions_endpoint,
                experiment_id=experiment_id,
                country_iso_code=country_iso_code,
                region=region,
                cloud_provider=cloud_provider,
                cloud_region=cloud_region,
                gpu_ids=gpu_ids,
                co2_signal_api_token=co2_signal_api_token,
                log_level=log_level,
                default_cpu_power=default_cpu_power,
                pue=pue,
            )
            return decorator_instance(func)(*args, **kwargs)

        return wrapper

    return decorator


def setup_subprocess_env(gin_config_str, absl_flags):
    # Parse the gin config
    gin.parse_config(gin_config_str)
    logging.info("Gin config parsed")

    # Register absl flags from the dictionary
    for flag_name, flag_value in absl_flags.items():
        if flag_name in flags.FLAGS:
            flags.FLAGS[flag_name].value = flag_value
            logging.info("Flag %s set to %s", flag_name, flag_value)
        else:
            logging.warning("Flag %s not found", flag_name)


def _load_spec(module_path, filename):
    """Loads the spec from the given module path."""
    participant_file_path = os.path.join(module_path, filename)
    spec = importlib.util.spec_from_file_location(f"{filename}", participant_file_path)
    return spec


def _load_module(module_path, filename):
    """Loads the module from the given module path."""
    spec = _load_spec(module_path, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, spec


def _load_policy(module_path, env):
    """Loads the policy from the participant's module."""
    with working_directory(module_path):
        participant_module, participant_module_spec = _load_module(
            module_path, "inference.py"
        )
        policy = participant_module.load_policy(env)
    return policy, participant_module


def perform_rollouts(
    module_path,
    create_domain_fn,
    num_episodes=1,
    gin_config_str=None,
    absl_flags=None,
    rollout_rewards_queue=None,
):
    """Performs rollouts using the given policy.

    Args:
        create_domain_fn: Function that creates the domain.
        preprocess_obs_fn: Function that preprocesses the observation.
        infer_once_fn: Function that performs inference.
        num_episodes: Number of episodes to perform rollouts.
        policy: Policy to use for performing rollouts.
        gin_config_str: Gin config string to use for creating the domain.

    Returns:
        List of rewards from each episode.
    """
    setup_subprocess_env(gin_config_str, absl_flags)
    env = create_domain_fn()
    policy, participant_module = _load_policy(module_path, env)
    episode_reward_metric = py_metrics.AverageReturnMetric()
    rollout_actor = actor.Actor(
        env=env,
        train_step=policy._train_step_from_last_restored_checkpoint_path,
        policy=policy,
        observers=[episode_reward_metric],
        episodes_per_run=1,
    )

    all_rewards = []
    for _ in range(num_episodes):
        rollout_actor.run()
        all_rewards.append(float(episode_reward_metric.result()))
        episode_reward_metric.reset()

    if rollout_rewards_queue:
        for reward in all_rewards:
            rollout_rewards_queue.put(reward)

    return all_rewards


def _perform_rollout_task(
    generalization_task,
    domain,
    root_dir,
    participant_module_path,
    num_generalization_episodes,
    gin_config_str,
    absl_flags,
):
    generalization_env_vars = {}

    if domain == BenchmarkDomain.WEB_NAVIGATION:
        if generalization_task == "difficulty_level_1_num_websites_1":
            generalization_env_vars["DIFFICULTY_LEVEL"] = "1"
            generalization_env_vars["NUM_WEBSITES"] = "1"
        elif generalization_task == "difficulty_level_1_num_websites_5":
            generalization_env_vars["DIFFICULTY_LEVEL"] = "1"
            generalization_env_vars["NUM_WEBSITES"] = "5"
        elif generalization_task == "difficulty_level_1_num_websites_10":
            generalization_env_vars["DIFFICULTY_LEVEL"] = "1"
            generalization_env_vars["NUM_WEBSITES"] = "10"
        else:
            raise ValueError(
                "Generalization tasks for WebNavigation domain must be either"
                " difficulty_level_1_num_websites_1, "
                "difficulty_level_1_num_websites_5, "
                "or difficulty_level_1_num_websites_10"
            )
    elif domain == BenchmarkDomain.CIRCUIT_TRAINING:
        if generalization_task == "toy_macro_stdcell":
            generalization_env_vars["NETLIST_PATH"] = pkg_resources.resource_filename(
                "a2perf",
                "domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt",
            )
            generalization_env_vars["INIT_PLACEMENT_PATH"] = (
                pkg_resources.resource_filename(
                    "a2perf",
                    "domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/initial.plc",
                )
            )
        elif generalization_task == "ariane":
            generalization_env_vars["NETLIST_PATH"] = pkg_resources.resource_filename(
                "a2perf",
                "domains/circuit_training/circuit_training/environment/test_data/ariane/netlist.pb.txt",
            )
            generalization_env_vars["INIT_PLACEMENT_PATH"] = (
                pkg_resources.resource_filename(
                    "a2perf",
                    "domains/circuit_training/circuit_training/environment/test_data/ariane/initial.plc",
                )
            )
        else:
            raise ValueError(
                "Generalization tasks for CircuitTraining domain must be either toy_macro_stdcell or ariane"
            )
    elif domain == BenchmarkDomain.QUADRUPED_LOCOMOTION:
        if generalization_task == "dog_pace":
            generalization_env_vars["MOTION_FILE_PATH"] = (
                pkg_resources.resource_filename(
                    "a2perf",
                    "domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt",
                )
            )
        elif generalization_task == "dog_trot":
            generalization_env_vars["MOTION_FILE_PATH"] = (
                pkg_resources.resource_filename(
                    "a2perf",
                    "domains/quadruped_locomotion/motion_imitation/data/motions/dog_trot.txt",
                )
            )
        elif generalization_task == "dog_spin":
            generalization_env_vars["MOTION_FILE_PATH"] = (
                pkg_resources.resource_filename(
                    "a2perf",
                    "domains/quadruped_locomotion/motion_imitation/data/motions/dog_spin.txt",
                )
            )
    else:
        raise ValueError(
            "Generalization tasks are only supported for WebNavigation, CircuitTraining, and QuadrupedLocomotion domains."
        )

    for key, value in generalization_env_vars.items():
        os.environ[key] = value

    create_domain_fn = functools.partial(
        suite_gym.create_domain, env_name=domain.value, root_dir=root_dir
    )
    all_rewards = perform_rollouts(
        module_path=participant_module_path,
        create_domain_fn=create_domain_fn,
        num_episodes=num_generalization_episodes,
        gin_config_str=gin_config_str,
        absl_flags=absl_flags,
        rollout_rewards_queue=None,
    )

    # Reset the environment variables after rolling out each task
    for key in generalization_env_vars.keys():
        os.environ.pop(key)

    return generalization_task, all_rewards


def train(
    module_path, gin_config_str=None, absl_flags=None, participant_args: dict = None
):
    """Trains the participant's policy."""
    setup_subprocess_env(gin_config_str, absl_flags)
    with working_directory(module_path):
        participant_module, participant_module_spec = _load_module(
            module_path, "train.py"
        )
        if participant_args is None:
            participant_args = {}
        print(participant_args)
        participant_module.train(**participant_args)


@gin.configurable
class Submission:

    def __init__(
        self,
        root_dir: str,
        metric_values_dir: str,
        participant_module_path: str,
        mode: BenchmarkMode,
        domain: BenchmarkDomain,
        participant_args: str,
        num_inference_steps: int = 1000,
        num_inference_episodes: int = 1,
        num_generalization_episodes: int = 1,
        time_participant_code: bool = True,
        measure_emissions: bool = False,
        plot_metrics: bool = True,
        run_offline_metrics_only: bool = False,
        reliability_metrics: typing.List[ReliabilityMetrics] = None,
        generalization_tasks: typing.List = None,
    ):
        """Object that represents a submission to the benchmark.

        Args:
            participant_module_path: Path to the module that contains the
              participant's code.
            mode: Benchmark mode (train or inference).
            domain: Benchmark domain (web navigation, circuit training or quadruped
              locomotion).
            root_dir: Root directory for the submission.
            metric_values_dir: Directory where the metric values will be saved.
            num_inference_steps: Number of steps to run inference for.
            num_inference_episodes: Number of episodes to run inference for.
            time_participant_code: Whether to time the participant's code.
            measure_emissions: Whether to measure emissions.
            plot_metrics: Whether to plot the metrics.
            run_offline_metrics_only: Whether to run only the offline metrics.
            reliability_metrics: List of reliability metrics to compute.
        """
        self.root_dir = root_dir
        self.generalization_tasks = generalization_tasks
        self.metric_values_dir = metric_values_dir
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.metric_values_dir, exist_ok=True)
        self.run_offline_metrics_only = run_offline_metrics_only
        self.mp_context = multiprocessing.get_context("spawn")
        self.gin_config_str = None
        self.absl_flags = None

        self.participant_args = parse_participant_args(participant_args)
        self.measure_emissions = measure_emissions
        self.plot_metrics = plot_metrics
        self.num_inference_steps = num_inference_steps
        self.num_inference_episodes = num_inference_episodes
        self.num_generalization_episodes = num_generalization_episodes
        self.time_inference_steps = time_participant_code
        self.participant_module_path = os.path.abspath(participant_module_path)
        self.domain = domain
        self.mode = mode
        self.reliability_metrics = reliability_metrics

        self.metrics_results = {}

        if self.mode == BenchmarkMode.TRAIN:
            metrics_path = os.path.join(self.metric_values_dir, "train_metrics.json")
        elif self.mode == BenchmarkMode.INFERENCE:
            metrics_path = os.path.join(
                self.metric_values_dir, "inference_metrics.json"
            )
        elif self.mode == BenchmarkMode.GENERALIZATION:
            metrics_path = os.path.join(
                self.metric_values_dir, "generalization_metrics.json"
            )
        else:
            raise ValueError(
                "Benchmark mode must be either train, inference or generalization"
            )

        if os.path.exists(metrics_path):
            logging.info(f"Loading pre-existing metric results from {metrics_path}")
            with open(metrics_path, "r") as f:
                self.metrics_results = json.load(f)

    def _get_observation_data(self, env):
        data = []
        for _ in range(self.num_inference_steps):
            observation = env.observation_space.sample()
            data.append(observation)
        return data

    def _train(
        self,
    ):
        setup_subprocess_env(self.gin_config_str, self.absl_flags)

        @track_emissions_decorator(
            output_dir=self.metric_values_dir, output_file="train_emissions.csv"
        )
        def train_and_track_emissions():
            train_process = self.mp_context.Process(
                target=train,
                args=(
                    self.participant_module_path,
                    self.gin_config_str,
                    self.absl_flags,
                    self.participant_args,
                ),
            )
            train_process.start()
            train_process.join()

        if self.measure_emissions:
            return train_and_track_emissions()
        else:
            return train(
                self.participant_module_path,
                self.gin_config_str,
                self.absl_flags,
                self.participant_args,
            )

    def _perform_rollouts(
        self, num_episodes, measure_emissions, output_dir, rollout_rewards_queue
    ):
        """
        Perform rollouts and optionally track emissions.

        Args:
            num_episodes: Number of episodes to perform rollouts.
            measure_emissions: Flag to indicate if emissions should be measured.
            output_dir: Directory to save the emissions data.

        Returns:
            List of rewards from each episode.
        """
        setup_subprocess_env(self.gin_config_str, self.absl_flags)

        create_domain_fn = functools.partial(
            suite_gym.create_domain, env_name=self.domain.value, root_dir=self.root_dir
        )
        if measure_emissions:

            @track_emissions_decorator(
                output_dir=output_dir, output_file="inference_emissions.csv"
            )
            def perform_rollouts_and_track_emissions():
                rollout_process = multiprocessing.Process(
                    target=perform_rollouts,
                    args=(
                        self.participant_module_path,
                        create_domain_fn,
                        num_episodes,
                        self.gin_config_str,
                        self.absl_flags,
                        rollout_rewards_queue,
                    ),
                )
                rollout_process.start()
                rollout_process.join()

            return perform_rollouts_and_track_emissions()
        else:
            return perform_rollouts(
                create_domain_fn=create_domain_fn,
                num_episodes=num_episodes,
                module_path=self.participant_module_path,
                gin_config_str=self.gin_config_str,
                absl_flags=self.absl_flags,
            )

    def _run_training_benchmark(self):
        if not self.run_offline_metrics_only:
            participant_training_process = self.mp_context.Process(
                target=self._train,
            )

            participant_training_process.start()
            logging.info(
                f"Participant training process ID: {participant_training_process.pid}"
            )

            participant_training_process.join()
            logging.info(
                f"Participant module process {participant_training_process.pid} finished"
            )

            if participant_training_process.is_alive():
                logging.error("Participant process is still running")
            elif participant_training_process.exitcode != 0:
                logging.error(
                    "Participant process exited with code"
                    f" {participant_training_process.exitcode}"
                )
            else:
                logging.info(
                    f"Participant process {participant_training_process.pid} finished"
                )

    def _run_generalization_benchmark(self):
        all_returns = collections.defaultdict(list)
        tasks = []

        for generalization_task in self.generalization_tasks:
            logging.info("Running generalization task: %s", generalization_task)
            tasks.append(
                (
                    generalization_task,
                    self.domain,
                    self.root_dir,
                    self.participant_module_path,
                    self.num_generalization_episodes,
                    self.gin_config_str,
                    self.absl_flags,
                )
            )

        with multiprocessing.Pool() as pool:
            results = pool.starmap(_perform_rollout_task, tasks)
            pool.close()
            pool.join()

        for generalization_task, all_rewards in results:
            all_returns[generalization_task] = all_rewards

        # Save the rollouts for each generalization task to a file
        logging.info("Saving generalization rollouts to file")
        logging.info("Generalization rollouts: %s", all_returns)
        with open(
            os.path.join(self.metric_values_dir, "generalization_rollouts.json"), "w"
        ) as f:
            json.dump(all_returns, f)

    def _run_inference_benchmark(self):
        if not self.run_offline_metrics_only:
            logging.info("Creating Gymnasium domain...")
            env = suite_gym.create_domain(
                env_name=self.domain.value, root_dir=self.root_dir
            )
            logging.info("Successfully created domain")

            logging.info("Generating inference data...")
            inference_data = self._get_observation_data(env)
            logging.info("Successfully generated inference data")

            metric_results = {}

            logging.info("Loading the policy for inference...")
            participant_policy, participant_module = _load_policy(
                module_path=self.participant_module_path, env=env
            )

            # Only include time_step_spec if the participant policy has it as an
            # attribute. This will be useful for participants using TF agents.
            time_step_spec = getattr(participant_policy, "time_step_spec", None)
            preprocessed_data = [
                participant_module.preprocess_observation(
                    x, time_step_spec=time_step_spec
                )
                for x in inference_data
            ]
            logging.info("Finished preprocessing the observation data")

            if self.time_inference_steps:
                logging.info("Timing inference steps...")
                inference_times = []
                for i in range(self.num_inference_steps):
                    inference_step = lambda: participant_module.infer_once(
                        policy=participant_policy,
                        preprocessed_observation=preprocessed_data[i],
                    )
                    inference_times.append(timeit.timeit(inference_step, number=1))
                logging.info("Finished timing inference steps")

                metric_results["inference_time"] = {
                    "values": inference_times,
                    "mean": np.mean(inference_times),
                    "std": np.std(inference_times),
                    "max": np.max(inference_times),
                    "median": np.median(inference_times),
                    "min": np.min(inference_times),
                }

            # Running rollouts in a subprocess
            rollout_returns_queue = multiprocessing.Queue()
            rollout_process = multiprocessing.Process(
                target=self._perform_rollouts,
                args=(
                    self.num_inference_episodes,
                    self.measure_emissions,
                    self.metric_values_dir,
                    rollout_returns_queue,
                ),
            )

            rollout_process.start()
            rollout_process.join()

            all_rewards = []
            while not rollout_returns_queue.empty():
                all_rewards.append(rollout_returns_queue.get())

            print(f"All rewards: {all_rewards}")
            metric_results["rollout_returns"] = {
                "values": [float(reward) for reward in all_rewards],
                "mean": np.mean(all_rewards).astype(float),
                "std": np.std(all_rewards).astype(float),
                "max": np.max(all_rewards).astype(float),
                "median": np.median(all_rewards).astype(float),
                "min": np.min(all_rewards).astype(float),
            }

            logging.info("Metrics Results: %s", metric_results)
            with open(
                os.path.join(self.metric_values_dir, "inference_metrics_results.json"),
                "w",
            ) as f:
                json.dump(metric_results, f)

    def run_benchmark(self):
        # Gin configs and absl flags must be saved to pass to subprocesses
        self.gin_config_str = gin.config_str()
        self.absl_flags = {name: flags.FLAGS[name].value for name in flags.FLAGS}

        if not os.path.exists(self.participant_module_path):
            raise FileNotFoundError(
                f"Participant module path {self.participant_module_path} not found. This is necessary for running training and inference code."
            )

        if self.mode == BenchmarkMode.TRAIN:
            self._run_training_benchmark()
        elif self.mode == BenchmarkMode.INFERENCE:
            if not os.path.exists(self.root_dir):
                raise FileNotFoundError(
                    f"Root directory {self.root_dir} not found. This is necessary for loading the trained model"
                )
            self._run_inference_benchmark()
        elif self.mode == BenchmarkMode.GENERALIZATION:
            if not os.path.exists(self.root_dir):
                raise FileNotFoundError(
                    f"Root directory {self.root_dir} not found. This is necessary for loading the trained model"
                )
            self._run_generalization_benchmark()

        else:
            raise ValueError("Benchmark mode must be either train or inference")
