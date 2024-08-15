import minari
import tensorflow as tf
import tf_agents
from tf_agents.trajectories import Trajectory
from tf_agents.utils.nest_utils_test import DictWrapper


def minari_bc_dataset_iterator(dataset: minari.MinariDataset) -> tf.data.Dataset:
    """Creates an iterator for a MinariDataset that can be used for behavioral cloning.

    Args:
        dataset: The MinariDataset to create the iterator for.

    Returns:
        An iterator for the given MinariDataset.
    """

    while True:
        for episode in dataset:
            for i in range(episode.total_timesteps):
                if i == 0:
                    obs_step_type = tf_agents.trajectories.StepType.FIRST
                elif i == len(episode.observations) - 1:
                    obs_step_type = tf_agents.trajectories.StepType.LAST
                else:
                    obs_step_type = tf_agents.trajectories.StepType.MID

                next_obs_step_type = (
                    tf_agents.trajectories.StepType.LAST
                    if i == len(episode.observations) - 2
                    else tf_agents.trajectories.StepType.MID
                )

                # episode.observations may be a dict if the env uses a dict
                # observation space
                if isinstance(episode.observations, dict):
                    transition = Trajectory(
                        step_type=obs_step_type,
                        observation=DictWrapper(
                            {
                                k: tf.convert_to_tensor(v[i], dtype=v[i].dtype)
                                for k, v in episode.observations.items()
                            }
                        ),
                        action=tf.convert_to_tensor(
                            episode.actions[i], dtype=episode.actions[i].dtype
                        ),
                        policy_info=(),
                        next_step_type=next_obs_step_type,
                        reward=tf.convert_to_tensor(
                            episode.rewards[i], dtype=episode.rewards[i].dtype
                        ),
                        discount=(
                            tf.constant(0.0, dtype=tf.float32)
                            if obs_step_type == tf_agents.trajectories.StepType.LAST
                            else tf.constant(1.0, dtype=tf.float32)
                        ),
                    )
                else:
                    transition = Trajectory(
                        step_type=obs_step_type,
                        observation=episode.observations[i],
                        action=episode.actions[i],
                        policy_info=(),
                        next_step_type=next_obs_step_type,
                        reward=episode.rewards[i],
                        discount=(
                            tf.constant(0.0, dtype=tf.float32)
                            if obs_step_type == tf_agents.trajectories.StepType.LAST
                            else tf.constant(1.0, dtype=tf.float32)
                        ),
                    )

                expect_string = tf.constant("_", dtype=tf.string)
                yield transition, expect_string


def convert_to_tf_dataset(
    dataset: minari.MinariDataset,
    minari_dataset_iterator: callable = minari_bc_dataset_iterator,
    observation_spec: tf.TensorSpec = None,
    action_spec: tf.TensorSpec = None,
    batch_size: int = 32,
    shuffle_buffer_size: int = 1000,
) -> tf.data.Dataset:
    """Converts a MinariDataset to a TensorFlow Dataset.

    Args:
        dataset: The MinariDataset to convert.
        batch_size: The batch size for the TensorFlow Dataset.
        shuffle_buffer_size: The buffer size for shuffling the dataset.
        minari_dataset_iterator: The iterator to use for the MinariDataset.

    Returns:
        A TensorFlow Dataset.
    """
    iterator = minari_dataset_iterator(dataset)
    dataset = tf.data.Dataset.from_generator(
        lambda: iterator,
        output_signature=(
            tf_agents.trajectories.Trajectory(
                step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
                observation=observation_spec,
                action=action_spec,
                policy_info=(),
                next_step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
                reward=tf.TensorSpec(shape=(), dtype=tf.float32),
                discount=tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset
