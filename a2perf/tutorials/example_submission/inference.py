import stable_baselines3 as sb3
from absl import app


def load_model(env):
    # This function is intended to load and return the model.
    # The `env` parameter can be used to specify the environment or any other
    # context needed for loading the model.
    model = sb3.PPO.load("ppo_cartpole")
    return model


def infer_once(model, observation):
    # Use the model (assumed to be from Stable Baselines) to run inference on a
    # single observation. The function receives a `model` and an `observation`,
    # then returns the predicted action.
    action, _states = model.predict(observation)
    return action


def preprocess_observation(observation):
    # Preprocess the observation data before feeding it to the model.
    # Modify this function to suit the preprocessing needs of your specific model.
    return observation


def main(_):
    # Unused
    pass


if __name__ == "__main__":
    app.run(main)
