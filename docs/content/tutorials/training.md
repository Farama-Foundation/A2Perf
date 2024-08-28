# Benchmarking Training

## Prerequisites

Before you begin, ensure you have done the following:

### Install A2Perf

For detailed instructions, please refer to
our [Installation Guide](../basic_usage.md#Installation).

### Proper Submission folder structure

Your submission should be placed in
the `A2Perf/a2perf/a2perf_benchmark_submission/` directory. The structure should
look like this:

```
A2Perf/
├── a2perf/
│   ├── a2perf_benchmark_submission/
│   │   ├── init.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── requirements.txt
│   │   └── [your supporting files and directories]
```

To get started quickly, you can use our template repository as a starting point.
You can find the template at:

[https://github.com/Farama-Foundation/a2perf-benchmark-submission/tree/template](https://github.com/Farama-Foundation/a2perf-benchmark-submission/tree/template)

This template provides the basic structure and files needed for your submission.
You can clone this repository and modify it to fit your specific implementation.

#### Explanation of files:

- `__init__.py`:
  The `__init__.py` file can be an empty file.

- `train.py`:
  The `train.py` file includes the function `train()`, which A2Perf calls for
  the training of your algorithm.


- `inference.py`
  Next, the `inference.py` file is subsequently used for benchmarking the
  trained model.
  This file includes several key functions.\
  \
  __`load_model(env)`:__
  This function loads and returns the trained model. A2Perf passes the
  environment that is being tested via the `env` parameter. This allows the
  model loading logic to use any context needed, such as the environment name.
  \
  __`preprocess_observation(observation)`:__
  Preprocesses the observation before feeding it to the model. If no
  preprocessing is required, simply return the initial observation.
  \
  __`infer_once(model, observation)`:__
  Passes a single observation to the loaded model and returns the predicted
  action. This function performs a single inference step.

- `requirements.txt`:
  Specifies any package dependencies required to run the submission code. This
  file may include version constraints for the dependencies. Having an explicit
  requirements file ensures a consistent environment for evaluation.

---

## Update the `a2perf_benchmark_submission` Submodule

For the purposes of this tutorial, we will use the repository of our local
baselines as the "submission." To use this, please update
the `a2perf_benchmark_submission` submodule to the following
branch: `baselines-local`

### Navigate to the Submodule Directory

   ```bash
   cd a2perf/a2perf_benchmark_submission
   ```

### Checkout the branch with code for baselines

   ```bash
   git fetch origin
   git checkout baselines-local
   ```

### Pull Latest Changes

    ```bash
    git pull origin baselines-local
    ```

### Back to the Main Directory

Return to the main directory of the `A2Perf` repository:

      ```bash
      cd ../../..
      ```

---

## Running the Training Benchmark

### Running locally with XManager (Docker)

#### Running the Benchmark

```bash
xmanager launch xm_launch.py -- \
  --experiment-name=test \ 
  --root-dir=~/gcs/a2perf/experiments/ \
  --domain=QuadrupedLocomotion-DogPace-v0  \
  --submission-gin-config-path=a2perf/submission/configs/quadruped_locomotion/train.gin \
  --user=$USER \
  --participant-module-path=a2perf/a2perf_benchmark_submission \
  --participant-args="gin_config_path=configs/quadruped_locomotion/dog_pace/ppo.gin"
```

#### Command line arguments

- **`root-dir`**: Specifies the directory where experiment logs and artifacts
  will be saved.
- **`gin-config`**: Points to the Gin configuration file for the **Dog Pace**
  environment in the Quadruped Locomotion domain.
- **`participant-module-path`**: Indicates the path to the directory containing
  the submission code. Adjust this path to point to
  your `a2perf_benchmark_submission` directory.
- **`participant-args`**: Provides additional arguments for the participant's
  code, including the path to the algorithm-specific Gin configuration file.

[XManager](https://github.com/google-deepmind/xmanager) will automatically
launch a Docker container with the necessary dependencies installed. It will
also create a new experiment directory
at `~/gcs/a2perf/experiments/<experiment-number>/test/1/`. The number `1` is
included because we are running a single work unit in the experiment. For more
details on work units, refer
to [XManager's documentation](https://github.com/google-deepmind/xmanager).

The experiment directory will contain all logs and artifacts generated during
the benchmark. Here is how the directory structure will look at the end of the
training:

```plaintext
~/gcs/a2perf/experiments/1724700456099
└── test
    └── 1
        ├── collect
        ├── metrics
        ├── policies
        ├── submission_config.gin
        └── train
```

- **`collect/`**: Contains TensorBoard summaries for each of the separate
  collection policies. Each subdirectory (e.g., `actor_0`, `actor_1`, ...)
  corresponds to a different actor's summary data.

  Example structure:
  ```plaintext
  collect/
  ├── actor_0
  │   └── summaries
  │       └── 0
  ├── actor_1
  │   └── summaries
  │       └── 1
  └── ...
  ```

- **`metrics/`**: Stores system metrics collected during training, such
  as `train_emissions.csv`, which logs energy consumption and emissions data.

- **`policies/`**: Contains the saved policies generated during training. This
  includes different policies such as `collect_policy`, `greedy_policy`, and the
  main `policy`. Each subdirectory represents a different policy and contains
  the necessary files for TensorFlow models, including saved models and
  variables.

  Example structure:
  ```plaintext
  policies/
  ├── collect_policy
  ├── greedy_policy
  └── policy

- **`train/`**: Contains additional checkpoint information and TensorBoard logs
  from the training process, which are useful for monitoring training progress
  and debugging.

### Running Locally Without Docker

If you prefer to run the benchmark locally without using Docker, follow these
steps:

#### Installing Dependencies

To run the benchmark locally, you need to manually install the required Python
dependencies. Run the following command:

```bash
pip install -r A2Perf/a2perf/a2perf_benchmark_submission/requirements.txt
```

#### Running the Benchmark

Once the dependencies are installed, you can run the benchmark for the Quadruped
Locomotion - Dog Pace environment with the following command:

```bash
cd A2perf
export A2PERF_ROOT=$(pwd)
python a2perf/launch/entrypoint.py \
  --root-dir=~/gcs/a2perf/experiments/test_without_docker \
  --submission-gin-config-path=$A2PERF_ROOT/a2perf/submission/configs/quadruped_locomotion/train.gin \
  --participant-module-path=$A2PERF_ROOT/a2perf/a2perf_benchmark_submission \
  --participant-args="gin_config_path=configs/quadruped_locomotion/dog_pace/ppo.gin"

```

#### Command line arguments

- **`root-dir`**: Specifies the directory where experiment logs and artifacts
  will be saved.
- **`gin-config`**: Points to the Gin configuration file for the **Dog Pace**
  environment in the Quadruped Locomotion domain.
- **`participant-module-path`**: Indicates the path to the directory containing
  the submission code. Adjust this path to point to
  your `a2perf_benchmark_submission` directory.
- **`participant-args`**: Provides additional arguments for the participant's
  code, including the path to the algorithm-specific Gin configuration file.

Make sure to adjust the paths according to your setup if they differ from the
example provided.
