---
layout: "contents"
title: Benchmarking Inference
firstpage:
---

# Benchmarking Inference

## Prerequisites

Before you begin, ensure you have done the following:

### Install A2Perf

For detailed instructions, please refer to
our [Installation Guide](../basic_usage.md#Installation).

### Benchmarking Training Tutorial

Please refer to the [Benchmarking Training Tutorial](training.md) for
instructions on how to train your agent. We will use the artifacts generated
from the training tutorial for this inference tutorial.

### Update the `a2perf_benchmark_submission` Submodule

If you have not already done so for the training tutorial, update
the `a2perf_benchmark_submission` submodule to the `baselines-local` branch:

```bash
cd a2perf/a2perf_benchmark_submission
git fetch origin
git checkout baselines-local
git pull origin baselines-local
cd ../..
```

## Running the Inference Benchmark

After running the training benchmark, you will have a directory with the trained
agent and other artifacts. We will use these for the inference benchmark.

### Running locally with XManager (Docker)

#### Running the Benchmark

```bash
xmanager launch xm_launch.py -- \
  --experiment-name=test_inference \
  --root-dir=~/gcs/a2perf/experiments/ \
  --experiment-id=<experiment-id> \
  --domain=QuadrupedLocomotion-DogPace-v0  \
  --submission-gin-config-path=a2perf/submission/configs/quadruped_locomotion/dog_pace/inference.gin \
  --user=$USER \
  --participant-module-path=a2perf/a2perf_benchmark_submission \
  --participant-args="root_dir=/experiment_dir,policy_name=greedy_policy"
```

#### Command line arguments

- **`root-dir`**: Specifies the directory where experiment logs and artifacts
  will be saved.
- **`experiment-id`**: The ID of the training experiment from which to load the
  trained agent.
- **`submission-gin-config-path`**: Points to the Gin configuration file for
  inference in the Dog Pace environment.
- **`participant-module-path`**: Indicates the path to the directory containing
  the submission code.
- **`participant-args`**: Provides additional arguments for the participant's
  code, including the path to the trained agent and the policy name to use.

XManager will automatically launch a Docker container with the necessary
dependencies installed. It will create a new experiment directory for the
inference results.

### Running Locally Without Docker

If you prefer to run the benchmark locally without using Docker, follow these
steps:

#### Installing Dependencies

If you have not already done so for the training tutorial, install the required
Python dependencies:

```bash
pip install -r A2Perf/a2perf/a2perf_benchmark_submission/requirements.txt
```

#### Running the Benchmark

Once the dependencies are installed, you can run the inference benchmark with
the following command:

```bash
cd A2Perf
export A2PERF_ROOT=$(pwd)
a2perf $A2PERF_ROOT/a2perf/a2perf_benchmark_submission \
  --root-dir=~/gcs/a2perf/experiments/<experiment-id>/test/1 \
  --submission-gin-config-path=$A2PERF_ROOT/a2perf/submission/configs/quadruped_locomotion/dog_pace/inference.gin \
  --participant-args="root_dir=~/gcs/a2perf/experiments/<experiment-id>/test/1,policy_name=greedy_policy"
```

Note: Replace `<experiment-id>` with the actual ID of your training experiment.
This ID is unique for each run and can be found in the output of your training
command or in the experiment directory structure.

#### Command line arguments

The command line arguments are similar to those used in the Docker version, but
adapted for local execution:

- **`root-dir`**: Specifies the directory where the training artifacts are
  located and where inference results will be saved.
- **`submission-gin-config-path`**: Points to the Gin configuration file for
  inference in the Dog Pace environment.
- **`participant-module-path`**: Indicates the path to the directory containing
  the submission code.
- **`participant-args`**: Provides additional arguments for the participant's
  code, including the path to the trained agent and the policy name to use.

Make sure to adjust the paths according to your setup if they differ from the
example provided.

After running the inference benchmark, you will find the results in the
specified
root directory. These results will include metrics on the agent's performance
during inference, such as average returns, inference time, and resource usage.
