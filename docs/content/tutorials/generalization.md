---
layout: "contents"
title: Benchmarking Generalization
firstpage:
---

# Benchmarking Generalization

Generalization in A2Perf tests how well your trained agent performs on tasks it
wasn't specifically trained for. This helps evaluate the robustness and
adaptability of your agent across different scenarios within the same domain.

## Prerequisites

Before you begin, ensure you have completed the following:

1. Install A2Perf (refer to
   the [Installation Guide](../basic_usage.md#installation))
2. Complete the [Benchmarking Training Tutorial](training.md)
3. Complete the [Benchmarking Inference Tutorial](inference.md)

## Running the Generalization Benchmark

After running the training and inference benchmarks, you can proceed with the
generalization benchmark using the trained agent.

### Running Locally with XManager (Docker)

```bash
xmanager launch xm_launch.py -- \
  --experiment-name=test  \
  --root-dir=~/gcs/a2perf/experiments/ \
  --experiment-id=<experiment-id> \
  --domain=QuadrupedLocomotion-DogPace-v0  \
  --submission-gin-config-path=a2perf/submission/configs/quadruped_locomotion/generalization.gin \
  --user="$USER" \
  --participant-module-path=a2perf/a2perf_benchmark_submission \
  --participant-args="root_dir=/experiment_dir,policy_name=greedy_policy" \
  --num-gpus=0
```

Replace `<experiment-id>` with the actual ID of your training experiment.

#### Command Line Arguments

- `--experiment-name`: Sets a name for the generalization experiment.
- `--root-dir`: Specifies the directory where experiment logs and artifacts will
  be saved.
- `--experiment-id`: The ID of the training experiment from which to load the
  trained agent.
- `--domain`: Specifies the domain for the generalization benchmark.
- `--submission-gin-config-path`: Points to the Gin configuration file for
  generalization in the Quadruped Locomotion domain.
- `--user`: Sets the user running the experiment.
- `--participant-module-path`: Indicates the path to the directory containing
  the submission code.
- `--participant-args`: Provides additional arguments for the participant's
  code, including the path to the trained agent and the policy name to use.
- `--num-gpus`: Specifies the number of GPUs to use (set to 0 for CPU-only
  execution).

### Running Locally Without Docker

```bash
CUDA_VISIBLE_DEVICES=-1 a2perf ../a2perf_benchmark_submission \
  --root-dir=~/gcs/a2perf/experiments/<experiment-id>/test/1 \
  --submission-gin-config-path=../submission/configs/quadruped_locomotion/generalization.gin \
  --participant-args="root_dir=~/gcs/a2perf/experiments/<experiment-id>/test/1,policy_name=greedy_policy"
```

Replace `<experiment-id>` with the actual ID of your training experiment.

#### Command Line Arguments

- `CUDA_VISIBLE_DEVICES=-1`: This flag runs the benchmark on CPU. You can remove
  it to use GPU for faster execution, but be aware that loading multiple
  policies might exceed your GPU memory.
- `--root-dir`: Specifies the directory where the training artifacts are located
  and where generalization results will be saved.
- `--submission-gin-config-path`: Points to the Gin configuration file for
  generalization in the Quadruped Locomotion domain.
- `--participant-module-path`: Indicates the path to the directory containing
  the submission code.
- `--participant-args`: Provides additional arguments for the participant's
  code, including the path to the trained agent and the policy name to use.

## Generalization Configuration

The generalization benchmark uses a specific Gin configuration file ([`A2Perf/a2perf/submission/configs/quadruped_locomotion/generalization.gin`](../../../a2perf/submission/configs/quadruped_locomotion/generalization.gin)) to define the tasks and parameters. Here's a preview of its contents:

```python
# ----------------------
# IMPORTS
# ----------------------
import a2perf.submission.submission_util
import a2perf.domains.tfa.suite_gym

# ----------------------
# SUBMISSION SETUP
# ----------------------
Submission.mode = %BenchmarkMode.GENERALIZATION
Submission.domain = %BenchmarkDomain.QUADRUPED_LOCOMOTION
Submission.run_offline_metrics_only = False
Submission.measure_emissions = False

# ----------------------
# GENERALIZATION ENVIRONMENT PARAMETERS
# ----------------------
Submission.generalization_tasks = ['dog_pace', 'dog_trot', 'dog_spin']
Submission.num_generalization_episodes = 100
```

This configuration sets up the generalization benchmark to test your agent on
three different tasks: dog pace, dog trot, and dog spin. Each task will be run
for 100 episodes.
