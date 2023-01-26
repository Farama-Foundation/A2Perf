# RLPerf: Benchmark for Autonomous Agents.

RLPerf is a benchmark for evaluating agents on sequential decision problems that are relevant to the real world. This
repository contains code for running and evaluating participant's submissions on the benchmark platform.

## Installation

RLPerf can be installed on your local machine:

```bash
git clone https://github.com/harvard-edge/rl-perf.git
cd rl-perf
git submodule init --update --recursive
pip install -e .
```

---

## Domains

RLPerf provides benchmark environments in the following domains:

### Web Navigation

![Three web navigation environments](media/gminiwob_scene.png)

### Quadruped Locomotion

![Simulated quadrupeds](media/locomotion_scene.png)

### Chip Floorplanning

![Chip floorplanning environment](media/ariane_scene.png)

## Gym Interface

Environments in RLPerf are registered under the names `WebNavigation-v0`, `QuadrupedLocomotion-v0`,
and `CircuitTraining-v0`. For example, you can create an instance of the `WebNavigation-v0` environment as follows:

```python
from rl_perf.domains.web_nav import web_nav
import gym

env = gym.make('WebNavigation-v0', difficulty=1, seed=0)
```

---

## Participant Submission

- Participants can pull the template repository at https://github.com/harvard-edge/rlperf_benchmark_submission
    - The submission repository must include:
        - `train.py` - defines a train function that can train the participant's model from scratch
        - `train.gin` - defines configurations for how the participant wants their solution to be benchmarked. For
          example, thhe participant could include values for these contants

```python
@gin.constants_from_enum
class BenchmarkMode(enum.Enum):
    TRAIN = 'train'
    EVAL = 'eval'
    INFERENCE = 'inference'


@gin.constants_from_enum
class BenchmarkDomain(enum.Enum):
    WEB_NAVIGATION = 'web_navigation'
    CIRCUIT_TRAINING = 'circuit_training'
    QUADRUPED_LOCOMOTION = 'quadruped_locomotion'


@gin.constants_from_enum
class ReliabilityMetrics(enum.Enum):
    IqrWithinRuns = 'IqrWithinRuns'
    IqrAcrossRuns = 'IqrAcrossRuns'
    LowerCVaROnDiffs = 'LowerCVaROnDiffs'
    LowerCVaROnDrawdown = 'LowerCVaROnDrawdown'
    LowerCVarOnAcross = 'LowerCVarOnAcross'
    MedianPerfDuringTraining = 'MedianPerfDuringTraining'

```

---

## Baselines

TODO

## Credits

This package is maintained by Ikechukwu Uchendu and Jason Jabbour.