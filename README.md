# A2Perf: Benchmark for Autonomous Agents.

A2Perf is a benchmark for evaluating agents on sequential decision problems that are relevant to the real world. This
repository contains code for running and evaluating participant's submissions on the benchmark platform.

## Environments

A2Perf provides benchmark environments in the following domains:

* [Web Navigation](tutorials/WebNav.ipynb) - This environment facilitates the creation of compositional tasks represented by dependency graphs, where automatically generated websites are completed by the trained agent.
* [Quadruped Locomotion](tutorials/QuadrupedLocomotion.ipynb) - This quadruped locomotion environment aims to teach a legged robot with 18 degrees of freedom to replicate animal-like behaviors by imitating real-world motion data to develop a diverse repertoire of skills.
* [Circuit Training](tutorials/CircuitTraining.ipynb) - Chip floorplanning, a complex and traditionally manual process, has been addressed by Google's open-source Circuit Training framework, which uses reinforcement learning to optimize chip layouts for multiple objectives.


<!-- 
### Web Navigation

![Three web navigation environments](media/gminiwob_scene.png)

### Quadruped Locomotion

![Simulated quadrupeds](media/locomotion_scene.png)

### Chip Floorplanning

![Chip floorplanning environment](media/ariane_scene.png) -->

## Installation

A2Perf can be installed on your local machine:

```bash
git clone https://github.com/Farama-Foundation/A2Perf.git
cd A2Perf
git submodule sync --recursive
git submodule update --init --recursive
pip install -e .
pip install -r requirements.txt
```
Both x86-64 and Arch64 (ARM64) architectures are supported.
\
Please note that the Windows version is not as well-tested as Linux and macOS versions.
It can be used for development and testing but if you want to conduct serious (time and resource-extensive) experiments on Windows,
please consider using [Docker](https://docs.docker.com/docker-for-windows/install/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) with Linux version.




## API

Environments in A2Perf are registered under the names `WebNavigation-v0`, `QuadrupedLocomotion-v0`,
and `CircuitTraining-v0`. For example, you can create an instance of the `WebNavigation-v0` environment as follows:

```python
from a2perf.domains.web_nav import web_nav
import gymnasium as gym

env = gym.make('WebNavigation-v0', difficulty=1, seed=0)
```



## Participant Submission
A beginners guide to benchmarking with A2Perf is described [here](tutorials/beginners_guide.ipynb). 

- Participants can pull the template repository at https://github.com/harvard-edge/A2Perf_benchmark_submission
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



## Baselines

Baselines for all tasks are provided and are described in the article supporting A2Perf.

## Environment Versioning

A2Perf keeps strict versioning for reproducibility reasons. All environments end in a suffix like "-v0".  When changes are made to environments that might impact learning results, the number is increased by one to prevent potential confusion. This is follows the Gymnasium conventions.

## Credits

This package is maintained by Ikechukwu Uchendu and Jason Jabbour.

## Citation
You can cite A2Perf as:
\
TODO
```
@misc{ADD CITATION,
}
```