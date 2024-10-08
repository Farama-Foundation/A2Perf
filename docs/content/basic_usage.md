---
layout: "contents"
title: Basic Usage
firstpage:
---

# Basic Usage

After installing A2Perf, you can easily instantiate environments from different
domains. Here are some examples:

## Circuit Training

```python
import gymnasium as gym

env = gym.make('CircuitTraining-Ariane-v0')
# Other available environments:
# env = gym.make('CircuitTraining-ToyMacro-v0')
```

## Web Navigation

```python
import gymnasium as gym

env = gym.make('WebNavigation-Difficulty-01-v0', num_websites=1)
```

## Quadruped Locomotion

```python
import gymnasium as gym

env = gym.make('QuadrupedLocomotion-DogPace-v0')
# Other available environments:
# env = gym.make('QuadrupedLocomotion-DogTrot-v0')
# env = gym.make('QuadrupedLocomotion-DogSpin-v0')
```

# Installation

## Installing from PyPI

To install A2Perf from PyPI, run the following command:

```bash
pip install a2perf
```

You can also install specific domains by specifying the package name:

```bash
pip install a2perf[web-navigation]
pip install a2perf[circuit-training]
pip install a2perf[quadruped-locomotion]
```

## Installing from source

**Note:** We highly recommend using Conda to manage your environment for
installing A2Perf, as it simplifies dependency management and ensures
compatibility across different systems.

To install A2Perf from source, follow these steps:

1. Clone the repository:

    ```bash
       git clone --recursive https://github.com/Farama-Foundation/A2Perf.git
      ```

   or

    ```bash
       git clone https://github.com/Farama-Foundation/A2Perf.git
       cd A2Perf
       git submodule update --init --recursive
    ```

2. Install the package:

    ```bash
    # Install all domains
    pip install -e .[all]
    # Or install specific domains
    pip install -e .[circuit-training]
    pip install -e .[web-navigation]
    pip install -e .[quadruped-locomotion]
    ```

   If you do not need an editable installation, you can omit the `-e` flag:

    ```bash
    pip install .[all]
    ```
