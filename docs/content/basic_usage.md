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

env = gym.make('CircuitTraining-Ariane-v0',
               netlist_file='path/to/netlist.pb.txt')
```

## Web Navigation

```python
import gymnasium as gym

env = gym.make('WebNavigation-Difficulty-01-v0', difficulty=1, num_websites=1)
```

## Quadruped Locomotion

```python
import gymnasium as gym

env = gym.make('QuadrupedLocomotion-DogPace-v0')
# Other available environments:
# env = gym.make('QuadrupedLocomotion-DogTrot-v0')
# env = gym.make('QuadrupedLocomotion-DogSpin-v0')
```


For more detailed usage

# Installation

To install A2Perf, the easiest way is to use `pip`. You can install specific
domains or all domains depending on your needs:

```bash
# Install all domains
pip install a2perf[all]

# Install specific domains
pip install a2perf[circuit-training]
pip install a2perf[web-navigation]
pip install a2perf[quadruped-locomotion]
```

## Installing from source

If you would like to install A2Perf from source, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Farama-Foundation/A2Perf.git
cd A2Perf
git submodule update --init --recursive
pip install .
```

If you want to install the package in development mode, use the `-e` flag:

```bash
pip install -e .
```
