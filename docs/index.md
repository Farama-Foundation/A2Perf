---
hide-toc: false
firstpage:
lastpage:
---

```{project-logo} _static/img/logo/text/A2Perf-text.png
:alt: A2Perf Logo
```

```{project-heading}
A2Perf is a benchmarking suite for evaluating agents on sequential decision-making problems that are relevant to the real world.
```

```{figure} _static/REPLACE_ME.gif
   :alt: REPLACE ME
   :width: 500
```

This library contains a collection of environments from domains spanning
computer chip-floorplanning, web navigation, and quadruped locomotion.

The Gymnasium interface allows users to initialize and interact with the A2Perf
environments as follows:

```{code-block} python
import gymnasium as gym
from a2perf.domains import circuit_training
# from a2perf.domains import web_navigation
# from a2perf.domains import quadruped_locomotion

# Choose one of the A2Perf environments
env = gym.make("CircuitTraining-Ariane-v0")
# or env = gym.make("WebNavigation-Difficulty-01-v0")
# or env = gym.make("QuadrupedLocomotion-DogPace-v0")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # Replace with your policy
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

```{toctree}
:hidden:
:caption: Introduction

content/basic_usage
content/publications

```

```{toctree}
:hidden:
:caption: Environments

content/circuit_training/index
content/quadruped_locomotion/index
content/web_navigation/index

```

```{toctree}
:hidden:
:caption: Tutorials

content/tutorials/training
content/tutorials/inference
content/tutorials/generalization
content/tutorials/add_domain
```

```{toctree}
:hidden:
:caption: Development

release_notes
Github <https://github.com/Farama-Foundation/A2Perf>
```
