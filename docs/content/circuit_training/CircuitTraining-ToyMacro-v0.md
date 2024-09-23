# Toy Macro Standard Cell

![The Toy Macro Standard Cell  CPU](../../_static/img/CircuitTraining-ToyMacro-v0.gif)

## Environment Creation

```python
from a2perf.domains import circuit_training
import gymnasium as gym

env = gym.make('CircuitTraining-ToyMacro-v0')
```

#### Optional parameters:

| Parameter                  | Type                  | Default                                                | Description                                                                                                              |
|----------------------------|-----------------------|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `netlist_file`             | str                   | path to `netlist.pb.txt`                               | Path to the input netlist file. Predefined by using `Ariane` or `ToyMacro`.                                              |
| `init_placement`           | str                   | path to `initial.plc`                                  | Path to the input initial placement file, used to read grid and canvas size. Predefined by using `Ariane` or `ToyMacro`. |
| `plc_wrapper_main`         | str                   | `a2perf/domains/circuit_training/bin/plc_wrapper_main` | Main PLC wrapper.                                                                                                        |
| `create_placement_cost_fn` | Callable              | `placement_util.create_placement_cost`                 | A function that creates the `PlacementCost` object given the netlist and initial placement file.                         |
| `std_cell_placer_mode`     | str                   | `'fd'`                                                 | Options for fast standard cells placement. The `fd` option uses the force-directed algorithm.                            |
| `cost_info_fn`             | Callable              | `cost_info_function`                                   | The cost function that, given the `plc` object, returns the RL cost.                                                     |
| `global_seed`              | int                   | `0`                                                    | Global seed for initializing environment features, ensuring consistency across actors.                                   |
| `netlist_index`            | int                   | `0`                                                    | Netlist index in the model static features.                                                                              |
| `is_eval`                  | bool                  | `False`                                                | If set, saves the final placement in `output_dir`.                                                                       |
| `save_best_cost`           | bool                  | `False`                                                | If set, saves the placement if its cost is better than the previously saved placement.                                   |
| `output_plc_file`          | str                   | `''`                                                   | The path to save the final placement.                                                                                    |
| `cd_finetune`              | bool                  | `False`                                                | If True, runs coordinate descent to fine-tune macro orientations. Meant for evaluation, not training.                    |
| `cd_plc_file`              | str                   | `'ppo_cd_placement.plc'`                               | Name of the coordinate descent fine-tuned `plc` file, saved in the same directory as `output_plc_file`.                  |
| `train_step`               | Optional[tf.Variable] | `None`                                                 | A `tf.Variable` indicating the training step, used for saving `plc` files during evaluation.                             |
| `output_all_features`      | bool                  | `False`                                                | If true, outputs all observation features. Otherwise, only outputs dynamic observations.                                 |
| `node_order`               | str                   | `'descending_size_macro_first'`                        | The sequence order of nodes placed by RL.                                                                                |
| `save_snapshot`            | bool                  | `True`                                                 | If true, saves the snapshot placement.                                                                                   |
| `save_partial_placement`   | bool                  | `False`                                                | If true, evaluation also saves the placement even if RL does not place all nodes when an episode is done.                |
| `use_legacy_reset`         | bool                  | `False`                                                | If true, uses the legacy reset method.                                                                                   |
| `use_legacy_step`          | bool                  | `False`                                                | If true, uses the legacy step method.                                                                                    |
| `render_mode`              | str                   | `None`                                                 | Specifies the rendering mode `human` or `rgb_array`, if any.                                                             |

## Description

Circuit Training is an open-source framework for generating chip floor plans
with distributed deep reinforcement learning. This framework reproduces the
methodology published in the Nature 2021 paper:

A graph placement methodology for fast chip design. Azalia Mirhoseini, Anna
Goldie, Mustafa Yazgan, Joe Wenjie Jiang, Ebrahim Songhori, Shen Wang,
Young-Joon Lee, Eric Johnson, Omkar Pathak, Azade Nazi, Jiwoo Pak, Andy Tong,
Kavya Srinivasa, William Hang, Emre Tuncer, Quoc V. Le, James Laudon, Richard
Ho, Roger Carpenter & Jeff Dean, 2021. Nature, 594(7862), pp.207-212. [PDF]

At each timestep, the agent must place a single macro onto the chip canvas.

**Note**: this environment is only supported on Linux based OSes.

## Action Space

Circuit Training represents the chip canvas as a grid.
The action space corresponds to the different locations that the next macro can
be placed onto the canvas without violating any hard constraints on density or
blockages.
At each step, the agent places a macro. Once all macros are placed, a
force-directed method is used to place clusters of standard cells.

## Observation Space

The observation space encodes information about the partial placement of the
circuit.
This includes:

- `current_node`: the current node to be placed, which is a single integer
  ranging from 0 to 3499.
- `fake_net_heatmap`: a fake net heatmap, which provides a continuous
  representation of the heatmap with values between 0.0 and 1.0 across 16,384
  points.
- `is_node_placed`: the placement status of nodes, a binary array of size 3500,
  showing whether each node has been placed (1) or not (0).
- `locations_x`: node locations in the x-axis, a continuous array of size 3500
  with values ranging from 0.0 to 1.0, representing the x-coordinates of the
  nodes.
- `locations_y`: node locations in the y-axis, similar to locations_x, but for
  the y-coordinates.
- `mask`: a mask, a binary array of size 16,384 indicating the validity or
  usability of each point in the net heatmap.
- `netlist_index`: a netlist index. This usually acts as a placeholder, and is
  fixed at 0.

## Rewards

The reward is evaluated at the end of each episode. The placement cost binary is
used to calculate the reward based on proxy wirelength, congestion, and density.
An infeasible placement results in a reward of -1.0.

The reward function is defined as:

$$R(p, g) = -\text{Wirelength}(p, g) - \lambda \cdot \text{Congestion}(p, g) - \gamma \cdot \text{Density}(p, g)$$

Where:

- $p$ represents the placement
- $g$ represents the netlist graph
- $\lambda$ is the congestion weight
- $\gamma$ is the density weight

Default values in A2Perf:

- The congestion weight $\lambda$ is set to 0.01
- The density weight $\gamma$ is set to 0.01
- The maximum density threshold is set to 0.6

These default values are based on the methodology described
in [Mirhoseini et al. (2021)][1].

[1]: https://www.nature.com/articles/s41586-021-03544-w "A graph placement methodology for fast chip design"

## Episode End

The episode ends when all nodes have been placed.

## Termination

The episode is terminated once all macros have been placed on the canvas, then
the final reward is calculated.

## Registered Configurations

* `CircuitTraining-ToyMacro-v0`
