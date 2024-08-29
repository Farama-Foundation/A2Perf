# Adding Custom Domains

This tutorial will demonstrate how to add additional domains to A2Perf.

## Path to A2Perf Domains

Domains are stored in `A2Perf/a2perf/domains`. Currently, you'll find the main
domains in this folder such as `circuit_training`, `quadruped_locomotion`, and
`web_navigation`. There are also additional modules such as `tfa` that is there
for interfacing with Tensorflow Agents.


---

## Creating a new domain

To add a new domain:

1. Create a new folder in the domains directory. For the purposes of this
   tutorial, we will create a new domain called "my_domain":

```bash
cd A2Perf/a2perf/domains
mkdir my_domain
```

2. Inside of `my_domain`, create the following files:

- `__init__.py`
- [any other files needed for your domain]

3. Inside of `__init__.py`, you will need to register your domain with
   gymnasium. Here is an example of how we do this for the web navigation
   domain:

```python
import gymnasium as gym
import pkg_resources

base_url_path = pkg_resources.resource_filename('a2perf.domains.web_navigation',
                                                'gwob')
base_url = f'file://{base_url_path}/'
data_dir = pkg_resources.resource_filename(
        'a2perf.domains.web_navigation.environment_generation', 'data')

gym.envs.register(
        id='WebNavigation-v0',
        entry_point=(
                'a2perf.domains.web_navigation.gwob.CoDE.environment:WebNavigationEnv'
        ),
        apply_api_compatibility=False,
        disable_env_checker=False,
        kwargs=dict(
                use_legacy_step=False,
                use_legacy_reset=False,
                data_dir=data_dir,
                base_url=base_url),
)
```

Note: The `base_url_path`, `base_url`, and `data_dir` are specific to web
navigation and are not necessary for your domain.


---

## Additional configurations

When adding a new domain, you may need to configure additional settings:

1. Docker configurations:
   In `a2perf/launch/docker_utils.py`, add entries for your new domain:

   a. In the `get_docker_instructions` function:
   ```python
   docker_instructions = {
       # ... existing entries ...
       BenchmarkDomain.MY_DOMAIN.value: common_setup + [
           # Add your domain-specific Docker instructions here
           # E.g., installing dependencies, setting up the environment
       ],
   }
   ```

   b. In the `get_entrypoint` function:
   ```python
   entrypoints = {
       # ... existing entries ...
       BenchmarkDomain.MY_DOMAIN.value: xm.CommandList([
           # Add your domain-specific entrypoint commands here
       ]),
   }

2. Update constants:
   In `a2perf/constants.py`, add your new domain to the `BenchmarkDomain` enum
   and update the `ENV_NAMES` dictionary:

   ```python
   import enum
   import gin

   @gin.constants_from_enum
   class BenchmarkDomain(enum.Enum):
       QUADRUPED_LOCOMOTION = "QuadrupedLocomotion"
       WEB_NAVIGATION = "WebNavigation"
       CIRCUIT_TRAINING = "CircuitTraining"
       MY_DOMAIN = "MyDomain"  # Add your new domain here

   # ... (other existing code)

   ENV_NAMES = {
       # ... (existing domains)
       BenchmarkDomain.MY_DOMAIN: [
           "MyDomain-Task1-v0",
           "MyDomain-Task2-v0",
           # Add all the environment names for your new domain
       ],
   }
   ```
