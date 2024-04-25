# Autopilot

Autopilot development platform.

See [setup](docs/setup.md) for setting up backends.

## Development Guide

### Create virtual environments

Change directory into the repo:

```shell
cd UTD-Autopilot
```

Create a vritual environment:

```shell
virtualenv venv
```

Then activate it:

On Linux and MacOS:

```shell
source ./venv/bin/activate
```

On Windows Powershell:

```PowerShell
& ./venv/Scripts/Activate.ps1
```

On Windows CMD:

```cmd
venv\scripts\activate.bat
```

### Install the package in edit mode

This step is required if you want to write seperate scripts in the `scripts` folder (or anywhere else as long as the venv is activated).

```shell
pip install -e .
```

With this you can do `import autopilot` or `from autopilot import agent`.

### Install dependencies

Dependencies other than PyTorch and CARLA are in `requirements.txt`. Install them with:

```shell
pip install -r requirements.txt
```

We're trying to use the latest version of the libraries, so the version info is not embedded into `requirements.txt`. If something break, please let us know.

#### CARLA
If you build CARLA yourself, follow the instructions in CARLA. Otherwise, you can install the CARLA python library with pip:

```shell
pip install carla
```

#### PyTorch

Before installing PyTorch, if you have a NVIDIA GPU and want to make use of your GPU, download and install CUDA first. You can download CUDA from here: https://developer.nvidia.com/cuda-downloads

Then, select the correct combination in this page: https://pytorch.org/get-started/locally/

And run the command provided.

You can try to run this code to verify your installation:

```python
import torch
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)
```

### Install CARLA

See [setup](docs/setup.md).

### GUI for visualization

While there's no intent to development user interface for the autonomous driving syste there's a simple GUI to help debugging. It can visualize the sensor and model outputs.

Simply run `run.py` to start the GUI.

The GUI supports live code reloading. Anything in the `tick()` functions and callbacks will be refreshed. However it will not automatically reload classes so you'll have to recreate the class (or just restart the program) if you modified something in `__init__()` functions.

### Write seperate scripts

You can also choose to write seperate scripts for testing. Create a subdirectory in `scripts` folder and put your scripts there.

## Architecture

There're three main concepts in the system: Agent, Environment and Preception Modules.

Agent is the autonomous driving system we're building. It holds the preception modules and output control signal to the vehicle.

Environment provides interface to interact with the simulation environment. It also provide the interface to interact with the physical or simulated vehicle.

Preception modules are individual model that performs a certain sensing tasks. It fetchs the raw data from the sensors or the output of other preception modules and produce high-level information to the Agent.

Here's a very simple example on setting up a basic environment (it's in `scripts/test_carla.py`):

```python
import time
from autopilot.environment.carla import CarlaEnvironment
from autopilot.agent import AutonomousVehicle

env = CarlaEnvironment(carla_host='127.0.0.1', carla_port=2000)
agent_vehicle = env.spawn_agent_vehicle()
agent = AutonomousVehicle(env=env, vehicle=agent_vehicle)

time.sleep(1) # Let the vehicle drive for 1 seconds.

env.add_tick_callback(lambda: print(agent_vehicle.get_transform()))

time.sleep(1) # Run the program for 1 seconds.
env.close()
```

`CarlaEnvironment` class will automatically start a thread and process messages in the background. You can use `add_tick_callback` to schedule calls on this thread. The function provided will be called every tick (can be set with `tick_interval` when creating the class instance). Here we simply print the localization model output.

You can refer to the source code of the modules to see how you can build your own.
