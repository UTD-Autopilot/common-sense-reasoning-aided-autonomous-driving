# Setup

## CARLA backend

### Install python package

If you build carla yourself, after successfully building the python API, the whl package will be output to `PythonAPI/carla/dist` folder.

The whl must be installed in the PythonAPI folder. It will search for dependencies folder in the current working directory.

```
cd PythonAPI
pip install carla-0.9.13-cp39-cp39-win_amd64.whl
```

Otherwise, you can use `pip install carla` to install the python package.
It seems that most things will work even if the version of the python package and the CARLA server is different.

### Launch CARLA Simulator

#### Pre-built binary

Binary:

Pre-built binary Carla can be obtained from [CARLA's github release page](https://github.com/carla-simulator/carla/releases).

Since they only release python packages for up to python 3.8 you'll have to use python 3.8.

Docker:

Docker images works only on Linux.

The official Docker image seems buggy. We'll need to build our own image to fix it.

Build the image and name (tag) it carla. (If you're on `pdml2`, then it's already there and you can skip this step.)

```
docker build -t carla ./docker/carla
```

And run it.

```
docker run --gpus 0 -d --rm -p 2000-2002:2000-2002/tcp carla /bin/bash ./CarlaUE4.sh -RenderOffScreen -vulkan -nosound -carla-rpc-port=2000
```

You can change the port mapping if the ports are already occupied by someone else, replace the port number like this:
Say we want to use port 12000:

```
docker run --gpus 0 -d --rm -p 12000-12002:12000-12002/tcp carla /bin/bash ./CarlaUE4.sh -RenderOffScreen -vulkan -nosound -carla-rpc-port=12000
```

You can also change the GPU to use like this: `--gpus 1`. Use `nvidia-smi` to see which GPU are free.

You can use `docker ps` to check the running docker containers and use `docker stop <CONTAINER_ID>` to stop it.
You can type the first several characters instead of the full container id, like: `docker stop 09b`.


Windows:

If you build CARLA yourself on windows, you'll need to activate the build environment used to compile it to run the Unreal Editor.
In x64 Native Tools Command Prompt for VS 2019:

```
make launch
```

### Connect to CARLA server

Run run.py, fill in the server address and port (default is 2000), click connect.
