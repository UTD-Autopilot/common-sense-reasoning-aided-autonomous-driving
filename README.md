# Common Sense Reasoning Aided Autonomous Driving

## Data Collection

Data collection related code is located in `data_collection` folder.

For NuScenes, we provide a Jupyter Notebook `reasoning_nuscenes_data_collection.ipynb` to convert the data to the data format we're using.

For Carla, we used the simulator to collect the data. To reproduce the data collection process, please follow the instructions in the [README.md](data_collection/carla/README.md) document in the `data_collection/carla` folder. Once you finished setup Carla simulator 0.9.13, 0.9.14 or 0.9.15 (note: windows version of 0.9.14 is not supported due to a bug in Carla, see [issue 6315](https://github.com/carla-simulator/carla/issues/6315) in Carla's github repo), you can run `run.py` on a computer with graphical user interface. In the carla window, select the town you want to collect the data, spawn at least 1 agent (ego) vehicle and spawn as many npc (obstacle) vehicles you want (usually 100-400 is a good number for town 1-10), then click record. You can also load a scenario from the GUI. Once you collected enough samples you can click stop record and move to another town. Note that sometime Carla simulator will crash when switching maps, in that case you'll need to restart both Carla simulator and the python script. You can also check the [recording_example.py](data_collection/carla/autopilot/scripts/recording_example.py) in script folder for some example code to automate the data collection process on a server without GUI.

## Deep Learning Models

Deep learning models used in the paper are located in the `deep_learning` folder. All the models are implemented with PyTorch. For the notebooks, simply modify the dataset path to the corresponding location and run the notebook, it will generate and write the predictions in the dataset folder, starts with `pred_` prefix.

For bev semantic segmentation, the command to train the lift-splat-shoot model is:

```shell
python train.py ./configs/train_nuscenes_lss_baseline.yaml -g 0 1 -l ./outputs_bin/nuscenes/lss --loss ce
python train.py ./configs/train_carla_lss_baseline.yaml -g 2 3 -l ./outputs_bin/carla/lss --loss ce
```

## Common Sense Reasoning

