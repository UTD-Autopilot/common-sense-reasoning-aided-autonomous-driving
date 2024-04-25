# Common Sense Reasoning Aided Autonomous Driving

## Data Collection

Data collection related code is located in `data_collection` folder.

For NuScenes, we provide a Jupyter Notebook `reasoning_nuscenes_data_collection.ipynb` to convert the data to the data format we're using.

For Carla, we used the simulator to collect the data. To reproduce the data collection process, please follow the instructions in the [README.md](data_collection/carla/README.md) document in the `data_collection/carla` folder. Once you finished setup Carla simulator 0.9.13, 0.9.14 or 0.9.15 (note: windows version of 0.9.14 is not supported due to a bug in Carla, see [issue 6315](https://github.com/carla-simulator/carla/issues/6315) in Carla's github repo), you can run `run.py` on a 
computer with graphical user interface. In the carla window, select the town you want to collect the data, spawn at least 1 agent (ego) vehicle and spawn as many npc (obstacle) vehicles you want (usually 100-400 is a good number for town 1-10), then click record. You can also load a scenario from the GUI. Once you collected enough samples you can click stop record and move to another town. Note that sometime Carla simulator will crash when switching maps, in that case you'll need to restart both Carla simulator and the python script. You can also check the [recording_example.py](data_collection/carla/autopilot/scripts/recording_example.py) in script folder for some example code to automate the data collection process on a server without GUI.

## Deep Learning Models

Deep learning models used in the paper are located in the `deep_learning` folder. All the models are implemented with PyTorch. For the notebooks, simply modify the dataset path to the corresponding location and run the notebook, it will generate and write the predictions in the dataset folder, starts with `pred_` prefix.

For bev semantic segmentation, the command to train the lift-splat-shoot model is:

```shell
python train.py ./configs/train_nuscenes_lss_baseline.yaml -g 0 1 -l ./outputs_bin/nuscenes/lss --loss ce
python train.py ./configs/train_carla_lss_baseline.yaml -g 2 3 -l ./outputs_bin/carla/lss --loss ce
```

## Common Sense Reasoning

Once the CARLA data is collected, the commonsense model can be run. Ensure that behavior_cluters.json, consisten_logic_rules.pl, predicates.py, spreadsheetC_lights.py, and spreadsheet_obstacle.py are all inside the CARLA data folder. The appropriate directory should look like "...CARLA_DATASET_NAME\TOWN_RECORDING\agents\0". 

Program versions used:
Python 3.10.11
SWI-Prolog 9.0.4

Python Packages used:
json
csv
numpy 1.24.3
math
xlsxwriter 3.1.6

First run predicate script. This will generate a list of prolog facts that represent the CARLA data. Use the script corresponding to the scenario you want to evaluate.
```shell
python predicates_lights.py

OR

python predicates_obstacles.py
```

Open SWI-Prolog or your choice of Prolog interpreter and load the "consistent_logic_rules_lights.pl" or "consistent_logic_rules_obstacle.pl" commonsense reasoning base and the newly generated knowledge "predicates.txt". You'll need a string that represents the filepath to a target text file 'output.txt' to generate the output. File path should be a prolog string such as '...CARLA_DATASET\\TOWN_RECORDING\\agents\\0\\output.txt'. Make sure the directory is the same as the rest of the commonsense programs. After that make the query:

```shell
writefacts(FILEPATH).
```

This will generate the commonsense output in 'output.txt'. Finally if you want to record the results, run the python spreadsheet script corresponding to the desired evaluation scenario. The spreadsheet will record the metrics presented in the paper along with the false/true negative/positive rates.
```shell
python spreadsheet_lights.py

OR

python spreadsheet_obstacle.py
```
