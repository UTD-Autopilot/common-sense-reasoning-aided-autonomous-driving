import time
from autopilot.environment.carla import CarlaEnvironment
from autopilot.environment.carla import CarlaRecorder
from autopilot.agent import AutonomousVehicle

env = CarlaEnvironment(carla_host='127.0.0.1', carla_port=2000, carla_timeout=5.0)
recorder = CarlaRecorder(env)
agent_vehicle = env.spawn_agent_vehicle()
agent = AutonomousVehicle(env=env, vehicle=agent_vehicle)

# Let the vehicle drive for 1 seconds. Note that this is 1 real-world second, there's no guarantee how long the simulation will actually run!
time.sleep(1)

# Spawn 10 npc vehicle and 0 pedestians
env.spawn_traffic(10, 0)

# You can add custom callback
env.add_tick_callback(lambda: print(agent_vehicle.get_transform()))

# Start the recorder
print("Record started.")
recorder.start_record(save_path="../tmp/record_test")

time.sleep(30) # Run the program for 30 seconds.

recorder.stop_record()
env.close()
