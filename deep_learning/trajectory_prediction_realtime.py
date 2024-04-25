import torch
from torch import nn
from torchvision import transforms
# We reuse the CoverNet implemention in nuscenes
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import socket
from contextlib import closing
import traceback
import carla
import pickle
import os

###
#   Initialization
###

# Locations of required files
lattice_filepath = '../tmp/nuscenes-prediction-challenge-trajectory-sets/epsilon_4.pkl'
model_filepath = '../tmp/TrajectoryPrediction/resnet50_epsilon_4/model_23.pth'
backbone = ResNetBackbone('resnet50')
plot_top_k_trajectory = 1   # number of trajectory to plot

# BEV configuration. Must match the model training.
image_size_x = 500
image_size_y = 500
bev_range = 50.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Connect to and initialize Carla
carla_host = '127.0.0.1'
carla_port = 12000

carla_timeout = 30.0
carla_sync = True

tick_interval = 0.5
fixed_delta_seconds = 0.05
skip_frame_number = int(tick_interval / fixed_delta_seconds)
if skip_frame_number != tick_interval / fixed_delta_seconds:
    raise RuntimeError("tick_interval must be a multiple of fixed_delta_seconds")

print(f"Connecting to carla simulator on {carla_host}:{carla_port}")
client = carla.Client(carla_host, carla_port)
client.set_timeout(carla_timeout)

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
traffic_manager_port = find_free_port()
traffic_manager = client.get_trafficmanager(traffic_manager_port)

# Load map
client.load_world("Town06")

world = client.get_world()

# Apply simulation settings
original_settings = world.get_settings()
world.unload_map_layer(carla.MapLayer.Foliage)

if carla_sync:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)

# Set traffic light interval
for tl in world.get_actors().filter('traffic.traffic_light*'):
    tl.set_green_time(20)
    tl.set_yellow_time(3)
    tl.set_red_time(10)

# Spawn agent vehicle
agent_vehicle_model = 'charger_2020'
bp = world.get_blueprint_library().filter(agent_vehicle_model)[0]
bp.set_attribute('role_name', 'hero')   # Set hero tag

agent_vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
agent_control = carla.VehicleControl()
agent_vehicle.apply_control(agent_control)  # Attach custom control object

agent_vehicle.set_autopilot(True, traffic_manager_port) # Start autopilot with the traffic manager we created

# Attach BEV camera to the agent vehicle
bev_camera_transform = carla.Transform(carla.Location(x=0, y=0, z=bev_range), carla.Rotation(pitch=-90))
bev_camera_attributes = {
    'image_size_x': image_size_x,
    'image_size_y': image_size_x,
    'fov': 90,
}
camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
for key, value in bev_camera_attributes.items():
    camera_bp.set_attribute(key, str(value))
bev_camera = world.spawn_actor(camera_bp, bev_camera_transform, attach_to=agent_vehicle)

calibration = np.identity(3)
calibration[0, 2] = bev_camera_attributes['image_size_x'] / 2.0
calibration[1, 2] = bev_camera_attributes['image_size_y'] / 2.0
calibration[0, 0] = calibration[1, 1] = bev_camera_attributes['image_size_x'] / (
            2.0 * np.tan(bev_camera_attributes['fov'] * np.pi / 360.0))
bev_camera.calibration = calibration

def map_color(img, target, old, new):
    mask = np.all(img == old, axis=-1)
    target[mask, :] = np.array(new)
    return target

bev_camera_data = np.ones((500, 500, 3), dtype=np.uint8) * 255  # Holds the camera data

def save_rgb_image(image):
    global bev_camera_data
    try:
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        img = array.copy()

        bev_camera_data = img

    except Exception as e:
        print(e)

bev_camera.listen(save_rgb_image)

# Spawn some traffic

vehicles_list = []

def spawn_traffic(number_of_vehicles):
    blueprints = world.get_blueprint_library().filter("vehicle.*")

    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('t2')]
    blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
    blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

    blueprints_walkers = world.get_blueprint_library().filter("walker.pedestrian.*")
    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        print(f'requested {number_of_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points')
        number_of_vehicles = number_of_spawn_points

    spawn_actor = carla.command.SpawnActor
    set_autopilot = carla.command.SetAutopilot
    future_actor = carla.command.FutureActor

    batch = []

    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        else:
            blueprint.set_attribute('role_name', 'autopilot')

        batch.append(spawn_actor(blueprint, transform)
                        .then(set_autopilot(future_actor, True, traffic_manager_port)))

    for response in client.apply_batch_sync(batch, carla_sync):
        if response.error:
            print(response.error)
        else:
            vehicles_list.append(response.actor_id)

    all_vehicle_actors = world.get_actors(vehicles_list)
    for actor in all_vehicle_actors:
        traffic_manager.update_vehicle_lights(actor, True)

def despawn_traffic():
    print(f'Despawning traffic')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

spawn_traffic(50)

###
#   Data Processing Related Code
###

# Image preprocessor
preprocessor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def rotate_points(points, angel):
    rotated = np.zeros(points.shape, dtype=points.dtype)
    rotated[...,0] = np.cos(np.deg2rad(angel)) * points[...,0] - np.sin(np.deg2rad(angel)) * points[...,1]
    rotated[...,1] = np.sin(np.deg2rad(angel)) * points[...,0] + np.cos(np.deg2rad(angel)) * points[...,1]
    return rotated

def prepare_input(
    agent_image, agent_location, agent_rotation,
    vehicle_location, vehicle_rotation, vehicle_velocity,
    bev_range
):
    #agent_image = cv2.imread(record['bev_image_path'])
    #agent_image = cv2.cvtColor(agent_image, cv2.COLOR_BGR2RGB)
    agent_image = agent_image.astype(np.uint8)
    vehicle_pitch = vehicle_rotation[2]
    # print(agent_pitch)
    vehicle_velocity = rotate_points(np.array(vehicle_velocity), -vehicle_pitch)
    state_vector = vehicle_velocity
    width = agent_image.shape[1]
    height = agent_image.shape[0]
    assert agent_image.shape[0] == agent_image.shape[1]
    img_scale =  agent_image.shape[0] / (bev_range*2)
    
    image = agent_image
    # Rotate the image to standard rotation
    #agent_pitch = record['agent_rotation'][2]
    agent_pitch = agent_rotation[2] # Note that this is actually roll in Carla
    mat = cv2.getRotationMatrix2D((width/2, height/2), -agent_pitch, 1.0)
    image = cv2.warpAffine(src=agent_image, M=mat, dsize=(width, height))

    translation = np.array(vehicle_location) - np.array(agent_location)
    #translation = record['vehicle_location'] - record['agent_location']
    # Move the vehicle to the center
    translation_matrix = np.array([
        [1, 0, -translation[1]*img_scale],
        [0, 1, translation[0]*img_scale]
    ], dtype=np.float32)

    image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))
    
    # print(translation[0], translation[1])
    # print(translation[0]*img_scale, translation[1]*img_scale)

    # print(vehicle_pitch)
    mat = cv2.getRotationMatrix2D((width/2, height/2), vehicle_pitch, 1.0)
    image = cv2.warpAffine(src=image, M=mat, dsize=(width, height))

    if preprocessor is not None:
        image = preprocessor(image)

    image = torch.tensor(image, dtype=torch.float32)
    state_vector = torch.tensor(state_vector, dtype=torch.float32)

    return image, state_vector

###
#   Main Loop
###

# Load the model we trained

with open(lattice_filepath, 'rb') as f:
    lattice = pickle.load(f)
    # In our data the vehicle is heading to the x axis, but the lattice is assuming the vehicle
    # is heading toward y axis, so we need to rotate these -90 degrees.
    lattice = rotate_points(np.array(lattice), -90)

lattice = np.array(lattice)
model = CoverNet(backbone, num_modes=lattice.shape[0], input_shape=(3,224,224))
model.to(device)

criterion = ConstantLatticeLoss(lattice)

try:
    model.load_state_dict(torch.load(model_filepath))
except Exception as e:
    print(e)

model.eval()

# Create a matplotlib canvas
plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
plt.title("Trajectory Prediction")

try:
    while True:
        # If in synchronized mode, inform Carla simulator to proceed.
        # Otherwise, wait for the tick. Note that wait_for_tick can skip ticks if our program runs slower.
        if carla_sync:
            world.tick()
        else:
            world.wait_for_tick()
        
        # Clear the plot
        ax.clear()

        # Agent location information. We're getting the ground truth from carla.
        transform = agent_vehicle.get_transform()
        agent_location = [transform.location.x, transform.location.y, transform.location.z]
        agent_rotation = [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
        
        # For every vehicle that's within a certain range, we do trajectory prediction
        # IRL this should be given by a object detection model, but for now we just get it from carla.
        for vehicle in world.get_actors().filter('*vehicle*'):

            # if vehicle.id == agent_vehicle.id:
            #     continue
            dist = vehicle.get_transform().location.distance(agent_vehicle.get_transform().location)
            if dist > bev_range:
                continue

            transform = vehicle.get_transform()
            vehicle_location = [transform.location.x, transform.location.y, transform.location.z]
            vehicle_rotation = [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
            velocity = vehicle.get_velocity()
            vehicle_velocity = [velocity.x, velocity.y, velocity.z]
            # prepare the input for the model
            # these variables are globals that will be updated in the tick callback.
            image, state_vector = prepare_input(
                bev_camera_data, agent_location, agent_rotation,
                vehicle_location, vehicle_rotation, vehicle_velocity,
                bev_range
            )

            # Move to device and make it a batch
            #ax.imshow(image.numpy().transpose(1, 2, 0))
            image = image.to(device).unsqueeze(0)
            state_vector = state_vector.to(device).unsqueeze(0)

            with torch.no_grad():
                logits = model(image, state_vector)
                _, predictions = torch.topk(logits, plot_top_k_trajectory, 1)
                predictions = predictions.detach().cpu().numpy()[0]
            pred_trajectories = lattice[predictions]

            for trajectory in pred_trajectories:
                # Transform the trejectory to image space for plotting
                trajectory = rotate_points(trajectory, vehicle_rotation[2])
                
                trajectory[:, 0] = trajectory[:, 0] + vehicle_location[0] - agent_location[0]
                trajectory[:, 1] = trajectory[:, 1] + vehicle_location[1] - agent_location[1]

                trajectory = rotate_points(trajectory, -agent_rotation[2])

                trajectory_x = trajectory[:, 0] * (image_size_x/(bev_range*2)) + (image_size_x/2)
                trajectory_y = trajectory[:, 1] * (image_size_y/(bev_range*2)) + (image_size_y/2)
                ax.scatter(trajectory_y, (image_size_x - trajectory_x))
        
        # Update the plots.
        plt_img = ax.imshow(bev_camera_data)
        #plt_img.set_data(bev_camera_data)

        figure.canvas.draw()
        figure.canvas.flush_events()

except Exception as e:
    traceback.print_exc()
finally:
    if client is not None:
        client.apply_batch([carla.command.DestroyActor(agent_vehicle)])
        despawn_traffic()
        world.apply_settings(original_settings)
