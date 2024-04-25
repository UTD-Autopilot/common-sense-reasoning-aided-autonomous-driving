{'frame': 3494, 'timestamp': 179.51944218948483, 'location': [137.78372192382812, 5.128137111663818, 0.0002869033778551966], 'rotation': [-0.004547119606286287, -0.026856303215026855, -166.76368713378906], 'current_action': 'Unknown'}
import carla
import numpy as np
import math
import traceback

def rpy_to_pyr(rot):
    return np.array(rot)[..., [1, 2, 0]]

def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)

def setup_obstacle_avoidance_town05(env):
    client = env.carla
    world = env.world
    traffic_manager = env.traffic_manager

    #client.load_world('Town05')

    for obstacle in env.obstacle_list:
        obstacle.destroy()
    env.obstacle_list.clear()

    managed_vehicles = set()

    def obstacle_avoidance_callback():
        try:
            debug = world.debug
            vehicles_to_manipulate = []
            for vehicle_id in env.vehicles_list:
                vehicles_to_manipulate.append(vehicle_id)
            for agent_vehicle in env.agent_vehicles:
                vehicles_to_manipulate.append(agent_vehicle.vehicle.id)
            
            for vehicle_id in managed_vehicles:
                vehicle = world.get_actor(vehicle_id)
                env.traffic_manager.auto_lane_change(vehicle, False)
                env.traffic_manager.vehicle_percentage_speed_difference(vehicle, 30)
            managed_vehicles.clear()

            for vehicle_id in vehicles_to_manipulate:
                vehicle = world.get_actor(vehicle_id)
                for obstacle in env.obstacle_list:
                    norm_target, d_angle = compute_magnitude_angle(obstacle.get_transform().location, vehicle.get_transform().location, vehicle.get_transform().rotation.yaw)
                    
                    try:
                        action = env.traffic_manager.get_next_action(vehicle)[0]
                    except Exception as e:
                        action = "Unknown"

                    if norm_target < 60.0 and d_angle < 5.0:
                        debug.draw_string(vehicle.get_transform().location, f'{action} {norm_target:.2f} {d_angle:.2f}', life_time=0.5)
                        env.traffic_manager.auto_lane_change(vehicle, False)
                        env.traffic_manager.vehicle_percentage_speed_difference(vehicle, 60) # Slow down
                        managed_vehicles.add(vehicle_id)

                    if norm_target < 40.0 and d_angle < 5.0:
                        if action == 'Unknown':
                            continue
                        #if action not in ['ChangeLaneLeft']:
                        env.traffic_manager.force_lane_change(vehicle, False) # True is change to right lane, false is left.

        except Exception as e:
            traceback.print_exc()

    obstacle_callback_qualname = obstacle_avoidance_callback.__qualname__

    env.tick_callbacks = [cb for cb in env.tick_callbacks if cb.__qualname__ != obstacle_callback_qualname]

    obstacle_location = [167.78, 10.00, 0.1]

    obstacle_rotation = [-0.01, -0.02, -166.76]

    bp = world.get_blueprint_library().filter("vehicle.audi.a2")[0]
    bp.set_attribute('role_name', 'hero')
    obstacle = world.spawn_actor(bp, carla.Transform(carla.Location(*obstacle_location), carla.Rotation(*rpy_to_pyr(obstacle_rotation))))

    control = carla.VehicleControl()
    control.throttle = 0.0
    control.brake = 1.0
    obstacle.apply_control(control)

    traffic_manager.global_percentage_speed_difference(35) # Make the vehicle slower otherwise they can't stop in time.
    traffic_manager.set_global_distance_to_leading_vehicle(20.0)

    for tl in world.get_actors().filter('traffic.traffic_light*'):
        tl.set_green_time(30)
        tl.set_yellow_time(3)
        tl.set_red_time(15)

    env.obstacle_list.append(obstacle)
    env.add_tick_callback(obstacle_avoidance_callback)

    # Ego location
    ego_location = [57.78, 5.12, 0.1]
    ego_rotation = [-0.01, -0.02, 0.00]

    env.scenario_ego_location = np.array([ego_location])
    env.scenario_ego_rotation = np.array([ego_rotation])
