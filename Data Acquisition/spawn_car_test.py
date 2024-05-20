import glob
import os
import sys
import time
import traceback
import argparse

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random

argparser = argparse.ArgumentParser(
        description='NPC Car for Experiment')
argparser.add_argument(
        '--name',
        metavar='NAME',
        default='anonymous',
        help='name of participant')

args = argparser.parse_args()

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

try:
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    random.seed(int(time.time()))

    # Get the spawn points from the map
    spawn_points = world.get_map().get_spawn_points()

    blueprint_library = world.get_blueprint_library()

    # Filter all the blueprints of type 'vehicle' and choose one at random.
    bp = blueprint_library.find('vehicle.audi.a2')
    sim_car_spawn_point = spawn_points[51]
    print('Spawning vehicle at', sim_car_spawn_point.location)

    # Spawn the vehicle
    sim_vehicle = world.spawn_actor(bp, sim_car_spawn_point)

    # Keep the script running to maintain the spawned vehicle in the world


    #tracking the agent
    all_actors = world.get_actors()
    agent = None
    # Iterate over the list to find the actor you're interested in
    for actor in all_actors: 
        if actor.type_id == 'vehicle.dodge.charger_2020': agent = actor

    while True:
        agent_location = agent.get_location()
        sim_vehicle_location = sim_vehicle.get_location()
        distance = agent_location.distance(sim_vehicle_location)
        print('Distance between agent and spawned vehicle:', distance)

        world.wait_for_tick()

except Exception as e:
    print(traceback.format_exc())
finally:
    print('\nDestroying NPC vehicle')
    if sim_vehicle is not None:
        client.apply_batch([carla.command.DestroyActor(sim_vehicle)])
