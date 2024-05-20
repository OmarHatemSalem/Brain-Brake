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


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import random
import cv2
import numpy as np
from carla import ColorConverter as cc

from visual_timer import submit


# # Create a pedestrian walker
# def create_walker(world):
#     # global world

#     blueprintsWalkers = random.choice(world.get_blueprint_library().filter("walker.*"))
#     walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')


#     spawn_points = world.get_map().get_spawn_points()
#     # print(spawn_points)
#     start_spawn_point = spawn_points[92]
#     print(f"Spawning pedestrian at {start_spawn_point.location}")
#     walker_actor = world.try_spawn_actor(blueprintsWalkers, start_spawn_point)
#     print(f"Pedestrian found at {walker_actor.get_location()}")
#     walker_controller_actor = world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
#     print('Creating walker with preset route')

#     end_spawn_point = spawn_points[88]
#     walker_controller_actor.start()
#     walker_controller_actor.go_to_location(end_spawn_point.location)
#     # print(world.get_random_location_from_navigation())
#     print(f"end spawn point at {end_spawn_point.location}")
#     # walker_controller_actor.go_to_location(world.get_random_location_from_navigation())
#     walker_controller_actor.set_max_speed(1.4)

#     return walker_actor, walker_controller_actor, start_spawn_point, end_spawn_point    

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
    # settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    random.seed(int(time.time()))

    # We will aslo set up the spectator so we can see what we do
    spectator = world.get_spectator()

    # Get the spawn points from the map
    spawn_points = world.get_map().get_spawn_points()

    blueprint_library = world.get_blueprint_library()

    # Now let's filter all the blueprints of type 'vehicle' and choose one
    # at random.
    bp = blueprint_library.find('vehicle.audi.a2')
    sim_car_spawn_point = spawn_points[51]
    print('spawning vehicle at', sim_car_spawn_point.location)

    

    vehicles = []

    sim_vehicle = world.spawn_actor(bp, sim_car_spawn_point)
    sim_vehicle.set_autopilot(True) # original

    traffic_manager.ignore_lights_percentage(sim_vehicle,100)
    

    distance = 10
    
    route_1_indices = [113, 88, 20, 44, 41, 98, 105, 45, 114, 49, 51, 90, 96, 2, 120, 4, 121, 82, 71, 60, 143, 149, 92, 21, 76, 32, 23]


    route_1 = []
    for ind in route_1_indices:
        route_1.append(spawn_points[ind].location)



    # Now let's print them in the map so we can see our routes
    world.debug.draw_string(sim_car_spawn_point.location, 'Spawn point 1', life_time=30, color=carla.Color(255,0,0))

    
    for ind in route_1_indices:
        spawn_points[ind].location
        world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60, color=carla.Color(255,0,0))

    # Set parameters of TM vehicle control, we don't want lane changes
    traffic_manager.set_path(sim_vehicle, route_1)


    for index, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(index), draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=50.0,
                                persistent_lines=True)





 

    print('Creating vehicle with preset route')
    world.wait_for_tick()

    #trail mechanism
    trail_counter = 0
    p1 = None    
    p2 = None    
    #Braking mechanism
    counter = 0
    braking = False  # Flag to indicate if we are currently braking
    braking_time = 500
    braking_wait = 100
    braking_events = [] # List to store the timestamps of braking events

   
    #tracking the agent
    all_actors = world.get_actors()
    agent = None
    # Iterate over the list to find the actor you're interested in
    for actor in all_actors: 
        if actor.type_id == 'vehicle.dodge.charger_2020': agent = actor

    brakeProb = random.randint(0, 100)
     # for testing dangerous remove later
    while True:
        # Check if the vehicle has been destroyed
        if not sim_vehicle.is_alive:
            print("Vehicle has been destroyed")
            break

        print(brakeProb)
        if (brakeProb >= 30) and (braking_time <= counter <= braking_time+braking_wait):  # 10% chance to start braking
            sim_vehicle.set_autopilot(False, traffic_manager.get_port())
            

                

            control = sim_vehicle.get_control()
            # Modify the control to brake
            control.brake =  1.0

            # Send the modified control
            sim_vehicle.apply_control(control)

            # set speed to zero
            
            if not braking:
                braking_event_time = time.time_ns()
                braking_events.append([braking_event_time, 1])

            braking = True
            if counter == braking_time + int(braking_wait/3):
                if 5 > distance: braking_events[-1][1] = 2
                elif distance > 40: braking_events.pop()
          
            



        elif counter > braking_time + braking_wait:  # After 50 ticks, stop braking
            sim_vehicle.set_autopilot(True, traffic_manager.get_port())
            counter = 0
            braking = False
            brakeProb = random.randint(0, 100)

        
        
        if trail_counter == 100: 
            p1 = sim_vehicle.get_location()
        elif trail_counter == 105:
            p2 = sim_vehicle.get_location()
            world.debug.draw_arrow(p1, p2, life_time=120, color=carla.Color(0,255,0))
            trail_counter = 0
            p1 = None
            p2 = None

        # distance = (sim_vehicle.get_location() - agent.get_location()).length()
        
        # if distance > 60:
        #     print("Vehicle is too far from agent. Pausing Route...")
        # else:
        #     print("Vehicle is back. Resuming Route...")

        world.wait_for_tick()

        counter += 1
        trail_counter += 1

except Exception as e:
    print(traceback.format_exc())
finally:
    print('\nDestroying NPC vehicle')
    print('\nDestroying Pedestrian walker')
    
    if sim_vehicle is not None: client.apply_batch([carla.command.DestroyActor(sim_vehicle)])


    with open(args.name+'_braking_events.csv', 'w') as eventFile:
        eventFile.writelines(f"{event[0]},{event[1]}\n" for event in braking_events)