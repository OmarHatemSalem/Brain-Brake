## Modifeid by Kasra  Mokhtari, Apr 26th, features:  (run this pls: python spawn_npc_Kasra_FV_1.py -w 1)
# 1) has class for spawning vehicles and pedestrians
# 2) Pedesrian control function is separated from pedestrians spawning function so I can update target location
# 3) check the pedestrian location to target location and will stop pedestrian when it hits the target
#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time
import numpy as np
import argparse
import logging
import random


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

start = time.time()
print('Initialization ...')

class SpawningTask:
    def __init__(self):
        self.argparser = argparse.ArgumentParser(
        description=__doc__)
        self.argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        self.argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        self.argparser.add_argument(
            '-n', '--number-of-vehicles',
            metavar='N',
            default=10,
            type=int,
            help='number of vehicles (default: 10)')
        self.argparser.add_argument(
            '-w', '--number-of-walkers',
            metavar='W',
            default=50,
            type=int,
            help='number of walkers (default: 50)')
        self.argparser.add_argument(
            '--safe',
            action='store_true',
            help='avoid spawning vehicles prone to accidents')
        self.argparser.add_argument(
            '--filterv',
            metavar='PATTERN',
            default='vehicle.*',
            help='vehicles filter (default: "vehicle.*")')
        self.argparser.add_argument(
            '--filterw',
            metavar='PATTERN',
            default='walker.pedestrian.*',
            help='pedestrians filter (default: "walker.pedestrian.*")')
        self.argparser.add_argument(
            '--tm-port',
            metavar='P',
            default=8000,
            type=int,
            help='port to communicate with TM (default: 8000)')
        self.argparser.add_argument(
            '--sync',
            action='store_true',
            help='Synchronous mode execution')
        self.argparser.add_argument(
            '--hybrid',
            action='store_true',
            help='Enanble')
        self.args = self.argparser.parse_args()
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        self.synchronous_master = False
        self.world = self.client.get_world()
        self.blueprints = self.world.get_blueprint_library().filter(self.args.filterv)
        self.blueprintsWalkers = self.world.get_blueprint_library().filter(self.args.filterw)

        if self.args.safe:
            self.blueprints = [x for x in self.blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('isetta')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('carlacola')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('cybertruck')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('t2')]

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.number_of_spawn_points = len(self.spawn_points)
        # @todo cannot import these directly.
        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.FutureActor = carla.command.FutureActor
        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        if self.args.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)
        if self.args.sync:
            self.settings = self.world.get_settings()
            self.traffic_manager.set_synchronous_mode(True)
            if not self.settings.synchronous_mode:
                self.synchronous_master = True
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(self.settings)
            else:
                self.synchronous_master = False

        if self.args.number_of_vehicles < self.number_of_spawn_points:
            random.shuffle(self.spawn_points)
        elif self.args.number_of_vehicles > self.number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.args.number_of_vehicles, self.number_of_spawn_points)
            self.args.number_of_vehicles = self.number_of_spawn_points

    def Apply_Settings_Kasra(self):
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()
        print('Settings are applied!')
        return self.vehicles_list, self.walkers_list, self.all_id
        

    def SpawnVehicles(self, vehicles_list):   
        # --------------
        # Spawn vehicles
        # --------------
        self.batch = []
        self.vehicles_list = vehicles_list
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.number_of_spawn_points = len(self.spawn_points)
        for n, transform in enumerate(self.spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            self.batch.append(self.SpawnActor(blueprint, transform).then(self.SetAutopilot(self.FutureActor, True, self.traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(self.batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)
        self.world.tick()                
        print('Vehicles are spawned!')

    def SpawnWalker(self, walkers_list, all_id, vehicles_list):
        self.walkers_list = walkers_list
        self.all_id = all_id
        percentagePedestriansRunning = 0.0      # percentage pedestrians will run
        percentagePedestriansCrossing = 0.0
        Pedestrian_Locations_Read_From_Map = np.array([[2835.450439,1153.49939,80.0]])
        Pedestrian_Locations_Read_From_Map = (10**-2)*(Pedestrian_Locations_Read_From_Map)    # convert cm to m
        spawn_points = []
        s = (1,3)
        Pedestrian_Location_Read_From_Map = np.zeros(s)
        epsilon = 0.5
        for i in range(self.args.number_of_walkers):
            if i!=1:
                Pedestrian_Location_Read_From_Map[0][0] = Pedestrian_Locations_Read_From_Map[i-1][0] - epsilon
                Pedestrian_Location_Read_From_Map[0][1] = Pedestrian_Locations_Read_From_Map[i-1][1] 
                Pedestrian_Location_Read_From_Map[0][2] = Pedestrian_Locations_Read_From_Map[i-1][2]
                Pedestrian_Locations_Read_From_Map = np.concatenate((Pedestrian_Locations_Read_From_Map, Pedestrian_Location_Read_From_Map))
            spawn_point = carla.Transform()
            spawn_point.location.x = Pedestrian_Locations_Read_From_Map[i][0]
            spawn_point.location.y = Pedestrian_Locations_Read_From_Map[i][1]
            spawn_point.location.z = Pedestrian_Locations_Read_From_Map[i][2]
            #print(loc)
            if (spawn_point.location != None):
                spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        self.batch = []    
        walker_speed = []
        for spawn_point in spawn_points:
            self.walker_bp = self.blueprintsWalkers [4]
            # set as not invincible
            if self.walker_bp.has_attribute('is_invincible'):
                self.walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if self.walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(self.walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(self.walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            self.batch.append(self.SpawnActor(self.walker_bp, spawn_point))
                
        results = self.client.apply_batch_sync(self.batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                print('Pedestrian %d-th is not spawned' %(i))
                print('Location of that pedestrian is:', (10**2)*spawn_points[i].location)
                #logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        #print(self.walkers_list)
        self.walker_speed = walker_speed2
        # 3. we spawn the walker controller
        self.batch = []
        self.walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        #print(walker_controller_bp)
        for i in range(len(self.walkers_list)):
            #print( walkers_list[i]["id"])
            self.batch.append(self.SpawnActor(self.walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
            #print('Batch2', batch)

        results = self.client.apply_batch_sync(self.batch, True)
        for i in range(len(results)):
            #print(results[i].error)
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id

        #print(walkers_list)        
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)
        #print(all_actors)
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
    
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()  
        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(self.walkers_list)))       
        return vehicles_list, self.walkers_list, self.all_id, self.all_actors, self.walker_speed 

        

    def PedestrianTarget(self, Target_Locations, vehicles_list, walkers_list, all_id, all_actors, walker_speed): 
        percentagePedestriansCrossing = 0.0
        self.walkers_list = walkers_list
        self.all_actors = all_actors
        self.all_id = all_id
        self.walker_speed = walker_speed
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            Target_1 = self.world.get_random_location_from_navigation()
            if int(i/2) %2 == 0:
                Target_1.x = Target_Locations[0][0]
                Target_1.y = Target_Locations[0][1]
                Target_1.z = Target_Locations[0][2]
            else:    
                Target_1.x = Target_Locations[1][0]
                Target_1.y = Target_Locations[1][1]
                Target_1.z = Target_Locations[1][2]
            #print('Target Location %f:' %(i+1), Target)
            Target = Target_1
            self.all_actors[i].go_to_location(Target)
            # max speed
            self.all_actors[i].set_max_speed(float(self.walker_speed[int(i/2)]))    
            time.sleep(1.0)

       
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick() 
        
        return vehicles_list, self.walkers_list, self.all_id, self.all_actors    

    
    def GetPedestrianLocation(self, Target_Location, Threshold,  all_id):
        ReachTarget = False
        id = all_id
        actor_list = self.world.get_actors()
        # Find an actor by id.
        actor = actor_list.find(id[0])
        PedestrianLocation = actor.get_location() 
        PedestrianLocation_matrix = np.array([[PedestrianLocation.x, PedestrianLocation.y, Target_Location[0][2]]])
        zero_Vector = [[0,0,0]]
        Target_Location_m = np.array(Target_Location)
        distance_target_N = np.linalg.norm(PedestrianLocation_matrix-Target_Location_m)
        distance_target_D = np.linalg.norm(zero_Vector-Target_Location_m)
        distance_target  = distance_target_N/ distance_target_D

        if distance_target < Threshold:
            ReachTarget = True
            print('Yohoo! Pedestrian arrived at the target!')
        
        #print(PedestrianLocation_matrix)
        print('Distance is:', distance_target)    
        return ReachTarget, PedestrianLocation_matrix
    
    def StopPedestrian(self,ReachTarget, walkers_list, all_id, all_actors, walker_speed):
        if ReachTarget == True:
            percentagePedestriansCrossing = 0.0
            self.walkers_list = walkers_list
            self.all_actors = all_actors
            self.all_id = all_id
            self.walker_speed = walker_speed
            for i in range(len(walker_speed)):
                walker_speed[i] = 0
            
            percentagePedestriansCrossing = 0.0
            self.walkers_list = walkers_list
            self.all_actors = all_actors
            self.all_id = all_id
            self.walker_speed = walker_speed
            for i in range(len(walker_speed)):
                walker_speed[i] = 0
            self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(all_id), 2):
                # start walker
                self.all_actors[i].start()
                # set walk to random point
                Target = self.world.get_random_location_from_navigation()
                self.all_actors[i].go_to_location(Target)
                # max speed
                self.all_actors[i].set_max_speed(float(self.walker_speed[int(i/2)]))
        else:
            pass            
       
    def DestroyActors(self, vehicles_list, walkers_list, all_id, all_actors):

        print('\ndestroying %d vehicles' % len(vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        '''
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()
        '''

def main():
    
    Threshold = 0.32
    Target_Locations = [[1500.087891,2257.480469,80.0], [2835.450439,1153.49939,80.0]]
    Target_Locations = (10**-2)*np.asarray(Target_Locations)
    
    # run codes only once
    vehicles_list, walkers_list, all_id = SpawningTask().Apply_Settings_Kasra()
    SpawningTask().SpawnVehicles(vehicles_list)
    vehicles_list, walkers_list, all_id, all_actors, walker_speed = SpawningTask().SpawnWalker(walkers_list, all_id, vehicles_list)
    
    # run codes every second
    ReachTarget = False
    Start_time  = time.time()
    end_time = time.time()
    elapsed_time = end_time- Start_time
    while elapsed_time < 25 and  ReachTarget == False:
        
        SpawningTask().PedestrianTarget(Target_Locations, vehicles_list, walkers_list, all_id, all_actors, walker_speed) # it takes 1.7 seconds to send the command to simulation in each loop (1.7sec* 1.8 m/s  = no control)
        ReachTarget, PedestrianLocation_matrix = SpawningTask().GetPedestrianLocation(Target_Locations, Threshold,  all_id)
        SpawningTask().StopPedestrian(ReachTarget, walkers_list, all_id, all_actors, walker_speed)
        if ReachTarget == True:
            print('Time to get there is %f seconds:' %(elapsed_time))
        end_time = time.time()
        elapsed_time = end_time- Start_time


    print('Almost Done!')
    print(all_id)
    print(all_actors)
    time.sleep(2)    
    SpawningTask().DestroyActors(vehicles_list, walkers_list, all_id, all_actors)

if __name__ == '__main__':
    main() 