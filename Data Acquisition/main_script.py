import os
import argparse
import multiprocessing as mp
import time


def main():
    argparser = argparse.ArgumentParser(
        description='Main Script for Experiment')
    argparser.add_argument(
            '--name',
            metavar='NAME',
            default='anonymous',
            help='name of participant')
    
    argparser.add_argument(
            '--minutes',
            default=15,
            type=float,
            help='The length of the experiment in minutes')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--steering',
        action='store_true',
        help='using steerring wheel (default: False)')
    argparser.add_argument(
        '--eeg',
        action='store_true',
        help='using steerring wheel (default: False)')

    args = argparser.parse_args()


    try:    
        if args.steering:
            control_command = f"python ali_manual_control_steeringwheel.py --res {args.res} --rolename driver"
        else:
            control_command = f"python manual_control.py --sync --res {args.res} --rolename driver"
        
        time.sleep(5)
        
        npc_command = f"python npc_car.py --name {args.name}"
        
        print(f"Hello {args.name}. Welcome to the EEG experiment. The experiment will run for {args.minutes} minutes.")
        commands = [ control_command, npc_command]
        if args.eeg: 
            eeg_command = f"python experiment_acquisition.py --name {args.name} --minutes {args.minutes}"
            commands.append(eeg_command)
        # commands = [eeg_command, control_command, npc_command]

        pool = mp.Pool(processes=mp.cpu_count())
        pool.map(os.system, commands)
        # os.system(f"python experiment_acquisition.py --name {args.name} --minutes {args.minutes}")

        # os.system(f"python manual_control.py --sync --res 2560x1440 --rolename driver")

        # os.system(f"python npc_car.py --name {args.name}")
    except KeyboardInterrupt:
        commands[1].terminate()
        commands[0].terminate()
        if args.eeg:
            commands[2].terminate()



    print("The experiment has finished. Thank you for participating.")

if __name__ == "__main__":
    main()