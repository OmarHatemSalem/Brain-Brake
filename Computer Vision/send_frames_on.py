import glob
import os
import sys
import cv2
import numpy as np
import socket
import struct
import carla
import time
import threading

# Adding CARLA PythonAPI to the path
try:
    sys.path.append(glob.glob('C:/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Global variables
latest_image1 = None
latest_image2 = None
image_lock = threading.Lock()

def process_image1(image):
    global latest_image1
    with image_lock:
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        latest_image1 = image_data.reshape((image.height, image.width, 4))[:, :, :3]

def process_image2(image):
    global latest_image2
    with image_lock:
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        latest_image2 = image_data.reshape((image.height, image.width, 4))[:, :, :3]

def send_frame():
    HOST = "192.168.1.2"
    PORT = 65433
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print("Server started. Waiting for a connection...")
        while True:  # Keep server running to accept multiple connections
            conn, addr = server_socket.accept()
            print(f"Connection from {addr}")
            try:
                while True:
                    with image_lock:
                        if latest_image1 is not None and latest_image2 is not None:
                            for img in [latest_image1, latest_image2]:
                                result, encoded_img = cv2.imencode('.jpg', img)
                                if result:
                                    data = encoded_img.tobytes()
                                    data_length = len(data)
                                    # print(f"Sending {data_length} bytes of image data.")
                                    conn.sendall(struct.pack(">L", data_length) + data)
                                else:
                                    print("Failed to encode image.")
                    time.sleep(0.05)
            except Exception as e:
                print(f"An error occurred with {addr}: {e}")
            finally:
                print(f"Closing connection with {addr}")
                conn.close()

def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    world.tick()

    vehicle = None
    for actor in world.get_actors():
        if actor.attributes.get("role_name") == "ali":
            vehicle = actor
            break

    if not vehicle:
        print("Vehicle not found!")
        return

    b = 0.5
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform1 = carla.Transform(carla.Location(x=2, y=(-b/2), z=1.5))
    camera_actor1 = world.spawn_actor(camera_bp, camera_transform1, attach_to=vehicle)
    camera_actor1.listen(lambda image: process_image1(image))

    offset = carla.Location(y=b/2)
    camera_transform2 = carla.Transform(camera_transform1.location + offset, camera_transform1.rotation)
    camera_actor2 = world.spawn_actor(camera_bp, camera_transform2, attach_to=vehicle)
    camera_actor2.listen(lambda image: process_image2(image))

    send_frame()

if __name__ == '__main__':
    main()
