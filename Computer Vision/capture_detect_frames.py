# from YOLO.ImageDetector import Yolo_Detector
import carla
import numpy as np
import cv2
import threading
import time
# from keras.models import load_model
from queue import Queue
import threading
import math 
# Global variables to store the latest images
latest_image1 = None
latest_image2 = None
baseline = 0.1  # Baseline in meters (ensure this matches the physical camera separation in CARLA)
# yolo = Yolo_Detector()
time_depth = 0
# load the model from file calibration_for_distance_regression.h5
# model = load_model('calibration_for_distance_regression')






def process_image1(image):
    """Convert CARLA image to numpy array for Camera 1."""
    global latest_image1
    print("debug")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    latest_image1 = array

def process_image2(image):
    """Convert CARLA image to numpy array for Camera 2."""
    global latest_image2
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    latest_image2 = array

def compute_depth_map(imgL, imgR, output_queue, method="SGBM", numDisparities=64, blockSize=15, preFilterType=1, smoothing=True, smoothing_kernel_size=7):
    """
    Compute depth map from stereo images.

    Parameters:
    imgL: Left image
    imgR: Right image
    method: Stereo matching method, either 'BM' or 'SGBM'
    numDisparities: Number of disparities
    blockSize: Size of the block window
    preFilterType: Pre-filter type for StereoBM
    smoothing: Apply median blur to the depth map
    smoothing_kernel_size: Kernel size for median blur
    """
    # Validate inputs
    global time_depth
    time_depth = time.time()
    if imgL is None or imgR is None:
        raise ValueError("Input images cannot be None")

    if imgL.shape[:2] != imgR.shape[:2]:
        raise ValueError("Left and right images must have the same size")

    # Convert images to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # StereoBM or StereoSGBM settings
    if method == "BM":
        stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
        stereo.setPreFilterType(preFilterType)
    else:
        stereo = cv2.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize, 
                                       minDisparity=0,
                                       uniquenessRatio=10,
                                       speckleWindowSize=100,
                                       speckleRange=32,
                                       disp12MaxDiff=1,
                                       P1=8*3*blockSize**2,
                                       P2=32*3*blockSize**2)

    # Compute disparity and normalize
    disparity = stereo.compute(grayL, grayR).astype(np.float32)
    depth_map = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Optional smoothing
    if smoothing:
        depth_map = cv2.medianBlur(depth_map, smoothing_kernel_size)
    
    #output_queue.put(disparity)
    output_queue.put(depth_map)

    print("Time to compute depth map: ", time.time() - time_depth)

minimum_depth = 0



def calculate_depth_for_bounding_box(disparity_map, bounding_box):
    """Calculate average and minimum depth in a bounding box from a disparity map."""
    global minimum_depth

    width = disparity_map.shape[1]
    

    camerafov = 90

    f = width * 10 / (2 * math.tan(camerafov * math.pi / 360))

    # f = 2800  # Focal length in pixels
    b = baseline  # Baseline in meters

    

    # # Connect to CARLA server
    # client = carla.Client('localhost', 2000) 
    # client.set_timeout(2.0)
    # world = client.get_world()

    # ego_car_x = -7.3
    # vehicle = None
    # for actor in world.get_actors():
    #     if 'vehicle' in actor.type_id:
    #         vehicle = actor
    #         break

    # driver_car_x = vehicle.get_location().x

    # distance_x = abs(ego_car_x - driver_car_x)

    # distance to half of the car
    

    # Convert disparity to depth
    with np.errstate(divide='ignore'):  # Handle division by zero
        depth_map = np.where(disparity_map != 0, (f * b) / disparity_map, 0) 

    # Extract depth values within the bounding box
    box_depth = depth_map[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
    valid_depths = box_depth[box_depth > 0]

    # Calculate average and minimum depth values
    if valid_depths.size > 0:
        quarter_depth = np.quantile(valid_depths, 0.25)
        minimum_depth = np.min(valid_depths)
        minimum_depth = round(minimum_depth, 1)
    else:
        quarter_depth, minimum_depth = float('inf'), float('inf')

    #Distance Calibration (temp solution)
    # quarter_depth = quarter_depth * (1800/143) - (146/143) - 0.95
    # minimum_depth = minimum_depth * (1800/143) - (146/143)

    

    #More Distane Calibration (lower part of  the car is not detected)
    # quarter_depth = quarter_depth -0.95
    # minimum_depth = minimum_depth -0.95

    return quarter_depth, minimum_depth



# def compare_hashem_regression_actual(file):

#     # Connect to CARLA server
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(2.0)
#     world = client.get_world()

#     ego_car_x = -7.3
#     vehicle = None
#     for actor in world.get_actors():
#         if 'vehicle' in actor.type_id:
#             vehicle = actor
#             break

#     driver_car_x = vehicle.get_location().x

#     actual = abs(ego_car_x - driver_car_x)

#     hashem = minimum_depth * (1800/143) - (146/143) - 0.95

    

#     regression = model.predict(np.array([minimum_depth]))[0][0]

#     # make them round to 1 decimal place
#     hashem = round(hashem, 1)
#     regression = round(regression, 1)
#     actual = round(actual, 1)

#     # append to csv file hashem, regression and actual under the column name of hashem, regression and actual
#     with open(file, 'a') as f:
#         f.write(f"{hashem},{regression},{actual}\n")








# def linear_regression_for_distance_calibration(file):
#     """Calculate the linear regression for distance calibration."""

#     # Connect to CARLA server
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(2.0)
#     world = client.get_world()

#     ego_car_x = -7.3
#     vehicle = None
#     for actor in world.get_actors():
#         if 'vehicle' in actor.type_id:
#             vehicle = actor
#             break

#     driver_car_x = vehicle.get_location().x

#     distance_x = abs(ego_car_x - driver_car_x)

    
#     # append to csv file distance_x and minimum_depth under the column name of actual and estimated
#     with open(file, 'a') as f:
#         f.write(f"{distance_x},{minimum_depth}\n")

    
# def detect_objects_and_get_bounding_boxes(image, yolo, output_queue):
#     bounding_boxes = yolo.detect_objects(image).cpu().numpy().astype(int)
#     #bounding_boxes = yolo.detect_objects(image).numpy().astype(int)
#     output_queue.put(bounding_boxes)
    





def display_loop():
    """Continuously display the latest images using OpenCV."""
    global latest_image1, latest_image2
    
    while True:
        if latest_image1 is not None and latest_image2 is not None:
            depth_map_queue = Queue()
            # bounding_boxes_queue = Queue()
            # remove noise from the image
            depth_map_thread= threading.Thread(target=compute_depth_map, args=(latest_image1, latest_image2, depth_map_queue))
            # object_detection_thread = threading.Thread(target=detect_objects_and_get_bounding_boxes, args=(latest_image1, yolo, bounding_boxes_queue))
            depth_map_thread.start()
            # object_detection_thread.start()
            
            depth_map_thread.join()
            # object_detection_thread.join()


            depth_map = depth_map_queue.get()
            # bounding_boxes_1 = bounding_boxes_queue.get()

            # print(dlinear_regression_for_distance_calibrationepth_map)
            latest_image1 = cv2.cvtColor(latest_image1, cv2.COLOR_RGB2RGBA)
            im_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            

            # for box in bounding_boxes_1:
            #     if box[5] == 2:  # Class '2' for cars
            #         quar_depth, min_depth = calculate_depth_for_bounding_box(depth_map, box)
            #         print(f"Quarter Depth: {quar_depth}, Minimum Depth: {min_depth}")
            #         cv2.rectangle(im_color, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            #         cv2.putText(im_color, f"Depth: {quar_depth:.2f}m", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print("Minimum Depth: ", minimum_depth)
            # cv2.imshow('Depth Map', im_color)
            # cv2.imshow('Camera 1 View', latest_image1)
            # cv2.imshow('Camera 2 View', latest_image2)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()

def main():
    """Main function to initialize and run the CARLA client."""
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    world.tick()


    # create csv file with two columns: actual and estimated
    # file = "distance_calibration.csv"
    # create a csv file hashem and regresion with the columns: hashem, regression
    # file = "test_hashem_regression_actual.csv"
    
    # with open(file, 'w') as f:
    #     f.write("Hashem,Regression,Actual\n")
    print(f'Connected to {world.get_map().name}')

    # Find a vehicle actor to attach the cameras
    vehicle = None
    for actor in world.get_actors():
        print(actor.attributes.get("role_name"))
        if actor.attributes.get("role_name") == "ali":
            vehicle = actor
            break

    if not vehicle:
        print("Vehicle not found!")
        return

    # Define the camera blueprint and initial transform
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform1 = carla.Transform(carla.Location(x=1.5,y = -baseline/2, z=1.5))
    # camera_transform2 = carla.Transform(carla.Location(x=1.5, y=baseline, z=1.5))
    camera_transform2 = carla.Transform(carla.Location( y= baseline))

    # Spawn the first camera and attach it to the vehicle
    camera_actor1 = world.spawn_actor(camera_bp, camera_transform1, attach_to=vehicle)
    camera_actor1.listen(lambda image: process_image1(image))

    # Spawn the second camera using the adjusted transform and attach it to the same vehicle
    camera_actor2 = world.spawn_actor(camera_bp, camera_transform2, attach_to=camera_actor1)
    camera_actor2.listen(lambda image: process_image2(image))

    # Start a separate thread to display the images
    display_thread = threading.Thread(target=display_loop)
    display_thread.start()



    try:
        # Keep the main script running to avoid exiting and losing the camera feed
        while True:
            # linear_regression_for_distance_calibration("distance_calibration.csv")
            # compare_hashem_regression_actual(file)
            # time.sleep(0.5)
            pass
    finally:
        # Cleanup
        # camera_actor1.stop_listening()
        # camera_actor2.stop_listening()
        camera_actor1.destroy()
        camera_actor2.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
