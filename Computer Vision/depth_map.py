import carla
import numpy as np
import cv2
import time
import math
import threading

# Global variables to store the latest images
latest_image1 = None
latest_image2 = None
baseline = 0.1  # Baseline in meters (ensure this matches the physical camera separation in CARLA)
time_depth = 0
minimum_depth = 0
image_lock = threading.Lock()  # Lock for thread-safe image access
stop_flag = threading.Event()  # Event to signal the processing thread to stop

def process_image1(image):
    """Convert CARLA image to numpy array for Camera 1."""
    global latest_image1
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    with image_lock:
        latest_image1 = array

def process_image2(image):
    """Convert CARLA image to numpy array for Camera 2."""
    global latest_image2
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    with image_lock:
        latest_image2 = array

def compute_depth_map(imgL, imgR, method="SGBM", numDisparities=64, blockSize=15, preFilterType=1, smoothing=True, smoothing_kernel_size=7):
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
    
    # print("Time to compute depth map: ", time.time() - time_depth)
    

    return depth_map

def calculate_depth_for_bounding_box(disparity_map, bounding_box):
    """Calculate average and minimum depth in a bounding box from a disparity map."""
    global minimum_depth

    width = disparity_map.shape[1]
    camerafov = 90
    f = width * 10 / (2 * math.tan(camerafov * math.pi / 360))
    b = baseline  # Baseline in meters

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

    return quarter_depth, minimum_depth

def process_images():
    """Thread function to process images and compute depth maps."""
    global latest_image1, latest_image2
    
    while not stop_flag.is_set():
        with image_lock:
            img1 = latest_image1
            img2 = latest_image2
        
        if img1 is not None and img2 is not None:
            depth_map = compute_depth_map(img1, img2)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2RGBA)
            im_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            
            print("Minimum Depth: ", minimum_depth)
            cv2.imshow('Depth Map', im_color)
            # cv2.imshow('Camera 1 View', img1)
            # cv2.imshow('Camera 2 View', img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()

    cv2.destroyAllWindows()

def main():
    """Main function to initialize and run the CARLA client."""
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    world.tick()

    print(f'Connected to {world.get_map().name}')

    # Set up sensors
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    # Find a vehicle actor to attach the cameras
    vehicle = None
    for actor in world.get_actors():
        if actor.attributes.get("role_name") == "ali":
            vehicle = actor
            break

    if not vehicle:
        print("Vehicle not found!")
        return

    # Set up cameras
    camera1_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera2_transform = carla.Transform(carla.Location(x=1.5, z=2.4, y=baseline))

    camera1 = world.spawn_actor(camera_bp, camera1_transform, attach_to=vehicle)
    camera2 = world.spawn_actor(camera_bp, camera2_transform, attach_to=vehicle)

    camera1.listen(process_image1)
    camera2.listen(process_image2)

    # Start the image processing thread
    processing_thread = threading.Thread(target=process_images)
    processing_thread.start()

    try:
        processing_thread.join()
    except KeyboardInterrupt:
        stop_flag.set()
        processing_thread.join()
    finally:
        camera1.stop()
        camera2.stop()
        # vehicle.destroy()

if __name__ == '__main__':
    main()


# cd C:\Users\G7\Documents\comm\CARLA-Object-Detection\ 