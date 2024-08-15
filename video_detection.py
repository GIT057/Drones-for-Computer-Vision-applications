import cv2
import numpy as np
import time

import CV_App_Algorithm as cv_alg

# execution times
algorithm_times = []

# start measuring execution time
start_time_total = time.time()

last_objects = []

# Read the video and extract frames
start_time_setup = time.time()
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the frame interval for object detection
f_detect = 10 # frequency of detection
print(f"Original FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
#n = cap.get(cv2.CAP_PROP_FPS) // f_detect  # detect objects every 5th frame
n = 1 # for now disabled the n-th sampling because sub-module doesn't support interpolating

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"output_{f_detect}Hz.mp4", fourcc, fps, (frame_width, frame_height))

# Initialize the last known object positions
last_objects = []

end_time_setup = time.time()

start_time_handling = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply object detection every n-th frame and at first frame
    if len(last_objects) == 0 or cap.get(cv2.CAP_PROP_POS_FRAMES) % n == 0:
        start_time = time.time()
        # now doing the magic
        new_image, obj_coords = cv_alg.detect(frame)
        end_time = time.time()
        duration = end_time - start_time
        algorithm_times.append(duration)
        last_objects = [] #soon to be depreciated
    else:
        objects = last_objects

    # Write the frame to the output video
    out.write(new_image)

cap.release()
out.release()

end_time_handling = time.time()

# calculate average execution time and standard deviation
end_time_total = time.time()
execution_time_total = end_time_total - start_time_total
algorithm_mean = np.mean(algorithm_times)
algorithm_std = np.std(algorithm_times)
algorithm_total = np.sum(algorithm_times)
non_algorithm_total = execution_time_total - algorithm_total
setup_total = end_time_setup - start_time_setup
handling_total = (end_time_handling - start_time_handling) - algorithm_total

# print results
print(f"Average detection execution time: {algorithm_mean:.2f} seconds")
print(f"Standard deviation: {algorithm_std:.2f} seconds")
print(f"Total detection algorithm time: {algorithm_total:.2f} seconds")
print(f"Setuptime: {setup_total:.2f} seconds")
print(f"Non-detection-algorithm time: {non_algorithm_total:.2f} seconds")
print(f"Total handling time: {handling_total:.2f} seconds")
print(f"Total execution time: {execution_time_total:.2f} seconds")

print("Done!")