import cv2
import numpy as np
import time

# execution times
algorithm_times = []

# start measuring execution time
start_time_total = time.time()

last_objects = []

# Define an object detection function
def detect_objects(image):
    # Blur (Weichzeichnung)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Color Based Segmentation (Farbsegmentierung)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([5, 10, 10]) #230, 189, 168
    upper_brown = np.array([30, 255, 255]) #18, 188, 42
    mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    color_segmented_image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)

    # Grayscale (Umwandlung in Graustufen)
    gray_image = cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2GRAY)

    # Morphological Filtering (Morphologische Filterung)
    kernel = np.ones((5, 5), np.uint8)
    morph_filtered_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

    # Object Filtering
    contours, _ = cv2.findContours(morph_filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Liste zur Speicherung der Objektskoordinaten
    object_coordinates = []
    objects = []

    # Filtern der Objekte nach Fläche
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500:  # Größe der Kontur als Filterkriterium
            x, y, w, h = cv2.boundingRect(contour)
            cX = x + w // 2
            cY = y + h // 2
            radius = int(max(w, h) / 2)
            object_coordinates.append((cX, cY))
            objects.append([x, y, w, h])
    
    end_time = time.time()
    algorithm_times.append(end_time - start_time)

    return object_coordinates, objects

# Read the video and extract frames
start_time_setup = time.time()
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Define the frame interval for object detection
f_detect = 1 # frequency of detection
n = cap.get(cv2.CAP_PROP_FPS) // f_detect  # detect objects every 5th frame

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
        object_coordinates, objects = detect_objects(frame)
        end_time = time.time()
        duration = end_time - start_time
        algorithm_times.append(duration)
        last_objects = objects
    else:
        objects = last_objects

    # Draw circles around the detected objects
    for x, y, w, h in objects:
        cv2.circle(frame, (x + w // 2, y + h // 2), max(w, h) // 2, (0, 255, 0), 2)
 
    # Write the frame to the output video
    out.write(frame)

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