import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

file_path = r"C:\Users\grenz\Documents\GitHub\Drones-for-Computer-Vision-applications\Images\9 aug 2024\photo_6174946972373467775_y.jpg"


#file_path = r"C:\Users\grenz\Documents\GitHub\Drones-for-Computer-Vision-applications\Images\capture_14 Aug 2024\img_15.jpg"

# Load the image
image = cv2.imread(file_path)

# Display the original image
plt.figure(figsize=(14, 10))  # Adjust figure size for better fitting
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image', fontsize=10)
plt.axis('off')

# Noise Reduction with Median Blur
blurred_image = cv2.medianBlur(image, 5)

# Display the blurred image
plt.subplot(2, 4, 2)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image', fontsize=10)
plt.axis('off')

# Color Segmentation in HSV Color Space
hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
lower_brown = np.array([5, 20, 5])
upper_brown = np.array([30, 255, 255])
mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

# Display the mask
plt.subplot(2, 4, 3)
plt.imshow(mask, cmap='gray')
plt.title('Color Segmentation Mask', fontsize=10)
plt.axis('off')

# Morphological Operations
opened_image = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
processed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# Display the processed image
plt.subplot(2, 4, 4)
plt.imshow(processed_image, cmap='gray')
plt.title('After Morphological Operations', fontsize=10)
plt.axis('off')

# Find Contours for the Detected Regions
contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

object_coordinates = []
min_area = 500
max_area = 50000

# Process Each Detected Contour
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        else:
            continue

        cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)  # Yellow contour
        cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        text = f'({x}, {y})'
        cv2.putText(image, text, (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        object_coordinates.append((x, y))

# Display the final image with detected pots and coordinates
plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Final Detection with Coordinates', fontsize=10)
plt.axis('off')

# Calculate distances between detected pots and draw them
conversion_factor = 0.80  # Diameter of a pot in meters --> Should be dynamic adjusted for future applications
pixel_diameter = np.mean([cv2.norm(np.array(p1) - np.array(p2)) for p1, p2 in itertools.combinations(object_coordinates, 2)])
meters_per_pixel = conversion_factor / pixel_diameter

distances = np.zeros((len(object_coordinates), len(object_coordinates)))
for i, (x1, y1) in enumerate(object_coordinates):
    for j, (x2, y2) in enumerate(object_coordinates):
        if i != j:
            distance_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_meters = distance_pixels * meters_per_pixel
            distances[i, j] = distance_meters
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue line for distances
            cv2.putText(image, f'{distance_meters:.2f}m', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Distances Between Pots', fontsize=10)
plt.axis('off')

# Plot the graph
if object_coordinates:
    plt.subplot(2, 4, 7)
    plt.scatter(*zip(*object_coordinates), c='r', label='Pots')
    for i, (x, y) in enumerate(object_coordinates):
        plt.text(x, y, f'({x},{y})', fontsize=10, ha='right')
    for i, j in itertools.combinations(range(len(object_coordinates)), 2):
        plt.plot([object_coordinates[i][0], object_coordinates[j][0]], [object_coordinates[i][1], object_coordinates[j][1]], 'b-')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    # Invert the y-axis
    plt.gca().invert_yaxis()

    plt.title('Graph of Pots', fontsize=10)
    plt.legend()

# Display the adjacency matrix
plt.subplot(2, 4, 8)
plt.imshow(distances, cmap='viridis', interpolation='none')
plt.colorbar(label='Distance (meters)')
plt.title('Adjacency Matrix', fontsize=10)

# Set ticks to whole numbers
plt.xticks(ticks=np.arange(len(object_coordinates)), labels=np.arange(len(object_coordinates)))
plt.yticks(ticks=np.arange(len(object_coordinates)), labels=np.arange(len(object_coordinates)))


plt.tight_layout()
plt.show()

# Print coordinates of detected pots and adjacency matrix
print(f"Number of objects found: {len(object_coordinates)}")
print(f"Coordinates of objects: {object_coordinates}")
print("Adjacency Matrix (in meters):")
print(distances)
