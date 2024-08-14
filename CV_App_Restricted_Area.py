import cv2
import numpy as np
import matplotlib.pyplot as plt


file_path = r"C:\Users\grenz\Documents\GitHub\Drones-for-Computer-Vision-applications\Images\9 aug 2024\photo_6174946972373467775_y.jpg"


#file_path = r"C:\Users\grenz\Documents\GitHub\Drones-for-Computer-Vision-applications\Images\12 aug 2024\img_6.jpg"

#file_path = r"C:\Users\grenz\Documents\GitHub\Drones-for-Computer-Vision-applications\Images\12 aug 2024\img_14.jpg"
# Step 1: Load the image
image = cv2.imread(file_path)

# Display the original image
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Step 2: Noise Reduction with Median Blur (Optional but Recommended)
blurred_image = cv2.medianBlur(image, 5)

# Display the blurred image
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image')
plt.axis('off')

# Step 3: Color Segmentation in HSV Color Space
hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

# Define brown color range in HSV (adjust if necessary)
lower_brown = np.array([5, 20, 5])  #5, 20, 5
upper_brown = np.array([30, 255, 255]) #30, 255, 255
mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

# Display the mask
plt.subplot(2, 3, 3)
plt.imshow(mask, cmap='gray')
plt.title('Color Segmentation Mask')
plt.axis('off')

# Step 4: Morphological Operations
# Use opening to remove small noise (if any)
opened_image = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

# Display the opened image
plt.subplot(2, 3, 4)
plt.imshow(opened_image, cmap='gray')
plt.title('After Opening')
plt.axis('off')

# Use closing to fill small holes in detected regions
processed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# Display the closed image
plt.subplot(2, 3, 5)
plt.imshow(processed_image, cmap='gray')
plt.title('After Closing')
plt.axis('off')

# Step 5: Find Contours for the Detected Regions
contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

object_coordinates = []
min_area = 500  # Adjust based on image size and expected object size
max_area = 50000  # Adjust based on image size and expected object size

# Step 6: Process Each Detected Contour
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        # Get the center of mass (centroid) and bounding box
        M = cv2.moments(contour)
        if M['m00'] != 0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        else:
            continue

        # Draw the contour
        cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)  # Yellow contour
        cv2.drawMarker(image, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        # Add the coordinates and display them on the image
        text = f'({x}, {y})'
        cv2.putText(image, text, (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        object_coordinates.append((x, y))

# Step 7: Display the final image with detected pots and coordinates
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Final Detection with Coordinates')
plt.axis('off')

# Plot the coordinates on the final image
for idx, (x, y) in enumerate(object_coordinates):
    plt.text(x, y, f'({x},{y})', color='white', fontsize=8, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

plt.tight_layout()
plt.show()

# Print coordinates of detected pots
print(f"Number of objects found: {len(object_coordinates)}")
print(f"Coordinates of objects: {object_coordinates}")
