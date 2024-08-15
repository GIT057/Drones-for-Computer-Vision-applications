import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image with the annotated colours
img_ref = cv2.imread(r'C:\Users\grenz\Documents\GitHub\Drones-for-Computer-Vision-applications\Images\capture_14 Aug 2024\img_15 - Kopie.jpg')
img_orig = cv2.imread(r'C:\Users\grenz\Documents\GitHub\Drones-for-Computer-Vision-applications\Images\capture_14 Aug 2024\img_15.jpg')

# create mask out of annotated image (blue [186, 81, 0] (BGR) is signal colour)
mask = cv2.inRange(img_ref, (210, 150, 0), (240, 170, 0))
cv2.imwrite('img_annotated_mask.jpg', mask)
mask_pixels = np.reshape(mask, (-1))

# Determine mean value, standard deviations and covariance matrix
# for the annotated pixels.
mean, std = cv2.meanStdDev(img_orig, mask=mask)
print("Mean color values of the annotated pixels (RGB):")
print(mean)
print("Standard deviation of color values of the annotated pixels (RGB):")
print(std)

# apply the mask to the original image
masked_img = cv2.bitwise_and(img_orig, img_orig, mask=mask)
cv2.imwrite('masked_img.jpg', masked_img)

# get the colour values of the pixels that match the mask
pixels = np.reshape(img_orig, (-1, 3))
annot_pix_values = pixels[mask_pixels == 255]

# Convert the annotated RGB values to HSV
annot_pix_values_hsv = cv2.cvtColor(annot_pix_values.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

# Calculate mean HSV values
mean_hsv = np.mean(annot_pix_values_hsv, axis=0)
std_hsv = np.std(annot_pix_values_hsv, axis=0)

# Calculate min and max HSV values
min_hsv = np.min(annot_pix_values_hsv, axis=0)
max_hsv = np.max(annot_pix_values_hsv, axis=0)

print("Mean color values of the annotated pixels (HSV):")
print(mean_hsv)
print("Standard deviation of color values of the annotated pixels (HSV):")
print(std_hsv)
print("Min HSV values of the annotated pixels:")
print(min_hsv)
print("Max HSV values of the annotated pixels:")
print(max_hsv)

# plot the values in a graph (RGB as axes)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = np.array([annot_pix_values[:, 2], annot_pix_values[:, 1], annot_pix_values[:, 0]]).T
ax.scatter(annot_pix_values[:, 2], annot_pix_values[:, 1], annot_pix_values[:, 0], s=1, c=colors/255.0)

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()
