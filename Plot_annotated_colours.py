import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image with the annotated colours
img_ref = cv2.imread('Bild1_annotiert.jpg')
img_orig = cv2.imread('Bild1.jpg')

# convert to cieLab colourspace
#img_ref_cielab = cv2.cvtColor(img_ref, cv2.COLOR_BGR2LAB)
#img_orig_cielab = cv2.cvtColor(img_orig, cv2.COLOR_BGR2LAB)

# create mask out of annotated image (blue [186, 81, 0] (BGR) is signal colour)
mask = cv2.inRange(img_ref, (180, 80, 0), (190, 90, 0))
cv2.imwrite('img_annotated_mask.jpg', mask)
mask_pixels = np.reshape(mask, (-1))

# Determine mean value, standard deviations and covariance matrix
# for the annotated pixels.
# Using cv2 to calculate mean and standard deviations
mean, std = cv2.meanStdDev(img_orig, mask = mask)
print("Mean color values of the annotated pixels")
print(mean)
print("Standard deviation of color values of the annotated pixels")
print(std)

# apply the mask to the original image
masked_img = cv2.bitwise_and(img_orig, img_orig, mask=mask)
cv2.imwrite('masked_img.jpg', masked_img)

# get the colour values of the pixels that match the mask
pixels = np.reshape(img_orig, (-1, 3))
annot_pix_values = pixels[mask_pixels == 255, ]

# plot the values in a graph (RGB as axes)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = np.array([annot_pix_values[:, 2], annot_pix_values[:, 1], annot_pix_values[:, 0]]).T
print(type(annot_pix_values))
ax.scatter(annot_pix_values[:, 2], annot_pix_values[:, 1], annot_pix_values[:, 0], s=1, c=colors/255.0)

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()