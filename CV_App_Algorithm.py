import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

# function for detection algorithm
def detect(image):
    # Noise Reduction with Median Blur
    blurred_image = cv2.medianBlur(image, 5)

    # Color Segmentation in HSV Color Space
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([5, 20, 5])
    upper_brown = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Morphological Operations
    opened_image = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    processed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

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

    return image, object_coordinates

