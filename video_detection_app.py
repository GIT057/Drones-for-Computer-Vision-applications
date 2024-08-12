import cv2
import numpy as np
import matplotlib.pyplot as plt

# Schritt 1: Bild einlesen
image = cv2.imread('Bild1.jpg')

# Schritt 2: Blur (Weichzeichnung)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Schritt 3: Color Based Segmentation (Farbsegmentierung)
hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
lower_brown = np.array([5, 10, 10]) #230, 189, 168
upper_brown = np.array([30, 255, 255]) #18, 188, 42
mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
color_segmented_image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)

# Schritt 4: Grayscale (Umwandlung in Graustufen)
gray_image = cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2GRAY)

# Schritt 5: Morphological Filtering (Morphologische Filterung)
kernel = np.ones((5, 5), np.uint8)
morph_filtered_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

# Schritt 6: Object Filtering
contours, _ = cv2.findContours(morph_filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Liste zur Speicherung der Objektskoordinaten
object_coordinates = []

# Zeichnen der Konturen, Kreise und Kreuzes auf dem Originalbild
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 500:  # Größe der Kontur als Filterkriterium
        x, y, w, h = cv2.boundingRect(contour)
        cX = x + w // 2
        cY = y + h // 2
        radius = int(max(w, h) / 2)
        # Zeichne einen Kreis um das erkannte Objekt
        cv2.circle(image, (cX, cY), radius, (0, 255, 255), 2)  # Gelber Kreis
        # Zeichne ein Kreuz im Mittelpunkt des Kreises
        cv2.drawMarker(image, (cX, cY), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        # Füge die Position als Text hinzu, inkl. der Objektnummer
        font_scale = 1  # Schriftgröße experimentell angepasst
        font_thickness = 2
        text = f'{i+1}: ({cX}, {cY})'
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        # Berechne die Position für den Text (unterhalb des Mittelpunkts)
        text_x = cX + 10
        text_y = cY + radius + text_height + 10  # Unterhalb des Kreises
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        object_coordinates.append((cX, cY))