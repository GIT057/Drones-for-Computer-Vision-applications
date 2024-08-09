import cv2
import numpy as np

# Lade das Bild
image = cv2.imread('drone_image.jpg')

# Konvertiere das Bild in den HSV-Farbraum
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definiere die Farbgrenzen für braun im HSV-Farbraum
lower_brown = np.array([10, 100, 20])
upper_brown = np.array([20, 255, 200])

# Segmentiere das Bild nach der Farbe braun
mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

# Führe eine morphologische Operation durch, um Rauschen zu entfernen
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Erkenne die Konturen im maskierten Bild
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtere die Konturen basierend auf der Kreisförmigkeit und der Größe
pot_count = 0
for contour in contours:
    # Berechne den Flächeninhalt und den Umfang der Kontur
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Berechne die Kreisförmigkeit
    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Setze eine Schwelle für die Kreisförmigkeit und Größe
        if 0.7 < circularity < 1.3 and area > 500:
            pot_count += 1
            # Zeichne die erkannten Töpfe
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Zeige die Anzahl der erkannten Töpfe an
print(f"Anzahl der erkannten Blumentöpfe: {pot_count}")

# Zeige das resultierende Bild an
cv2.imshow('Detected Pots', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
