import cv2
import numpy as np
import matplotlib.pyplot as plt

# Schritt 1: Bild einlesen
image = cv2.imread(r'S:\Studium\Master\SS24\06_SDU_Drones\Computer_Vision\CV\photo_5319002394492788159_y.jpg')

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

# Schritt 6: Object Filtering (Kreis hinzufügen)
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
        text_x = cX
        text_y = cY + radius + text_height + 10  # Unterhalb des Kreises
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        object_coordinates.append((cX, cY))

# Anzahl der gefundenen Objekte
object_count = len(object_coordinates)
print(f"Anzahl der gefundenen Objekte: {object_count}")
print(f"Koordinaten der Objekte: {object_coordinates}")

# Erstelle eine Collage der Zwischenschritte mit matplotlib
step_images = [
    ('Original Image', image),
    ('Blurred Image', blurred_image),
    ('Color Segmentation', color_segmented_image),
    ('Grayscale Image', gray_image),
    ('Morphological Filtering', morph_filtered_image)
]

# Anzahl der Bilder
num_images = len(step_images)

# Erstelle einen Plot mit einer Reihe von Subplots
fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

for i, (label, img) in enumerate(step_images):
    # Falls das Bild ein Einzelkanal-Bild ist, in RGB umwandeln
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Bild in den entsprechenden Subplot einfügen
    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(label)
    axes[i].axis('off')  # Achsen ausblenden

# Zeige den Plot an
plt.show()

# Visualisiere die Objektskoordinaten in einem separaten Plot
if object_coordinates:
    data = np.array(object_coordinates)
    x, y = data.T
    plt.scatter(x, y)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title("Object Positions")
    plt.gca().invert_yaxis()  # Y-Achse invertieren, um Bildkoordinaten zu entsprechen
    plt.show()
