import cv2
import numpy as np
import matplotlib.pyplot as plt


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def analyze_zabyria_contours(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Файл не знайдено.")
        return

    # різкість + розмиття шуму
    sharpened = sharpen_image(image)
    blurred = cv2.medianBlur(sharpened, 5)

    # HSV сегментація (фіолетовий/синій)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([110, 40, 50])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # очистка маски
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Watershed (розділення об'єктів)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_fg = cv2.erode(sure_fg, kernel, iterations=1)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    output_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    count = 0

    for label in np.unique(markers):
        if label <= 1: continue

        mask_temp = np.zeros(markers.shape, dtype="uint8")
        mask_temp[markers == label] = 255

        cnts, _ = cv2.findContours(mask_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 15:
                cv2.drawContours(output_img, [c], -1, (255, 0, 0), 2)  # Червоний контур, товщина 2
                count += 1

    # --- ВІЗУАЛІЗАЦІЯ ---
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.title("1. Cleaned Mask (Merged blobs visible)")
    plt.imshow(opening, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("2. Sure Foreground (Centers separated)")
    plt.imshow(sure_fg, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("3. Watershed Markers (Separation Logic)")
    plt.imshow(markers, cmap='jet')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title(f"4. Result: {count} buildings")
    plt.imshow(output_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"------------- Результати -------------")
    print(f"Знайдено об'єктів: {count}")


if __name__ == "__main__":
    analyze_zabyria_contours('../images/zabyria.png')
