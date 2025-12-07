import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(image, title, ax):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.plot(hist, color='black')
    ax.set_xlim([0, 256])
    ax.grid(True, linestyle='--', alpha=0.5)


def enhance_contrast_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def apply_advanced_filtering(image):
    # Bilateral Filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(bilateral, -1, kernel)

    return sharpened


def analyze_zabyria_lab3(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Файл не знайдено.")
        return

    contrast_enhanced = enhance_contrast_clahe(image)

    # Bilateral + Sharpening
    filtered = apply_advanced_filtering(contrast_enhanced)

    # HSV сегментація
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([105, 30, 40])  # S підняли з 20 до 30 завдяки CLAHE
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # морфологічна очистка
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Watershed підготовка
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # поріг
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_fg = cv2.erode(sure_fg, kernel, iterations=1)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(filtered, markers)

    output_img = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    count = 0

    for label in np.unique(markers):
        if label <= 1: continue

        mask_temp = np.zeros(markers.shape, dtype="uint8")
        mask_temp[markers == label] = 255

        cnts, _ = cv2.findContours(mask_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 15:
                cv2.drawContours(output_img, [c], -1, (255, 0, 0), 2)
                count += 1

    # --- ВІЗУАЛІЗАЦІЯ ---
    plt.figure(figsize=(14, 10))
    plt.suptitle("Lab 3: Image Enhancement Analysis", fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("1. Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title("2. Filtered & Enhanced (CLAHE + Bilateral)")
    plt.axis('off')

    ax3 = plt.subplot(2, 2, 3)
    plot_histogram(image, "3. Original Histogram", ax3)

    ax4 = plt.subplot(2, 2, 4)
    plot_histogram(filtered, "4. Enhanced Histogram (Wider Range)", ax4)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.imshow(output_img)
    plt.title(f"Final Result: {count} buildings detected", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"------------- Результати -------------")
    print(f"Знайдено об'єктів: {count}")


if __name__ == "__main__":
    analyze_zabyria_lab3('../images/zabyria.png')