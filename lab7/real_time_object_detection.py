import numpy as np
import time
import cv2
import imutils
import os
from imutils.video import FPS

PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

CONFIDENCE_THRESHOLD = 0.4

VIDEO_FILES = {
    1: "videos/thief_1.mp4",
    2: "videos/thief_2.mp4",
    3: "videos/thief_3.mp4",
    4: "videos/dogs.mp4"
}

print("------------------------------------------------")
print("Оберіть відео для тестування системи безпеки:")
print("1 - Thief 1")
print("2 - Thief 2")
print("3 - Thief 3")
print("4 - Dogs (Негативний тест - тварини)")
print("------------------------------------------------")

try:
    choice = int(input("Введіть номер відео (1-4): "))
    if choice in VIDEO_FILES:
        VIDEO_PATH = VIDEO_FILES[choice]
    else:
        print("[WARNING] Невірний номер. Використовується відео за замовчуванням (1).")
        VIDEO_PATH = VIDEO_FILES[1]
except ValueError:
    print("[WARNING] Введено не число. Використовується відео за замовчуванням (1).")
    VIDEO_PATH = VIDEO_FILES[1]

if not os.path.isfile(VIDEO_PATH):
    print(f"\n[ERROR] Файл '{VIDEO_PATH}' не знайдено!")
    print("Перевірте, чи створили ви папку 'videos' і чи поклали туди файли.")
    exit()


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLORS[15] = [0, 0, 255]

print(f"\n[INFO] Завантаження моделі нейромережі...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

print(f"[INFO] Відкриття відеопотоку: {VIDEO_PATH}...")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Не вдалося відкрити відеофайл.")
    exit()

time.sleep(1.0)
fps = FPS().start()

while True:
    ret, frame = cap.read()

    if not ret:
        print("[INFO] Відео завершено.")
        break

    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    person_detected = False

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_detected = True

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format("PERSON DETECTED", confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


    if person_detected:
        cv2.putText(frame, "!!! SECURITY ALERT: MOTION DETECTED !!!", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Security Feed - Variant 11", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] Роботу завершено.")
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
cap.release()