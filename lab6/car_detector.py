import cv2

car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

if car_cascade.empty():
    raise IOError('Не вдалося завантажити файл каскадного класифікатора для автомобілів')

VIDEO_PATH = 'video/british_highway_traffic.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise IOError('Не вдалося відкрити відеофайл або камеру')

SCALING_FACTOR = 0.5
MIN_NEIGHBORS = 2
MIN_SIZE = (30, 30)
MAX_SIZE = (400, 400)

frame_count = 0

print("Початок обробки відео...")
print("Натисніть 'Esc' для виходу, 'Space' для паузи")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Кінець відео або помилка читання")
        break

    frame_count += 1

    frame_resized = cv2.resize(frame, None,
                               fx=SCALING_FACTOR,
                               fy=SCALING_FACTOR,
                               interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=MIN_NEIGHBORS,
                                        minSize=MIN_SIZE,
                                        maxSize=MAX_SIZE)

    cars_in_frame = len(cars)


    for i, (x, y, w, h) in enumerate(cars):
        x_orig = int(x / SCALING_FACTOR)
        y_orig = int(y / SCALING_FACTOR)
        w_orig = int(w / SCALING_FACTOR)
        h_orig = int(h / SCALING_FACTOR)

        cv2.rectangle(frame,
                      (x_orig, y_orig),
                      (x_orig + w_orig, y_orig + h_orig),
                      (0, 255, 0),
                      3)

        label = f'Car {i + 1}'
        cv2.putText(frame, label,
                    (x_orig, y_orig - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    info_text = [
        f'Frame: {frame_count}',
        f'Cars in frame: {cars_in_frame}',
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    y_offset = 35
    for text in info_text:
        cv2.putText(frame, text,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)
        y_offset += 30


    cv2.imshow('Car Detection - Haar Cascades', frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC - вихід
    if key == 27:
        print("Зупинка обробки...")
        break

    # Space - пауза
    elif key == 32:
        print("Пауза. Натисніть будь-яку клавішу для продовження...")
        cv2.waitKey(0)


print("\n" + "=" * 50)
print(f"Оброблено кадрів: {frame_count}")
print("=" * 50)

cap.release()
cv2.destroyAllWindows()

print("\nРобота завершена успішно!")
