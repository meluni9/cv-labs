import cv2

def create_tracker_by_name(tracker_type):
    try:
        if tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.TrackerMedianFlow_create()
        elif tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()

    print(f"Трекер {tracker_type} не знайдено!")
    return None


def run_tracker(tracker_type, video_source):
    tracker = create_tracker_by_name(tracker_type)
    if not tracker:
        return

    video = cv2.VideoCapture(video_source)
    if not video.isOpened():
        print("Не вдалося відкрити відео/камеру")
        return

    ok, frame = video.read()
    if not ok:
        print('Неможливо прочитати відео файл')
        return

    print("1. Виділіть об'єкт мишкою.")
    print("2. Натисніть SPACE або ENTER для початку трекінгу.")
    print("3. Натисніть ESC, щоб зупинити і повернутися в меню.")

    bbox = cv2.selectROI(f"Tracking: {tracker_type}", frame, False)
    # cv2.destroyWindow(f"Tracking: {tracker_type}")

    if bbox == (0, 0, 0, 0):
        print("Область не обрана.")
        return

    ok = tracker.init(frame, bbox)

    fps_history = []

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)

        # Розрахунок FPS
        curr_fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps_history.append(curr_fps)

        # Логіка відображення
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            status_text = "Tracking"
            color = (0, 255, 0)  # Зелений
        else:
            status_text = "Lost (Failure)"
            color = (0, 0, 255)  # Червоний


        cv2.putText(frame, f"{tracker_type} Tracker", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, f"FPS: {int(curr_fps)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, f"Status: {status_text}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow(f"Tracking: {tracker_type}", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:  # ESC
            break

    video.release()
    cv2.destroyAllWindows()

    # --- АНАЛІЗ РЕЗУЛЬТАТІВ ---
    if len(fps_history) > 0:
        avg_fps = sum(fps_history) / len(fps_history)
        print(f"\n>>> РЕЗУЛЬТАТИ ДЛЯ {tracker_type}:")
        print(f">>> Середній FPS: {avg_fps:.2f}")
        print(f">>> Оброблено кадрів: {len(fps_history)}")
    else:
        print("\n>>> Недостатньо даних для статистики.")
    print("-" * 30)

if __name__ == '__main__':
    # шлях до файлу або 0 для веб-камери
    # video_path = "video.mp4"
    video_path = 0

    available_trackers = [
        'CSRT',
        'MOSSE',
        'KCF',
        'MEDIANFLOW'
    ]

    while True:
        print("\n=== МЕНЮ ВИБОРУ ТРЕКЕРА ===")
        print("0. Вихід")
        for idx, name in enumerate(available_trackers):
            print(f"{idx + 1}. {name}")

        choice = input("\nВведіть номер трекера: ")

        if choice == '0':
            print("Вихід з програми.")
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_trackers):
                selected_tracker = available_trackers[idx]
                print(f"\nЗапуск {selected_tracker}...")
                run_tracker(selected_tracker, video_path)
            else:
                print("Невірний номер. Спробуйте ще раз.")
        except ValueError:
            print("Будь ласка, введіть число.")
