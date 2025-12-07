import cv2
import os
import shutil

input_image_path = "../raw_image/sat_img.png"
output_dir = "../dataset"
rows = 14
cols = 19


def slice_grid_image():
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Помилка: Не знайдено файл {input_image_path}")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    height, width, _ = img.shape

    tile_h = height // rows
    tile_w = width // cols

    print(f"Розмір зображення: {width}x{height}")
    print(f"Розмір однієї плитки: {tile_w}x{tile_h}")

    count = 0
    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_h
            y2 = (r + 1) * tile_h
            x1 = c * tile_w
            x2 = (c + 1) * tile_w

            tile = img[y1:y2, x1:x2]

            filename = os.path.join(output_dir, f"tile_{count:03d}.jpg")
            cv2.imwrite(filename, tile)
            count += 1

    print(f"Збережено {count} зображень у папку '{output_dir}'.")


if __name__ == "__main__":
    slice_grid_image()