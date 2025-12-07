import cv2
import os
import glob
import shutil
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

source_dir = "dataset"
output_dir = "clustered_output_cv"
n_clusters = 3


def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None: return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    return hist.flatten()


def show_clusters(n_clusters, output_dir, max_per_cluster=200, num_cols=8):
    for cluster_id in range(n_clusters):
        cluster_path = os.path.join(output_dir, f"Cluster_{cluster_id}")
        images = glob.glob(os.path.join(cluster_path, "*.jpg"))

        if len(images) == 0:
            continue

        selected_images = images[:max_per_cluster]

        num_samples = len(selected_images)
        num_rows = math.ceil(num_samples / num_cols)

        plt.figure(figsize=(num_cols * 4, num_rows * 4))
        plt.suptitle(f"Cluster {cluster_id}", fontsize=18)

        for idx, img_path in enumerate(selected_images):
            clustered_img = cv2.imread(img_path)
            clustered_img = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)

            row = idx // num_cols
            col = idx % num_cols

            base_index = row * num_cols + col + 1

            ax2 = plt.subplot(num_rows, num_cols, base_index)
            ax2.imshow(clustered_img)
            ax2.axis("off")

        plt.show()


def main():
    filelist = glob.glob(os.path.join(source_dir, '*.*'))
    filelist = [f for f in filelist if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    featurelist = []
    valid_files = []

    print("Виділення колірних ознак (HSV Histograms)...")
    for file in filelist:
        features = extract_color_histogram(file)
        if features is not None:
            featurelist.append(features)
            valid_files.append(file)

    print("Кластеризація...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(featurelist)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for i in range(n_clusters):
        os.makedirs(os.path.join(output_dir, f"Cluster_{i}"))

    for i, label in enumerate(kmeans.labels_):
        src = valid_files[i]
        dst = os.path.join(output_dir, f"Cluster_{label}", os.path.basename(src))
        shutil.copy(src, dst)

    print(f"Готово! Результат у {output_dir}")


if __name__ == "__main__":
    main()
    show_clusters(n_clusters, output_dir)
