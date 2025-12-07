import cv2
import os
import glob
import shutil
from sklearn.cluster import KMeans

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