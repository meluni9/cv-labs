import numpy as np
import cv2
from matplotlib import pyplot as plt

IMG_LEFT_PATH = 'images/view0.png'   # ліве зображення
IMG_RIGHT_PATH = 'images/view1.png'  # праве зображення
OUTPUT_FILE = 'reconstruction.ply'

print("[INFO] Завантаження зображень...")
imgL = cv2.imread(IMG_LEFT_PATH)
imgR = cv2.imread(IMG_RIGHT_PATH)

if imgL is None or imgR is None:
    print(f"[ERROR] Не знайдено зображення. Перевірте файли {IMG_LEFT_PATH} та {IMG_RIGHT_PATH}")
    exit()

imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

window_size = 3
min_disp = 16
num_disp = 112 - min_disp

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=16,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000.0)
wls_filter.setSigmaColor(1.5)

print("[INFO] Обчислення карти глибини...")
displ = left_matcher.compute(imgL_gray, imgR_gray)
dispr = right_matcher.compute(imgR_gray, imgL_gray)

filtered_disp = wls_filter.filter(displ, imgL, disparity_map_right=dispr)

disparity = filtered_disp.astype(np.float32) / 16.0

h, w = imgL.shape[:2]
f = 0.8 * w
Q = np.float32([[1, 0, 0, -0.5 * w],
                [0, -1, 0, 0.5 * h],
                [0, 0, 0, -f],
                [0, 0, 1, 0]])

print("[INFO] Генерація 3D хмари точок...")
points = cv2.reprojectImageTo3D(disparity, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

mask = disparity > disparity.min()
out_points = points[mask]
out_colors = colors[mask]

def write_ply(fn, verts, colors):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

write_ply(OUTPUT_FILE, out_points, out_colors)
print(f"[SUCCESS] 3D хмару збережено у файл: {OUTPUT_FILE}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Ліве зображення')
plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Карта глибини (Disparity Map)')
plt.imshow(disparity, cmap='plasma')
plt.colorbar()
plt.axis('off')

plt.show()

try:
    import open3d as o3d
    print("[INFO] Спроба відкрити візуалізацію Open3D...")
    pcd = o3d.io.read_point_cloud(OUTPUT_FILE)
    o3d.visualization.draw_geometries([pcd], window_name="3D Reconstruction", width=800, height=600)
except ImportError:
    print("[INFO] Бібліотеку open3d не знайдено. Відкрийте файл .ply у MeshLab або онлайн-в'ювері.")
except Exception as e:
    print(f"[WARNING] Помилка Open3D: {e}")