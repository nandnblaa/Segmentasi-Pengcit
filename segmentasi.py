# ============================
#   IMPORT LIBRARY
# ============================
import matplotlib
# pilih backend yang stabil di sistemmu, pakai 'TkAgg' atau 'Qt5Agg'
matplotlib.use('TkAgg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================
#   FUNSI BACA CITRA DENGAN CEK ERROR
# ============================
def load_image(path):
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[ERROR] Gagal membaca file: {path}")
    else:
        print(f"[OK] Berhasil load: {path}  | shape = {img.shape}")

    return img

# ============================
#   GANTI 4 PATH GAMBAR DI SINI !!!
# ============================
path_gray = r"C:\Users\User\OneDrive\Documents\Phyton\grayscale_kucing.png"
path_sp   = r"C:\Users\User\OneDrive\Documents\Phyton\salt and pepper.png"
path_gauss = r"C:\Users\User\OneDrive\Documents\Phyton\gaussian kucing.png"
path_original = r"C:\Users\User\OneDrive\Documents\Phyton\kucing.jpeg"

# ============================
#   LOAD CITRA
# ============================
img_gray = load_image(path_gray)
img_sp   = load_image(path_sp)
img_gauss = load_image(path_gauss)
img_original = load_image(path_original)

# STOP JIKA GAMBAR TIDAK TERBACA
if any(img is None for img in [img_gray, img_sp, img_gauss, img_original]):
    print("\n!!! PROGRAM DIHENTIKAN: Ada gambar yang tidak terbaca !!!")
    exit()

# helper: pastikan input jadi float64 sebelum filter untuk menghindari wrap-around
def to_float(img):
    return img.astype(np.float64)

# ============================
#   OPERATOR ROBERTS (ddepth CV_64F)
# ============================
def roberts_operator(img):
    img_f = to_float(img)
    kernel_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float64)

    gx = cv2.filter2D(img_f, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(img_f, cv2.CV_64F, kernel_y)
    grad = np.sqrt(gx**2 + gy**2)
    return grad

# ============================
#   OPERATOR PREWITT
# ============================
def prewitt_operator(img):
    img_f = to_float(img)
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float64)

    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]], dtype=np.float64)

    gx = cv2.filter2D(img_f, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(img_f, cv2.CV_64F, kernel_y)
    grad = np.sqrt(gx**2 + gy**2)
    return grad

# ============================
#   OPERATOR SOBEL (sudah memakai CV_64F)
# ============================
def sobel_operator(img):
    img_f = to_float(img)
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return grad

# ============================
#   OPERATOR FREI-CHEN
# ============================
def frei_chen_operator(img):
    img_f = to_float(img)
    sqrt2 = np.sqrt(2.0)
    Gx = np.array([[-1, 0, 1],
                   [-sqrt2, 0, sqrt2],
                   [-1, 0, 1]], dtype=np.float64)
    Gy = np.array([[-1, -sqrt2, -1],
                   [ 0,     0,   0],
                   [ 1,  sqrt2,  1]], dtype=np.float64)

    gx = cv2.filter2D(img_f, cv2.CV_64F, Gx)
    gy = cv2.filter2D(img_f, cv2.CV_64F, Gy)
    grad = np.sqrt(gx**2 + gy**2)
    return grad

# ============================
#   NORMALISASI GRADIENT
# ============================
def normalize(img):
    img = np.abs(img)
    maxv = img.max()
    if maxv == 0 or np.isnan(maxv):
        return np.zeros_like(img, dtype=np.uint8)
    out = (img / maxv) * 255.0
    return out.astype(np.uint8)

# ============================
#   TAMPILKAN HASIL (1 figure per call)
# ============================
def show_results(title, img):
    rob = normalize(roberts_operator(img))
    pre = normalize(prewitt_operator(img))
    sob = normalize(sobel_operator(img))
    frei = normalize(frei_chen_operator(img))

    plt.figure(figsize=(12,8))
    plt.suptitle(title, fontsize=14)

    plt.subplot(2,2,1); plt.imshow(rob, cmap='gray', vmin=0, vmax=255); plt.title("Roberts"); plt.axis('off')
    plt.subplot(2,2,2); plt.imshow(pre, cmap='gray', vmin=0, vmax=255); plt.title("Prewitt"); plt.axis('off')
    plt.subplot(2,2,3); plt.imshow(sob, cmap='gray', vmin=0, vmax=255); plt.title("Sobel"); plt.axis('off')
    plt.subplot(2,2,4); plt.imshow(frei, cmap='gray', vmin=0, vmax=255); plt.title("Frei-Chen"); plt.axis('off')

    # show with blocking so window stays until you close it
    plt.show(block=True)

# ============================
#   JALANKAN SEMUA
# ============================
print("Menjalankan segmentasi: grayscale")
show_results("Segmentasi - Citra Grayscale", img_gray)

print("Menjalankan segmentasi: salt & pepper")
show_results("Segmentasi - Salt & Pepper", img_sp)

print("Menjalankan segmentasi: gaussian")
show_results("Segmentasi - Gaussian", img_gauss)

print("Menjalankan segmentasi: original")
show_results("Segmentasi - Citra Asli", img_original)

print("Selesai.")