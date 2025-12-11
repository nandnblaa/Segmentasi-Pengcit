import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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


path_original = r"C:\Users\User\OneDrive\Documents\Phyton\kucing.jpeg"
img_gray = load_image(path_original)

if img_gray is None:
    print("\n!!! PROGRAM DIHENTIKAN: Gambar tidak terbaca !!!")
    exit()

def add_gaussian_noise(img, std=15):
    noise = np.random.normal(0, std, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, prob=0.05):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape)
    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255
    return noisy


img_gauss = add_gaussian_noise(img_gray, std=15)
img_sp = add_salt_pepper_noise(img_gray, prob=0.05)

def to_float(img):
    return img.astype(np.float64)

def roberts_operator(img):
    img_f = to_float(img)
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    gx = cv2.filter2D(img_f, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(img_f, cv2.CV_64F, kernel_y)
    return np.sqrt(gx**2 + gy**2)

def prewitt_operator(img):
    img_f = to_float(img)
    kernel_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float64)
    kernel_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float64)
    gx = cv2.filter2D(img_f, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(img_f, cv2.CV_64F, kernel_y)
    return np.sqrt(gx**2 + gy**2)

def sobel_operator(img):
    img_f = to_float(img)
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)

def frei_chen_operator(img):
    img_f = to_float(img)
    sqrt2 = np.sqrt(2.0)
    Gx = np.array([[-1,0,1],[-sqrt2,0,sqrt2],[-1,0,1]], dtype=np.float64)
    Gy = np.array([[-1,-sqrt2,-1],[0,0,0],[1,sqrt2,1]], dtype=np.float64)
    gx = cv2.filter2D(img_f, cv2.CV_64F, Gx)
    gy = cv2.filter2D(img_f, cv2.CV_64F, Gy)
    return np.sqrt(gx**2 + gy**2)

def normalize(img):
    img = np.abs(img)
    maxv = img.max()
    if maxv == 0:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img / maxv) * 255
    return out.astype(np.uint8)

def show_method(title, method_func):
    out_gauss = normalize(method_func(img_gauss))
    out_sp = normalize(method_func(img_sp))

    plt.figure(figsize=(10,5))
    plt.suptitle(title, fontsize=14)

    plt.subplot(1,2,1)
    plt.imshow(out_gauss, cmap='gray')
    plt.title("Grayscale Gaussian (std = 15)")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(out_sp, cmap='gray')
    plt.title("Grayscale Salt & Pepper (0.05)")
    plt.axis("off")

    plt.show(block=True)

print("\n=== MENJALANKAN METODE DETEKSI TEPI ===\n")

show_method("Roberts Operator", roberts_operator)
show_method("Prewitt Operator", prewitt_operator)
show_method("Sobel Operator", sobel_operator)
show_method("Frei-Chen Operator", frei_chen_operator)

print("Selesai.")