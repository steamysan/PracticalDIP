# Sobel Edge Detection Filter on HSI Intensity
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.ndimage
import skimage.data

IMAGE_FILE = "images/sample_cat.png"

def rgb_to_hsi(rgb):
    # convert RGB image channels to HSI color space
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-8
    theta = np.arccos(np.clip(0.5 * ((r - g) + (r - b)) / den, -1.0, 1.0))
    h = np.where(b <= g, theta, 2 * np.pi - theta) / (2 * np.pi)
    s = np.where((r + g + b) < 1e-6, 0.0, 1.0 - 3.0 * np.minimum(np.minimum(r, g), b) / (r + g + b + 1e-8))
    return np.stack([h, s, (r + g + b) / 3.0], axis=-1)

def hsi_to_rgb(hsi):
    # convert HSI channels back to RGB
    h, s, i = hsi[..., 0] * 2 * np.pi, hsi[..., 1], hsi[..., 2]
    sec = np.clip((h // (2 * np.pi / 3)).astype(int), 0, 2)
    ha = h - sec * (2 * np.pi / 3)
    c3 = i * (1 - s)
    c1 = i * (1 + s * np.cos(ha) / (np.cos(np.pi / 3 - ha) + 1e-8))
    c2 = 3 * i - (c1 + c3)
    r = np.select([sec == 0, sec == 1, sec == 2], [c1, c3, c2])
    g = np.select([sec == 0, sec == 1, sec == 2], [c2, c1, c3])
    b = np.select([sec == 0, sec == 1, sec == 2], [c3, c2, c1])
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)

def apply_sobel(im):
    # compute horizontal and vertical Sobel gradients
    w = sliding_window_view(np.pad(im, 1, mode="reflect"), (3, 3))
    gx, gy = np.sum(w * Kx, axis=(-2, -1)), np.sum(w * Ky, axis=(-2, -1))
    return np.sqrt(gx ** 2 + gy ** 2), gx, gy

# check gradient magnitude on a 3x3 vertical edge block
test_mat = np.array([[10, 10, 100], [10, 10, 100], [10, 10, 100]], float)
c, pdf_ans = float(apply_sobel(test_mat)[0][1, 1]), 360.0
print("Sobel Edge Detection Filter (3x3)")
print(f"Gradient magnitude: {c:.4f} | Answer in PDF (Page 8): {pdf_ans:.4f} | Match: {np.isclose(c, pdf_ans)}")

# load image and compute Sobel edge map on HSI intensity
raw = Image.open(IMAGE_FILE).convert("RGB") if IMAGE_FILE else Image.fromarray(skimage.data.astronaut())
rgb = np.asarray(raw.resize((256, 256)), dtype=float) / 255.0
hsi = rgb_to_hsi(rgb)
print(f"RGB->HSI->RGB round-trip error: {np.max(np.abs(rgb - hsi_to_rgb(hsi))):.2e}")
orig_i = hsi[..., 2]
filt_i, gx, gy = apply_sobel(orig_i)
gx_sp = scipy.ndimage.correlate(orig_i, Kx, mode="mirror")
gy_sp = scipy.ndimage.correlate(orig_i, Ky, mode="mirror")
print(f"Matches SciPy: {np.allclose(filt_i, np.sqrt(gx_sp ** 2 + gy_sp ** 2))}")
disp_i = np.abs(filt_i) / (np.max(np.abs(filt_i)) + 1e-8)

# reconstruct edge overlay RGB and display comparison panels
filt_rgb = hsi_to_rgb(np.stack([hsi[..., 0], hsi[..., 1], np.clip(disp_i, 0.0, 1.0)], axis=-1))
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
plots = [(rgb, "Original RGB", None), (orig_i, "Original Intensity (I)", "gray"),
         (gx, "Gx (Horizontal)", "gray"), (gy, "Gy (Vertical)", "gray"),
         (disp_i, "Sobel Edge Map (Scaled)", "gray"), (filt_rgb, "Edge Overlay RGB", None)]
for ax, (img, title, cm) in zip(axes.flat, plots):
    ax.imshow(img, cmap=cm)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/06_sobel_filter.jpg", dpi=150)
try:
    plt.show()
except Exception:
    print("Plot saved as output.png")
