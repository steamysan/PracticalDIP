# 5x5 Gaussian Spatial Filter on HSI Intensity
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

xx, yy = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
K = np.exp(-(xx ** 2 + yy ** 2) / 2.0) / np.sum(np.exp(-(xx ** 2 + yy ** 2) / 2.0))

def apply_filter(im):
    # apply 5x5 Gaussian kernel with pad width 2
    return np.sum(sliding_window_view(np.pad(im, 2, mode="reflect"), (5, 5)) * K, axis=(-2, -1))

# verify center pixel output on a 5x5 test block
test_mat = np.array([[10, 10, 10, 10, 10], [10, 20, 20, 20, 10], [10, 20, 50, 20, 10],
                     [10, 20, 20, 20, 10], [10, 10, 10, 10, 10]], float)
c, exp_ans = float(apply_filter(test_mat)[2, 2]), 22.8023
print("Gaussian Spatial Filter (5x5, sigma=1.0)")
print(f"Center value: {c:.4f} | Expected (hand calculation): {exp_ans:.4f} | Match: {np.isclose(c, exp_ans)}")

# load image, add Gaussian noise (std=0.05), and filter intensity
raw = Image.open(IMAGE_FILE).convert("RGB") if IMAGE_FILE else Image.fromarray(skimage.data.astronaut())
rgb = np.asarray(raw.resize((256, 256)), dtype=float) / 255.0
hsi = rgb_to_hsi(rgb)
print(f"RGB->HSI->RGB round-trip error: {np.max(np.abs(rgb - hsi_to_rgb(hsi))):.2e}")
np.random.seed(42)
orig_i = hsi[..., 2]
noisy_i = np.clip(orig_i + np.random.normal(0.0, 0.05, orig_i.shape), 0.0, 1.0)
filt_i = apply_filter(noisy_i)
sp_f = scipy.ndimage.convolve(noisy_i, K, mode="mirror")
print(f"Matches SciPy: {np.allclose(filt_i, sp_f)}")
pn, pf = [10.0 * np.log10(1.0 / np.mean((orig_i - x) ** 2)) for x in (noisy_i, filt_i)]
print(f"PSNR vs original: noisy {pn:.1f} dB, filtered {pf:.1f} dB")

# reconstruct RGB and display comparison panels
filt_rgb = hsi_to_rgb(np.stack([hsi[..., 0], hsi[..., 1], filt_i], axis=-1))
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
plots = [(rgb, "Original RGB", None), (orig_i, "Original I", "gray"), (noisy_i, "Noisy I", "gray"),
         (filt_i, "Filtered I", "gray"), (filt_rgb, "Filtered RGB", None), (K, "5x5 Gaussian Kernel", "Blues")]
for ax, (img, title, cm) in zip(axes.flat, plots):
    ax.imshow(img, cmap=cm)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/03_gaussian_filter.jpg", dpi=150)
try:
    plt.show()
except Exception:
    print("Plot saved as output.png")
