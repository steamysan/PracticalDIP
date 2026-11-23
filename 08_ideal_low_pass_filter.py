# Ideal Low-Pass Filter (ILPF) in Frequency Domain
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import skimage.data

IMAGE_FILE = "images/sample_cat.png"
D0 = 40.0

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

def transfer_fn(d, d0):
    # compute frequency transfer function H(u, v)
    return (d <= d0).astype(float)

# check transfer function values on sample distances D = [0, 1, 2, 3]
test_d = np.array([0.0, 1.0, 2.0, 3.0])
c_vals = transfer_fn(test_d, 2.0)
exp_vals = np.array([1.0 if d <= 2.0 else 0.0 for d in test_d], float)
c_s, e_s = ", ".join(f"{v:.4f}" for v in c_vals), ", ".join(f"{v:.4f}" for v in exp_vals)
print("Ideal Low-Pass Filter (ILPF)")
print(f"H(D) sample: [{c_s}]")
print(f"Expected (formula): [{e_s}] | Match: {np.allclose(c_vals, exp_vals)}")

# load image and extract HSI intensity channel
raw = Image.open(IMAGE_FILE).convert("RGB") if IMAGE_FILE else Image.fromarray(skimage.data.astronaut())
rgb = np.asarray(raw.resize((256, 256)), dtype=float) / 255.0
hsi = rgb_to_hsi(rgb)
print(f"RGB->HSI->RGB round-trip error: {np.max(np.abs(rgb - hsi_to_rgb(hsi))):.2e}")
orig_i = hsi[..., 2]
# construct 2D frequency grid and apply filter
U, V = np.meshgrid(np.arange(256) - 128, np.arange(256) - 128)
H = transfer_fn(np.sqrt(U ** 2 + V ** 2), D0)
F = np.fft.fftshift(np.fft.fft2(orig_i))
filt_i = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
F_sp = scipy.fft.fftshift(scipy.fft.fft2(orig_i))
filt_sp = np.real(scipy.fft.ifft2(scipy.fft.ifftshift(F_sp * H)))
print(f"Matches SciPy: {np.allclose(filt_i, filt_sp)}")
disp_i = filt_i
# reconstruct RGB and display comparison panels
filt_rgb = hsi_to_rgb(np.stack([hsi[..., 0], hsi[..., 1], filt_i], axis=-1))
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
plots = [(rgb, "Original RGB", None), (orig_i, "Original Intensity (I)", "gray"), (np.log1p(np.abs(F)), "FFT Spectrum", "viridis"),
         (disp_i, "Filtered Intensity", "gray"), (filt_rgb, "Filtered RGB", None), (H, f"|H(u,v)| Cut-off D0={int(D0)}", "magma")]
for ax, (img, label, cm) in zip(axes.flat, plots):
    ax.imshow(img, cmap=cm)
    ax.set_title(label, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/08_ideal_low_pass_filter.jpg", dpi=150)
try:
    plt.show()
except Exception:
    print("Plot saved as output.png")
