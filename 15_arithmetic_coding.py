# Arithmetic Coding for Lossless Image Compression
from decimal import Decimal, getcontext
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.data

getcontext().prec = 100
IMAGE_FILE = "images/sample_cat.png"

def build_cdf(probs):
    # build cumulative probability intervals [low, high)
    cdf, cur, syms = {}, Decimal(0), list(probs)
    for s in syms:
        nxt = Decimal(1) if s == syms[-1] else cur + probs[s]
        cdf[s], cur = (cur, nxt), nxt
    return cdf

def arith_encode(seq, cdf):
    # narrow interval [lo, hi) per symbol and return midpoint tag
    lo, hi = Decimal(0), Decimal(1)
    for s in seq:
        lo, hi = lo + (hi - lo) * cdf[s][0], lo + (hi - lo) * cdf[s][1]
    return Decimal("0.5") * (lo + hi), lo, hi

def arith_decode(tag, length, cdf):
    # decode fixed-length sequence from scalar tag in [0, 1)
    lo, hi, out, last_s = Decimal(0), Decimal(1), [], list(cdf)[-1]
    for _ in range(length):
        sc = (tag - lo) / (hi - lo)
        for s, (s_lo, s_hi) in cdf.items():
            if s_lo <= sc < s_hi or s == last_s:
                out.append(s)
                lo, hi = lo + (hi - lo) * s_lo, lo + (hi - lo) * s_hi
                break
    return out

# verify interval narrowing on sample sequence "a1 a2 a3 a3 a4"
ex_cdf = build_cdf({k: Decimal(v) for k, v in [("a1", "0.2"), ("a2", "0.2"), ("a3", "0.4"), ("a4", "0.2")]})
ex_seq = ["a1", "a2", "a3", "a3", "a4"]
tag, lo, hi = arith_encode(ex_seq, ex_cdf)
print("Arithmetic Coding (5-Symbol Verification)")
print(f"Interval: [{lo:.5f}, {hi:.5f}) | Tag: {tag:.5f}")
print(f"Expected: [0.06752, 0.06880) | Lossless: {arith_decode(tag, len(ex_seq), ex_cdf) == ex_seq}")

# load 256x256 quantized grayscale image (16 levels) and encode in 32-px blocks
raw = Image.open(IMAGE_FILE).convert("L") if IMAGE_FILE else Image.fromarray(skimage.data.astronaut()).convert("L")
q_img = np.asarray(raw.resize((256, 256)), dtype=np.uint8) // 16
vals, cnts = np.unique(q_img, return_counts=True)
cdf = build_cdf({int(v): Decimal(int(c)) / Decimal(int(q_img.size)) for v, c in zip(vals, cnts)})
flat, blk = q_img.ravel().tolist(), 32
enc = [arith_encode(flat[i:i + blk], cdf) for i in range(0, len(flat), blk)]
rec = np.array([s for t, _, _ in enc for s in arith_decode(t, blk, cdf)], dtype=np.uint8).reshape(q_img.shape)
blk_bits = [int(np.ceil(-float((b_hi - b_lo).ln() / Decimal(2).ln()))) + 1 for _, b_lo, b_hi in enc]
comp_bits, probs = sum(blk_bits), cnts / q_img.size
print(f"\nImage Compression (256x256, 16-level, block={blk})")
print(f"Raw (4-bit): {q_img.size * 4} bits | Arith bits: {comp_bits} | Ratio: {q_img.size * 4 / comp_bits:.3f}x")
print(f"Avg bpp: {comp_bits / q_img.size:.4f} | Entropy: {float(-np.sum(probs * np.log2(probs))):.4f} | Lossless: {np.array_equal(q_img, rec)}")
# display original quantized image, decoded image, histogram, and block bits
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
for ax, im, t in [(axes[0, 0], q_img, "Quantized Grayscale (4.00 bpp)"),
                  (axes[0, 1], rec, f"Arithmetic Decoded ({comp_bits / q_img.size:.2f} bpp)")]:
    ax.imshow(im, cmap="gray", vmin=0, vmax=15)
    ax.set_title(t, fontsize=9)
    ax.axis("off")
axes[1, 0].bar(vals, probs, color="steelblue")
axes[1, 0].set_title("16-Level Probability Distribution", fontsize=9)
axes[1, 1].hist(blk_bits, bins=18, color="seagreen")
axes[1, 1].set_title("Encoded Bits per 32-Pixel Block", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/15_arithmetic_coding.jpg", dpi=150)
try:
    plt.show()
except Exception:
    print("Plot saved as output.png")
