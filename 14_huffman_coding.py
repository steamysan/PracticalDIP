# Huffman Coding for Lossless Grayscale Image Compression
import heapq
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.data

IMAGE_FILE = "images/sample_cat.png"

def build_huffman(counts):
    # build variable-length prefix codes from symbol frequencies
    heap = [[freq, [sym, ""]] for sym, freq in counts.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][1][0]: "0"}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heap[0][1:], key=lambda x: (len(x[1]), x[0])))

def decode_huffman(bits, codes):
    # decode binary bitstream using prefix lookup table
    rev = {code: sym for sym, code in codes.items()}
    out, buf = [], ""
    for b in bits:
        buf += b
        if buf in rev:
            out.append(rev[buf])
            buf = ""
    return out

# verify Huffman code length and lossless decode on 6-symbol example
ex_p = {"a1": 0.10, "a2": 0.40, "a3": 0.06, "a4": 0.10, "a5": 0.04, "a6": 0.30}
ex_codes = build_huffman(ex_p)
ex_avg = sum(ex_p[k] * len(ex_codes[k]) for k in ex_p)
ex_msg = ["a2", "a6", "a1", "a4", "a3", "a5"]
ex_dec = decode_huffman("".join(ex_codes[s] for s in ex_msg), ex_codes)
print("Huffman Coding (6-Symbol Verification)")
print(f"Avg length: {ex_avg:.4f} | Expected: 2.2000 | Entropy: {-sum(p * np.log2(p) for p in ex_p.values()):.4f}")
print(f"Sample decode lossless: {ex_dec == ex_msg}")

# load 256x256 grayscale image and encode pixel intensities
raw = Image.open(IMAGE_FILE).convert("L") if IMAGE_FILE else Image.fromarray(skimage.data.astronaut()).convert("L")
img = np.asarray(raw.resize((256, 256)), dtype=np.uint8)
vals, cnts = np.unique(img, return_counts=True)
probs = cnts / img.size
codes = build_huffman({int(v): int(c) for v, c in zip(vals, cnts)})
bits = "".join(codes[int(px)] for px in img.ravel())
rec = np.array(decode_huffman(bits, codes), dtype=np.uint8).reshape(img.shape)
ent = float(-np.sum(probs * np.log2(probs)))
print(f"\nImage Compression (256x256, 8-bit grayscale)")
print(f"Raw: {img.size * 8} bits | Compressed: {len(bits)} bits | Ratio: {img.size * 8 / len(bits):.3f}x")
print(f"Avg bpp: {len(bits) / img.size:.4f} | Entropy: {ent:.4f} | Lossless: {np.array_equal(img, rec)}")
# display original, decoded image, probability histogram, and code lengths
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
for ax, im, t in [(axes[0, 0], img, "Original Grayscale (8.00 bpp)"),
                  (axes[0, 1], rec, f"Huffman Decoded ({len(bits) / img.size:.2f} bpp)")]:
    ax.imshow(im, cmap="gray")
    ax.set_title(t, fontsize=9)
    ax.axis("off")
axes[1, 0].bar(vals, probs, width=1.0, color="steelblue")
axes[1, 0].set_title("Gray-Level Probability Distribution", fontsize=9)
axes[1, 1].bar(vals, [len(codes[int(v)]) for v in vals], width=1.0, color="darkorange")
axes[1, 1].set_title("Huffman Codeword Length (bits)", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/14_huffman_coding.jpg", dpi=150)
try:
    plt.show()
except Exception:
    print("Plot saved as output.png")
