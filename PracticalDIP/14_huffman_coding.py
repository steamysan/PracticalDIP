
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.ndimage
from pathlib import Path

IMAGE_FILE = "images/IRON.jpeg"

def rgb_to_hsi(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    den = np.sqrt((r-g)**2 + (r-b)*(g-b)) + 1e-8
    theta = np.arccos(np.clip(0.5*((r-g)+(r-b))/den, -1.0, 1.0))
    h = np.where(b <= g, theta, 2*np.pi-theta)/(2*np.pi)
    s = np.where(r+g+b < 1e-6, 0.0, 1.0-3*np.minimum(np.minimum(r,g),b)/(r+g+b+1e-8))
    return np.stack([h,s,(r+g+b)/3], axis=-1)

def hsi_to_rgb(hsi):
    h,s,i = hsi[...,0]*2*np.pi, hsi[...,1], hsi[...,2]
    sec=np.clip((h//(2*np.pi/3)).astype(int),0,2)
    ha=h-sec*(2*np.pi/3)
    c3=i*(1-s)
    c1=i*(1+s*np.cos(ha)/(np.cos(np.pi/3-ha)+1e-8))
    c2=3*i-(c1+c3)
    r=np.select([sec==0,sec==1,sec==2],[c1,c3,c2])
    g=np.select([sec==0,sec==1,sec==2],[c2,c1,c3])
    b=np.select([sec==0,sec==1,sec==2],[c3,c2,c1])
    return np.clip(np.stack([r,g,b],axis=-1),0,1)

def load_image(gray=False):
    im=Image.open(IMAGE_FILE).convert("L" if gray else "RGB").resize((256,256))
    a=np.asarray(im)
    return a.astype(float)/(255 if a.ndim==2 else 255)

def save_panels(name, plots, size=(10,6.5)):
    fig,axes=plt.subplots(2,3,figsize=size)
    for ax,(im,title,cm) in zip(axes.flat,plots):
        ax.imshow(im,cmap=cm)
        ax.set_title(title,fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(f"outputs/{name}.png",dpi=140)
    plt.close(fig)

# Huffman Coding for Lossless Grayscale Image Compression
import heapq
def build_huffman(counts):
    heap=[[freq,[[sym,""]]] for sym,freq in counts.items()]
    heapq.heapify(heap)
    if len(heap)==1: return {heap[0][1][0][0]:"0"}
    while len(heap)>1:
        lo=heapq.heappop(heap); hi=heapq.heappop(heap)
        for pair in lo[1]: pair[1]="0"+pair[1]
        for pair in hi[1]: pair[1]="1"+pair[1]
        heapq.heappush(heap,[lo[0]+hi[0],lo[1]+hi[1]])
    return {s:c for s,c in heap[0][1]}
def decode_huffman(bits,codes):
    rev={c:s for s,c in codes.items()}; out=[]; buf=""
    for b in bits:
        buf+=b
        if buf in rev: out.append(rev[buf]); buf=""
    return out
raw=load_image(gray=True)
img=np.asarray(np.round(raw*255),dtype=np.uint8)
vals,cnts=np.unique(img,return_counts=True)
codes=build_huffman({int(v):int(c) for v,c in zip(vals,cnts)})
bits="".join(codes[int(px)] for px in img.ravel())
rec=np.asarray(decode_huffman(bits,codes),dtype=np.uint8).reshape(img.shape)
probs=cnts/cnts.sum()
print("Raw bits:",img.size*8,"Encoded bits:",len(bits),"Compression ratio:",round(img.size*8/len(bits),3))
print("Lossless reconstruction:",np.array_equal(img,rec))
fig,axes=plt.subplots(2,2,figsize=(9,6))
axes[0,0].imshow(img,cmap="gray"); axes[0,0].set_title("Original grayscale")
axes[0,1].imshow(rec,cmap="gray"); axes[0,1].set_title("Huffman decoded (lossless)")
axes[1,0].bar(vals,probs,width=1); axes[1,0].set_title("Gray-level probability distribution")
axes[1,1].bar(vals,[len(codes[int(v)]) for v in vals],width=1); axes[1,1].set_title("Huffman code lengths")
for ax in axes.flat: ax.axis("off") if ax in (axes[0,0],axes[0,1]) else None
plt.tight_layout(); plt.savefig("outputs/14_huffman_coding.png",dpi=140); plt.close(fig)
