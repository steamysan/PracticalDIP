
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

# Arithmetic Coding for Lossless 4-bit Quantized Grayscale Image Compression
from decimal import Decimal, getcontext
getcontext().prec=100
def build_cdf(probs):
    cdf={}; cur=Decimal(0); syms=list(probs)
    for s in syms:
        nxt=Decimal(1) if s==syms[-1] else cur+probs[s]
        cdf[s]=(cur,nxt); cur=nxt
    return cdf
def encode(seq,cdf):
    lo,hi=Decimal(0),Decimal(1)
    for s in seq:
        width=hi-lo; a,b=cdf[s]
        lo,hi=lo+width*a,lo+width*b
    return (lo+hi)/2,lo,hi
def decode(tag,length,cdf):
    lo,hi=Decimal(0),Decimal(1); out=[]; last=list(cdf)[-1]
    for _ in range(length):
        scaled=(tag-lo)/(hi-lo)
        for s,(a,b) in cdf.items():
            if a<=scaled<b or s==last:
                out.append(s); width=hi-lo; lo,hi=lo+width*a,lo+width*b; break
    return out
img=np.asarray(np.round(load_image(gray=True)*255),dtype=np.uint8)//16
vals,cnts=np.unique(img,return_counts=True)
cdf=build_cdf({int(v):Decimal(int(c))/Decimal(img.size) for v,c in zip(vals,cnts)})
flat=img.ravel().tolist(); block=32; enc=[]
for i in range(0,len(flat),block): enc.append(encode(flat[i:i+block],cdf))
rec=np.asarray([s for tag,lo,hi in enc for s in decode(tag, min(block,len(flat)) if False else (block if (len(rec) if False else True) else block), cdf)],dtype=np.uint8) if False else None
decoded=[]
for i,(tag,lo,hi) in enumerate(enc):
    n=min(block,len(flat)-i*block); decoded.extend(decode(tag,n,cdf))
rec=np.asarray(decoded,dtype=np.uint8).reshape(img.shape)
block_bits=[int(np.ceil(float(-(hi-lo).ln()/Decimal(2).ln())))+1 for _,lo,hi in enc]
total=sum(block_bits); probs=cnts/img.size
print("Raw bits:",img.size*4,"Estimated arithmetic bits:",total,"Ratio:",round(img.size*4/total,3))
print("Lossless reconstruction:",np.array_equal(img,rec))
fig,axes=plt.subplots(2,2,figsize=(9,6))
axes[0,0].imshow(img,cmap="gray",vmin=0,vmax=15); axes[0,0].set_title("Original quantized grayscale")
axes[0,1].imshow(rec,cmap="gray",vmin=0,vmax=15); axes[0,1].set_title("Arithmetic decoded (lossless)")
axes[1,0].bar(vals,probs); axes[1,0].set_title("16-level probability distribution")
axes[1,1].hist(block_bits,bins=18); axes[1,1].set_title("Estimated bits per 32-pixel block")
for ax in axes.flat[:2]: ax.axis("off")
plt.tight_layout(); plt.savefig("outputs/15_arithmetic_coding.png",dpi=140); plt.close(fig)
