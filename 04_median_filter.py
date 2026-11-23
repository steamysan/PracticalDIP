
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.ndimage
from pathlib import Path

IMAGE_FILE = "images/sample_cat.png"

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

# 3x3 Median Filter on HSI Intensity
def apply_filter(im):
    return np.median(sliding_window_view(np.pad(im,1,mode="reflect"),(3,3)),axis=(-2,-1))
test=np.array([[10,20,10],[30,200,20],[10,30,20]],float)
print("Median-filter center:",apply_filter(test)[1,1],"(expected 20)")

# Process the supplied sample image
rgb=load_image(); hsi=rgb_to_hsi(rgb); orig=hsi[...,2]
np.random.seed(42); noisy=orig.copy(); rnd=np.random.rand(*orig.shape); noisy[rnd<.025]=0; noisy[rnd>.975]=1
filt=apply_filter(noisy); out=hsi_to_rgb(np.stack([hsi[...,0],hsi[...,1],filt],-1))
print("PSNR noisy / filtered:",*[round(10*np.log10(1/np.mean((orig-x)**2)),2) for x in (noisy,filt)])

save_panels("04_median_filter",[(rgb,"Original RGB",None),(orig,"Original intensity","gray"),(noisy,"Salt & pepper noise","gray"),(filt,"Median filtered intensity","gray"),(out,"Filtered RGB",None),(np.abs(orig-noisy),"Noise map","hot")])