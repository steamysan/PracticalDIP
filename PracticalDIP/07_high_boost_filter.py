
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

# High-Boost Sharpening Filter (A=1.5)
A=1.5; K=np.ones((3,3),float)/9
def apply_high_boost(im):
    blur=scipy.ndimage.convolve(im,K,mode="mirror")
    mask=im-blur
    return np.clip(im+A*mask,0,1),blur,mask

# Process the supplied sample image
rgb=load_image(); hsi=rgb_to_hsi(rgb); orig=hsi[...,2]
filt,blur,mask=apply_high_boost(orig); out=hsi_to_rgb(np.stack([hsi[...,0],hsi[...,1],filt],-1))

save_panels("07_high_boost_filter",[(rgb,"Original RGB",None),(orig,"Original intensity","gray"),(blur,"Blurred intensity","gray"),(mask,"High-frequency mask","gray"),(out,"High-boost RGB",None),(filt,"High-boost intensity","gray")])
