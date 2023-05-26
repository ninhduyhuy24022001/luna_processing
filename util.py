import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.patches import Rectangle
from PIL import Image
import glob

config_paths = {
    "luna16":"../../luna16/LUNA16/",
    "segment":"../../luna16/seg-lungs-LUNA16/",
    "annotations":"../../luna16/annotations.csv",
    "candidates":"../../luna16/candidates.csv",
    "slide_v1":"../center-coord-chunk",
    "slide_v2":"../center-coord-chunk-v2",
}

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))



def plot_ct(image, index):
    """
    image: np.array of iamge
    index: list index have mask value
    """
    num_fig = int(np.ceil(np.sqrt(len(index))))

   
    fig, axs = plt.subplots(num_fig, num_fig, figsize=(20,14))
    slide = 0
    for x in range(num_fig):
        for y in range(num_fig):
            try:
                axs[x, y].imshow(image[index[slide]])
                axs[x, y].set_title(index[slide])
                slide += 1
            except:
                pass
            
    plt.show()
    

def plot_ct_scan_bbox(image, index, center_irc, diamiter):
    """
    image: np.array of iamge
    index: list index have mask value
    """
    num_fig = int(np.ceil(np.sqrt(len(index))))

    fig, axs = plt.subplots(num_fig, num_fig, figsize=(20,14))

    slide = 0
    for x in range(num_fig):
        for y in range(num_fig):
            try:
                axs[x, y].imshow(image[index[slide]])
                axs[x, y].set_title(index[slide])
                axs[x, y].gca().add_patch(Rectangle((center_irc[2], center_irc[1]), diamiter, diamiter,
                    edgecolor='red',
                    facecolor='none',
                    lw=4))

                slide += 1
            except:
                pass
            
    plt.show()

def get_uids(subsets = list(range(10))):
    result = []

    for subset in subsets:
        files = glob.glob(config_paths["luna16"] + f"subset{subset}/*.mhd")

        for f in files:
            result.append(f.split("\\")[-1][:-4])

    return result