# link: https://github.com/uci-cbcl/NoduleNet
import sys
sys.path.append('./')
from pylung.annotation import *
from pylung.utils import *
from tqdm import tqdm
import sys
import nrrd
import SimpleITK as sitk
import cv2
import glob
import matplotlib.pyplot as plt

def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def get_pos(img_mask):
    index = []
    for i in range(len(img_mask)):
        if 1 in img_mask[i]:
            index.append(i)
    return index

def nodule_mask(luna_path, ctr_path):

    img, origin, spacing = load_itk_image(luna_path)
    ctr_arrs = np.load(ctr_path, allow_pickle=True)

    mask = np.zeros(img.shape)
    for ctr_arr in ctr_arrs:
        z_origin = origin[0]
        z_spacing = spacing[0]
        ctr_arr = np.array(ctr_arr)
        ctr_arr[:, 0] = np.absolute(ctr_arr[:, 0] - z_origin) / z_spacing
        ctr_arr = ctr_arr.astype(np.int32)

        for z in np.unique(ctr_arr[:, 0]):
            ctr = ctr_arr[ctr_arr[:, 0] == z][:, [2, 1]]
            ctr = np.array([ctr], dtype=np.int32)
            mask[z] = cv2.fillPoly(mask[z], ctr, color=(1,) * 1)

    index = get_pos(mask)

    return mask, index

if __name__ == "__main__":
    luna_path = "../../../luna16/LUNA16/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.mhd"
    ctr_path = "../../../luna16/annotation/mask_test/1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.npy"
    nodule_mask(luna_path, ctr_path)
