import numpy as np
import scipy.stats

from aux_functions.functionReadTIFFMultipage import read_multipage_tiff
import cv2
import os

def get_one_channel(img_ndarray):
    if len(img_ndarray.shape)==3: #RGB
        return img_ndarray[:,:,0]
    if len(img_ndarray.shape)==2: #one channel
        return img_ndarray
    if len(img_ndarray.shape)==4: #RGBA
        return img_ndarray[:,:,0,0]
    

def functionPercNorm(slice_original_quantile, prob=[0.001, 0.999]): # Wigert et al 2018
    vQuantiles = scipy.stats.mstats.mquantiles(slice_original_quantile, prob=prob, alphap=0.5, betap=0.5)

    quantile999low = vQuantiles[0]
    #print(vQuantiles)
    quantile999max = vQuantiles[1]

    slice_original_quantile = np.where(slice_original_quantile<quantile999low, quantile999low, slice_original_quantile)
    slice_original_quantile = np.where(slice_original_quantile>quantile999max, quantile999max, slice_original_quantile)

    slice_original_quantile = np.double(slice_original_quantile - quantile999low)/np.double(quantile999max - quantile999low + 0.0000001)

    return slice_original_quantile

def main():
    #Read one image, normalize and save
    #------------------PARAMETERS-----------------
    str_channel = ''
    folder_input = ''
    file_image = ''
    fullpath_tiff = os.path.join(folder_input, file_image)
    nbits = 16
    
    #--------------------------------------------------------------------------
    volume = functionReadTIFFMultipage(fullpath_tiff, bitdepth = nbits)
    n_slices = 1
    
    for i in range(n_slices):
        any_slice = volume[:,:,i]
        
        # 
        ending = '_HistNorm.png'
        any_slice_norm = functionPercNorm( np.single(any_slice))
        any_slice_norm = np.uint8(any_slice_norm * 255.)
        
        ending_crop = '_HistNorm.png'
        cv2.imwrite(os.path.join(folder_input, file_image + ending_crop), any_slice_norm)
        
    
if __name__ == "__main__":
    main()
