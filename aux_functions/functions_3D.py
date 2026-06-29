from xml.etree import ElementTree as ET
import numpy as np
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)
from cellpose import models
#from aux_functions.functionPercNorm import functionPercNorm
from scipy.ndimage import label, generate_binary_structure, binary_closing
from skimage import measure
from skimage.morphology import disk as skdisk, binary_erosion as sk_binary_erosion
from skimage.segmentation import expand_labels

class CellProperty3D:
  def __init__(self, xCentroid, yCentroid, zCentroid, label, volume, bbox):
    self.xCentroid = xCentroid
    self.yCentroid = yCentroid
    self.zCentroid = zCentroid
    self.label = label
    self.volume = volume
    self.bbox = bbox
  
  def to_dict(self):
    return {
        'xCentroid': self.xCentroid,
        'yCentroid': self.yCentroid,
        'zCentroid': self.zCentroid,
        'label': self.label,
        'volume': self.volume,
        'bbox' : self.bbox
    }

def get_props_per_cell3D(img_segmentation):
    
    regions = measure.regionprops(img_segmentation)
    regions = sorted(regions, key=lambda x: x.label, reverse=False)
    regionprops_selected = []
    for region in regions:
        cell_property = CellProperty3D(region.centroid[1], region.centroid[0], region.centroid[2], region.label, region.area, region.bbox)
        regionprops_selected.append(cell_property)
    
    return regionprops_selected

def segment_slice_by_slice(volume, cellpose_model_path, diameter, flag_gpu, erosion=-1, flag_closing = False,\
                            flag_fill_use_or = False, flag_projection_segmentation = False,\
                            flag_filter_by_size = False, size_min = 100):
    h,w,d = np.shape(volume)
    volume_segmented = np.uint16(np.zeros_like(volume))
    model_trained = models.CellposeModel(pretrained_model=cellpose_model_path, gpu=flag_gpu)
    for z in range(0,d):
        slice_volume = volume[:,:,z]
        masks_slice, _, _ = model_trained.eval(slice_volume, channels=[0,0], diameter=diameter, normalize=True)
        if erosion > 0:
            footprint = skdisk(erosion)
            eroded = np.zeros_like(masks_slice)
            print(np.unique(masks_slice))
            for lbl in np.unique(masks_slice):
                if lbl == 0:
                    continue
                obj_mask = sk_binary_erosion(masks_slice == lbl, footprint)
                eroded[obj_mask] = lbl
            masks_slice = eroded

        volume_segmented[:,:,z] = masks_slice > 0 # binarization of the slice
    
    volume_segmented, _ = label(volume_segmented)
    
    if flag_closing:
        structure = generate_binary_structure(rank=3, connectivity=1)
        closed_array = binary_closing(volume_segmented, structure=structure, iterations=1)
        # The closing generates in scipy leaves the top and bottom as black. I could not find a parameter as "mirror" or "reflect" as in scikit-image
        closed_array[:,:,0] = volume_segmented[:,:,0]>0
        closed_array[:,:,-1] = volume_segmented[:,:,-1]>0
        volume_segmented = closed_array
        volume_segmented, _ = label(volume_segmented)

    if flag_projection_segmentation:
        # Max projection on Z -> shape (X, Y)
        volume_segmented = volume_segmented>0
        max_proj = volume_segmented.max(axis=2)
        
        # Restore to 3D by repeating along Z -> shape (X, Y, Z)
        volume_segmented = np.repeat(max_proj[:, :, np.newaxis], volume.shape[2], axis=2)
        volume_segmented, _ = label(volume_segmented)
    
    # If erosion was applied, expand again
    if erosion > 0:
        for z in range(volume_segmented.shape[2]):
            volume_segmented[:, :, z] = expand_labels(volume_segmented[:, :, z], distance=erosion)

    if flag_filter_by_size:
        list_labels = np.unique(volume_segmented)
        for label_object in list_labels:
            if label_object > 0: # 0 is background
                size_object = np.count_nonzero(volume_segmented == label_object)
                if size_object < size_min:
                    volume_segmented[volume_segmented == label_object] = 0

    n_objects = len(np.unique(volume_segmented)) - 1 #0 is not an object, is the background
    return volume_segmented, n_objects

def segment_cellpose_3d(volume, cellpose_model_path, diameter, flag_gpu):
    model_trained_nuclei = models.CellposeModel(pretrained_model=cellpose_model_path, gpu=flag_gpu)
    masks, flows, styles = model_trained_nuclei.eval(volume, channels=[0,0], diameter=diameter, do_3D = True, normalize=True)
    masks = np.uint16(masks)
    return masks
