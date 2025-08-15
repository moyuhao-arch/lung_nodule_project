import os
import numpy as np
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import cv2

class DataProcessor:
    def __init__(self, image_size: int = 64, normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        self.min_bound = -1000
        self.max_bound = 400

    def extract_three_orthogonal_slices(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth,height,width = volume.shape
        center_d,center_h,center_w =depth//2,height//2,width//2

        axial_slice = volume[center_d, :, :]
        coronal_slice = volume[:, center_h, :]
        sagittal_slice = volume[:, :, center_w]

        return axial_slice,coronal_slice,sagittal_slice

    def resize_slice(self, slice_img: np.ndarray) -> np.ndarray:
        return cv2.resize(slice_img, (self.image_size, self.image_size))
        
    def normalize_slice(self, slice_img: np.ndarray) -> np.ndarray:
        if self.normalize:
            slice_img = (slice_img - self.min_bound) / (self.max_bound - self.min_bound)
            slice_img = np.clip(slice_img, 0.0, 1.0)
        return slice_img   
      
   def process_single_volume(self, volume_path: str, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            volume = np.load(volume_path)
            axial, coronal, sagittal = self.extract_three_orthogonal_slices(volume)
            
            axial = self.resize_slice(axial)
            coronal = self.resize_slice(coronal)
            sagittal = self.resize_slice(sagittal)
            
            axial = self.normalize_slice(axial)
            coronal = self.normalize_slice(coronal)
            sagittal = self.normalize_slice(sagittal)

            return axial,coronal,sagittal

        except Exception as e:
            print(f"处理文件 {volume_path} 时出错: {e}")
            empty_slice = np.zeros((self.image_size, self.image_size))
            return empty_slice, empty_slice, empty_slice











