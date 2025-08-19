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


    def load_dataset(self, data_dir: str, fold_structure: bool = True) -> Tuple[List, List, List]:
        file_paths, labels, folds = [], [], []
        
        if fold_structure:
            for fold in range(10):
                fold_dir = os.path.join(data_dir, f'fold{fold}')
                if not os.path.exists(fold_dir):
                    continue
                    
                for label in [0, 1]:
                    label_dir = os.path.join(fold_dir, str(label))
                    if not os.path.exists(label_dir):
                        continue
                        
                    for filename in os.listdir(label_dir):
                        if filename.endswith('.npy'):
                            file_path = os.path.join(label_dir, filename)
                            file_paths.append(file_path)
                            labels.append(label)
                            folds.append(fold)
        else:
            for label in [0, 1]:
                label_dir = os.path.join(data_dir, str(label))
                if not os.path.exists(label_dir):
                    continue
                    
                for filename in os.listdir(label_dir):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(label_dir, filename)
                        file_paths.append(file_path)
                        labels.append(label)
                        folds.append(0)
        
        return file_paths, labels, folds


    def create_data_generator(self, file_paths: List[str], labels: List[int], 
                            batch_size: int = 32, is_training: bool = True):
        num_samples = len(file_paths)
        indices = list(range(num_samples))
        
        while True:
            if is_training:
                random.shuffle(indices)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_paths = [file_paths[j] for j in batch_indices]
                batch_labels = [labels[j] for j in batch_indices]
                
                axial_slices, coronal_slices, sagittal_slices = [], [], []
                
                for path in batch_paths:
                    axial, coronal, sagittal = self.process_single_volume(path, is_training)
                    axial_slices.append(axial)
                    coronal_slices.append(coronal)
                    sagittal_slices.append(sagittal)
                
                axial_slices = np.expand_dims(np.array(axial_slices), axis=-1)
                coronal_slices = np.expand_dims(np.array(coronal_slices), axis=-1)
                sagittal_slices = np.expand_dims(np.array(sagittal_slices), axis=-1)
                batch_labels = np.array(batch_labels)
                
                yield [axial_slices, coronal_slices, sagittal_slices], batch_labels

if __name__ == "__main__":
    processor = DataProcessor(image_size=64, normalize=True)
    data_dir = "../DATASET/LUNA16_patch"
    file_paths, labels, folds = processor.load_dataset(data_dir, fold_structure=True)
    
    print(f"加载了 {len(file_paths)} 个样本")
    print(f"良性样本: {labels.count(0)}")
    print(f"恶性样本: {labels.count(1)}") 






