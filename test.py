import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import json

from data_processing import DataProcessor
from model import create_model

def main():
  config = {
    'data_dir':'../DATASET/LIDC_patch_2'
    'fold_structure':True,
    'test_folds':[8,9],
    'image_size': 64,
    'normalize': True,
    'model_path': './models/best_model.h5',
    'batch_size': 32,
    'save_dir': './test_results'
  }
  os.makedirs(config['save_dir'],exist_ok=True)

  data_processing
