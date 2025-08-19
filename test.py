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

    data_processor = DataProcessor(
        image_size=config['image_size'],
        normalize=config['normalize']
    )
    
    # 加载模型
    try:
        model = tf.keras.models.load_model(config['model_path'], compile=False)
        print(f"成功加载模型: {config['model_path']}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 加载测试数据
    print("加载测试数据...")
    file_paths, labels, folds = data_processor.load_dataset(
        config['data_dir'], 
        fold_structure=config['fold_structure']
    )
    
    if len(file_paths) == 0:
        print("没有找到数据文件！")
        return
    
    # 筛选测试数据
    test_indices = [i for i, fold in enumerate(folds) if fold in config['test_folds']]
    test_paths = [file_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"测试集: {len(test_paths)} 样本")
    print(f"良性样本: {test_labels.count(0)}")
    print(f"恶性样本: {test_labels.count(1)}")
    
    # 评估模型
    print("开始评估模型...")
    test_generator = data_processor.create_data_generator(
        test_paths, test_labels, 
        batch_size=config['batch_size'], 
        is_training=False
    )
    
    predictions = []
    true_labels = []
    test_steps = len(test_paths) // config['batch_size']
    
    for i in range(test_steps):
        batch_data, batch_labels = next(test_generator)
        batch_predictions = model.predict(batch_data, verbose=0)
        predictions.extend(batch_predictions.flatten())
        true_labels.extend(batch_labels)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # 计算指标
    auc = roc_auc_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions > 0.5)
    precision = precision_score(true_labels, predictions > 0.5)
    recall = recall_score(true_labels, predictions > 0.5)
    f1 = f1_score(true_labels, predictions > 0.5)
    
    print(f"测试集结果:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # 保存结果
    results = {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    results_path = os.path.join(config['save_dir'], 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"测试结果已保存到: {results_path}")

if __name__ == "__main__":
    main()
