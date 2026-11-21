import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Config ---
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "A_nckuh_model.pth"
DATA_DIR = "nckuh_0_1_inner"
METADATA_PATH = "nckuh_metadata.json"

# --- Dataset ---
class FeatureExtractionDataset(Dataset):
    def __init__(self, file_label_items, transform=None):
        self.file_label_items = file_label_items
        self.transform = transform
    def __len__(self):
        return len(self.file_label_items)
    def __getitem__(self, idx):
        path, label = self.file_label_items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 返回圖像、標籤和文件名
        filename = os.path.basename(path)
        return img, label, filename

# --- Transform ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_feature_extractor(pretrained_path):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    if os.path.exists(pretrained_path):
        print(f"載入模型: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"找不到模型: {pretrained_path}")
    
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.add_module("flatten", nn.Flatten())
    
    return feature_extractor.to(DEVICE).eval()

def extract_image_features(model, data_dir):
    file_label_items = []
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        file_label_items.extend([(f, int(label)) for f in files])
        
    dataset = FeatureExtractionDataset(file_label_items, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    features_dict = {}
    labels_dict = {}
    
    with torch.no_grad():
        for imgs, labels, filenames in tqdm(loader, desc="提取影像特徵"):
            imgs = imgs.to(DEVICE)
            features = model(imgs).cpu().numpy()
            for i, filename in enumerate(filenames):
                features_dict[filename] = features[i]
                labels_dict[filename] = labels[i].item()
                
    return features_dict, labels_dict

def load_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    metadata_dict = {}
    for item in metadata_list:
        filename = item.get("filename")
        if filename:
            # 將性別轉換為數值
            if 'gender' in item:
                # 假設 'male' 為 1, 'female' 為 0
                item['gender_numeric'] = 1 if item['gender'].lower() == 'male' else 0
            metadata_dict[filename] = item
            
    return metadata_dict

def combine_features(image_features, labels_dict, metadata):
    combined_X = []
    combined_y = []
    
    sorted_filenames = sorted(image_features.keys())
    
    for filename in sorted_filenames:
        if filename in metadata:
            meta = metadata[filename]
            img_feat = image_features[filename]
            
            # 確保所有需要的 metadata 欄位都存在
            if all(k in meta for k in ['gender_numeric', 'long_axis', 'short_axis']):
                # 組合特徵：影像特徵 + 性別 + 長軸 + 短軸
                tabular_features = np.array([
                    meta['gender_numeric'],
                    meta['long_axis'],
                    meta['short_axis']
                ])
                
                combined_feature = np.concatenate([img_feat, tabular_features])
                combined_X.append(combined_feature)
                combined_y.append(labels_dict[filename])
    
    return np.array(combined_X), np.array(combined_y)

if __name__ == "__main__":
    # 1. 提取影像特徵
    feature_extractor = get_feature_extractor(MODEL_PATH)
    image_features, labels_dict = extract_image_features(feature_extractor, DATA_DIR)
    
    # 2. 載入 Metadata
    metadata = load_metadata(METADATA_PATH)
    
    # 3. 合併特徵
    X, y = combine_features(image_features, labels_dict, metadata)
    
    if len(X) == 0:
        print("錯誤：無法合併任何特徵。請檢查檔名是否匹配以及 metadata 是否完整。")
    else:
        print(f"成功合併 {len(X)} 個樣本的特徵。")
        print(f"合併後特徵維度: {X.shape[1]}")

        # 4. 訓練邏輯迴歸模型
        # 我們使用一個 pipeline 來先標準化數據，然後進行邏輯迴歸
        # LogisticRegressionCV 會自動執行交叉驗證來找到最佳的正規化參數 C
        print("\n開始訓練邏輯迴歸模型 (使用交叉驗證)...")
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=10, 
                cv=5, 
                random_state=42, 
                max_iter=1000,
                solver='liblinear' # 適合小型數據集
            )
        )
        pipeline.fit(X, y)
        
        # 5. 分析結果
        model = pipeline.named_steps['logisticregressioncv']
        
        print("\n--- 邏輯迴歸模型分析結果 ---")
        print(f"交叉驗證後的最佳正規化參數 (C): {model.C_[0]}")
        print(f"模型在交叉驗證中的平均準確率: {np.mean(model.scores_[1]):.4f}")
        
        # 獲取係數
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        
        print(f"\n模型截距 (Intercept): {intercept:.4f}")
        
        # 顯示 metadata 特徵的係數
        print("\nMetadata 特徵的迴歸係數:")
        meta_feature_names = ['gender_numeric', 'long_axis', 'short_axis']
        for i, name in enumerate(meta_feature_names):
            # 係數在最後三個位置
            coeff_index = -len(meta_feature_names) + i
            print(f"  - {name}: {coefficients[coeff_index]:.4f}")
            
        # 顯示影像特徵中影響最大的幾個
        image_coeffs = coefficients[:-len(meta_feature_names)]
        top_5_positive_indices = np.argsort(image_coeffs)[-5:]
        top_5_negative_indices = np.argsort(image_coeffs)[:5]
        
        print("\n影像特徵中影響最大的迴歸係數 (Top 5):")
        print("  - 正向影響 (增加歸類為 1 的機率):")
        for idx in reversed(top_5_positive_indices):
            print(f"    - Feature {idx}: {image_coeffs[idx]:.4f}")
        print("  - 負向影響 (降低歸類為 1 的機率):")
        for idx in top_5_negative_indices:
            print(f"    - Feature {idx}: {image_coeffs[idx]:.4f}")
