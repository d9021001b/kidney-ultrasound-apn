import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm

# ⚙️ 資料預處理
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 適合 SIFT 特徵提取
    transforms.ToTensor()
])

dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
testset = CIFAR10(root='./data', train=False, transform=transform, download=True)

# ✅ 只取 3 類做分類（減少訓練時間與維度）
label_map = {0: 'airplane', 1: 'automobile', 2: 'bird'}
def filter_dataset(dataset):
    images, labels = [], []
    for img, label in dataset:
        if label in label_map:
            img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            images.append(gray)
            labels.append(label)
    return images, labels

train_imgs, train_labels = filter_dataset(dataset)
test_imgs, test_labels = filter_dataset(testset)

# ✅ 提取 SIFT 特徵
sift = cv2.SIFT_create(nfeatures=100)

def extract_sift_descriptors(imgs):
    descriptors = []
    for img in tqdm(imgs):
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors.append(des)
    return descriptors

train_descs = extract_sift_descriptors(train_imgs)

# ✅ 建立 Visual Vocabulary (Bag of Words)
all_desc = np.vstack([d for d in train_descs if d is not None])
kmeans = MiniBatchKMeans(n_clusters=64, batch_size=1000).fit(all_desc)

def build_bow_histograms(descs, kmeans):
    histograms = []
    for d in descs:
        if d is None:
            histograms.append(np.zeros(kmeans.n_clusters))
        else:
            labels = kmeans.predict(d)
            hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters+1))
            histograms.append(hist)
    return np.array(histograms)

X_train = build_bow_histograms(train_descs, kmeans)
X_test = build_bow_histograms(extract_sift_descriptors(test_imgs), kmeans)

# ⚙️ 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ SVM 分類器
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, train_labels)

# ✅ 評估
y_pred = svm.predict(X_test_scaled)
acc = accuracy_score(test_labels, y_pred)
print(f"SIFT + BoVW + SVM accuracy: {acc * 100:.2f}%")
