import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm

# ✅ 資料轉換設定
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# ✅ 載入 CIFAR-10 部分類別（3類）
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

trainset = CIFAR10(root='./data', train=True, transform=transform, download=True)
testset = CIFAR10(root='./data', train=False, transform=transform, download=True)

train_imgs, train_labels = filter_dataset(trainset)
test_imgs, test_labels = filter_dataset(testset)

# ✅ 使用 SURF 特徵（需 opencv-contrib-python）
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

def extract_surf_descriptors(imgs):
    descriptors = []
    for img in tqdm(imgs):
        kp, des = surf.detectAndCompute(img, None)
        descriptors.append(des)
    return descriptors

train_descs = extract_surf_descriptors(train_imgs)

# ✅ 建立視覺詞彙表（BoVW）
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
X_test = build_bow_histograms(extract_surf_descriptors(test_imgs), kmeans)

# ✅ 標準化特徵 + SVM 訓練
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='linear')
svm.fit(X_train_scaled, train_labels)

# ✅ 評估準確率
y_pred = svm.predict(X_test_scaled)
acc = accuracy_score(test_labels, y_pred)
print(f"SURF + BoVW + SVM accuracy: {acc * 100:.2f}%")
