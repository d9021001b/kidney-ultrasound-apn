import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, resnet18
from torch.utils.data import DataLoader

# ✅ 設定環境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ✅ CIFAR-10 資料集準備
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100)

# ✅ 定義 Teacher 和 Student 模型
teacher = resnet18(pretrained=True)
teacher.fc = nn.Linear(512, 10)  # CIFAR-10 有 10 類
teacher.to(device)

student = mobilenet_v2(pretrained=False)
student.classifier[1] = nn.Linear(student.last_channel, 10)
student.to(device)

# ✅ 簡易蒸餾損失 (KL Divergence + CrossEntropy)
def distillation_loss(y_student, y_teacher, y_true, T=3, alpha=0.7):
    loss_CE = nn.CrossEntropyLoss()(y_student, y_true)
    loss_KL = nn.KLDivLoss(reduction='batchmean')(
        torch.log_softmax(y_student / T, dim=1),
        torch.softmax(y_teacher / T, dim=1)
    )
    return alpha * loss_KL * (T ** 2) + (1 - alpha) * loss_CE

# ✅ 簡易訓練函式
def train(student, teacher, loader, optimizer, epochs=5):
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher(imgs)

            student_outputs = student(imgs)
            loss = distillation_loss(student_outputs, teacher_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

# ✅ 訓練學生模型（使用教師輔助）
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
train(student, teacher, train_loader, optimizer)

# ✅ 測試效能
def test(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")

test(student, test_loader)
