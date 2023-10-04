import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from custom_dataset import CustomDataset  # 데이터 로더를 직접 작성해야 함
from tqdm import tqdm

# 데이터셋 및 데이터 로더 초기화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_dir = 'D:/face_image_data/train/image'
json_dir = 'D:/face_image_data/train/label'

dataset = CustomDataset(img_dir=img_dir, json_dir=json_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 검증 데이터셋 및 데이터 로더 초기화
val_img_dir = 'D:/face_image_data/valid/image'
val_json_dir = 'D:/face_image_data/valid/label'

val_dataset = CustomDataset(img_dir=val_img_dir, json_dir=val_json_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 사전 학습된 ResNet-34 모델 불러오기
model = models.resnet34(pretrained=True)
model = model.to(device)

# 마지막 레이어를 성별 분류를 위한 레이어로 교체
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 성별은 보통 2개의 클래스로 분류
model.fc = model.fc.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습과 검증
for epoch in range(10):
    # Training Loop
    model.train()  # 학습 모드
    tqdm_iterator = tqdm(train_loader, desc=f"Training Epoch [{epoch + 1}/10]")
    for i, (inputs, labels) in enumerate(tqdm_iterator):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    # Validation Loop
    model.eval()  # 평가 모드
    total = 0
    correct = 0
    with torch.no_grad():
        tqdm_iterator = tqdm(val_loader, desc=f"Validation Epoch [{epoch + 1}/10]")
        for i, (inputs, labels) in enumerate(tqdm_iterator):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            tqdm_iterator.set_postfix({"accuracy": f"{accuracy:.2f}%"})

    print(f"Validation accuracy after epoch {epoch + 1}: {accuracy:.2f}%")

# 모델 저장
torch.save(model.state_dict(), '../models/gender_classification_model.pth')