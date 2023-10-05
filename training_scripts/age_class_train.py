from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from custom_dataset import AgeCustomDataset  # 데이터 로더를 직접 작성해야 함

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 0.001
epochs = 10
num_classes = 83

# 데이터셋 및 데이터 로더 초기화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_dir = 'D:/face_image_data/train/image'
json_dir = 'D:/face_image_data/train/label'
dataset = AgeCustomDataset(img_dir=img_dir, json_dir=json_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 검증 데이터셋 및 데이터 로더 초기화
val_img_dir = 'D:/face_image_data/valid/image'
val_json_dir = 'D:/face_image_data/valid/label'
val_dataset = AgeCustomDataset(img_dir=val_img_dir, json_dir=val_json_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 모델 생성 및 옵티마이저, 손실 함수 설정
model = models.resnet50(pretrained=True) # 101, 152
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 학습 및 검증 정확도 및 손실 저장을 위한 리스트
train_accs = []
train_losses = []
val_accs = []
val_losses = []

# 가장 좋은 모델 저장을 위한 변수
best_val_acc = 0.0

# 모델 학습 및 검증
for epoch in tqdm(range(epochs)):
    model.train()
    correct_train = 0
    total_train = 0
    loss_train = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_train += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0:  # Print every 10th batch
            tqdm.write(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_acc = 100 * correct_train / total_train
    train_accs.append(train_acc)
    train_losses.append(loss_train / len(train_loader))
    print(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {train_acc:.2f}%, Train Loss: {loss_train / len(train_loader):.4f}')


    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:  # Print every 10th batch
                tqdm.write(f"Validation Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(val_loader)}], Loss: {loss.item():.4f}")

    val_acc = 100 * correct_val / total_val
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_acc)

    print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss / len(val_loader):.4f}')


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../models/best_age_classification_model.pth')

# Plotting accuracy and loss
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()