import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from custom_dataset import AgeCustomDataset  # Assume you have this implemented
import time


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,1),
        )

        self.f1 = nn.Linear(18432,7)
        #self.dropout = nn.Dropout2d(0.2)
        self.f2 = nn.Linear(7,1)

    def forward(self, input):
        scale = self.feature(input)
        batch_size = scale.size(0)
        cat_scale = scale.view(batch_size, -1)
        # print("Shape of cat_scale:", cat_scale.shape)  # Add this line
        feat = self.f1(cat_scale)
        pred = self.f2(feat)
        if torch.isnan(scale).any():
            print('NAN in scale')
        if torch.isnan(cat_scale).any():
            print('NAN in cat_scale')
        if torch.isnan(feat).any():
            print('NAN in feat')
        if torch.isnan(pred).any():
            print('NAN in pred')
        return feat, pred


# # GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 모델, 옵티마이저, 손실 함수 초기화
model = Backbone().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 30 에폭마다 학습률에 0.1을 곱함
criterion = nn.MSELoss()

# # 하이퍼파라미터 설정
batch_size = 16
learning_rate = 0.001
epochs = 10
num_classes = 7

# # 데이터셋 및 데이터 로더 초기화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_dir = 'D:/face_image_data/train/image'
json_dir = 'D:/face_image_data/train/label'

val_img_dir = 'D:/face_image_data/valid/image'
val_json_dir = 'D:/face_image_data/valid/label'

# For saving the best model
best_val_loss = float('inf')
model_save_path = '../models/kp_best_model.pth'

# 데이터 로더 초기화
train_dataset = AgeCustomDataset(img_dir=img_dir, json_dir=json_dir, transform=transform)  # Fill in details
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = AgeCustomDataset(img_dir=val_img_dir, json_dir=val_json_dir, transform=transform)  # Fill in details
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# For storing losses and plotting
train_losses = []
val_losses = []

#####
import logging

logging.basicConfig(level=logging.INFO)

# Training loop
for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    batch_count = 0

    start_time_epoch = time.time()

    for i, (inputs, labels) in enumerate(train_loader):
        start_time_batch = time.time()

        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        _, outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        batch_count += 1

        if (i + 1) % 100 == 0:
            end_time_batch = time.time()
            batch_time = end_time_batch - start_time_batch
            logging.info(
                f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Train Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")

    scheduler.step()

    end_time_epoch = time.time()
    epoch_time = end_time_epoch - start_time_epoch
    avg_train_loss = running_train_loss / batch_count
    train_losses.append(avg_train_loss)
    logging.info(
        f"Epoch [{epoch + 1}/{epochs}], Average Train Loss: {avg_train_loss}, Time per epoch: {epoch_time:.2f}s")

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            start_time_batch = time.time()  # Start time for batch

            inputs, labels = inputs.to(device), labels.to(device).float()
            _, outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_val_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1

            if (i + 1) % 100 == 0:  # Log every 100 batches
                end_time_batch = time.time()
                batch_time = end_time_batch - start_time_batch
                logging.info(
                    f"Validation - Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(val_loader)}], Val Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")

    avg_val_loss = running_val_loss / batch_count
    val_losses.append(avg_val_loss)

    val_acc = 100 * correct / total
    logging.info(f'Epoch [{epoch + 1}/10], Average Val Loss: {avg_val_loss}, Val Acc: {val_acc}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)

# Plotting loss curves
plt.figure()
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../test/plot_image/loss_curves.png')

#####

# # 하이퍼파라미터 설정
# batch_size = 8
# learning_rate = 0.001
# epochs = 10
# num_classes = 7
#
# # 데이터셋 및 데이터 로더 초기화
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
#
# img_dir = 'D:/face_image_data/train/image'
# json_dir = 'D:/face_image_data/train/label'
# train_dataset = AgeCustomDataset(img_dir=img_dir, json_dir=json_dir, transform=transform)  # Fill in details
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# val_img_dir = 'D:/face_image_data/valid/image'
# val_json_dir = 'D:/face_image_data/valid/label'
# val_dataset = AgeCustomDataset(img_dir=val_img_dir, json_dir=val_json_dir, transform=transform)  # Fill in details
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# # Progress bars
# train_pbar = tqdm(total=len(train_loader), desc="Training  ", position=0, leave=True)
# val_pbar = tqdm(total=len(val_loader) // 100, desc="Validation", position=1, leave=True)
#
# # GPU 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
#
# # 모델 생성 및 옵티마이저, 손실 함수 설정
# model = models.resnet50(pretrained=False)
# model = models.resnet50(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_classes)
# model = model.to(device)
# print(model.cuda())
#
# # Optimizer and Loss
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.7)  # Learning Rate Scheduler
# criterion = nn.CrossEntropyLoss()
#
# # 학습 및 검증 정확도 및 손실 저장을 위한 리스트
# train_accs = []
# train_losses = []
# val_accs = []
# val_losses = []
#
# # 가장 좋은 모델 저장을 위한 변수
# best_val_acc = 0.0
#
# # 모델 학습 및 검증
# for epoch in range(epochs):
#     model.train()
#     correct_train = 0
#     total_train = 0
#     loss_train = 0
#     train_pbar.reset()
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss_train += loss.item()
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels).sum().item()
#
#         train_pbar.set_postfix({'Training Loss': loss.item()})
#         train_pbar.update(1)
#
#     scheduler.step()
#
#     train_acc = 100 * correct_train / total_train
#     train_losses.append(loss_train / len(train_loader))
#     train_accs.append(train_acc)
#     train_pbar.write(
#         f'Epoch {epoch + 1}/{epochs}, Train Accuracy: {train_acc:.2f}%, Train Loss: {loss_train / len(train_loader):.4f}')
#
#     model.eval()
#     val_loss = 0.0
#     correct_val = 0
#     total_val = 0
#     val_pbar.reset()
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(val_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#
#             _, predicted = torch.max(outputs.data, 1)
#             total_val += labels.size(0)
#             correct_val += (predicted == labels).sum().item()
#
#             if (i + 1) % 100 == 0:  # 100의 배수일 때마다 업데이트
#                 val_pbar.set_postfix({'Validation Loss': loss.item()})
#                 val_pbar.update(1)
#
#     val_acc = 100 * correct_val / total_val
#     val_losses.append(val_loss / len(val_loader))
#     val_accs.append(val_acc)
#     val_pbar.write(
#         f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss / len(val_loader):.4f}')
#
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), '../models/kp_50_best_age_classification_model.pth')
#
# # Plotting results
# plt.figure()
# plt.plot(train_accs, label='Train Accuracy')
# plt.plot(val_accs, label='Validation Accuracy')
# plt.legend()
# plt.savefig('../test/plot_image/accuracy_plot.png')
# plt.show()
#
# plt.figure()
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.legend()
# plt.savefig('../test/plot_image/loss_plot.png')
# plt.show()