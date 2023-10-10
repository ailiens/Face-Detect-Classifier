import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from custom_dataset import CustomDataset  # 데이터 로더를 직접 작성해야 함
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# 데이터셋 및 데이터 로더 초기화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 데이터셋 및 로더 설정
test_img_dir = 'D:/Ddrive/3rd/data_ex/validation/images'
test_json_dir = 'D:/Ddrive/3rd/data_ex/validation/labels'
test_dataset = CustomDataset(img_dir=test_img_dir, json_dir=test_json_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle=False)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 사전학습모델
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 파라미터 불러오기
model.load_state_dict(torch.load('../models/1006_gender_classification_model_re.pth'))
model.eval()

total = 0
correct = 0

# test loop
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# accuracy
accuracy = 100 * correct / total
print(f'Test accuracy: {accuracy:.2f}%')
