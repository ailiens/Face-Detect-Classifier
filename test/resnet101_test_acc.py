import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 디바이스 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터 전처리
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 테스트 데이터 로딩
val_dataset = ImageFolder(root='D:\\image_data\\test\\', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 모델 불러오기
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False)
model.load_state_dict(torch.load('../models/1011_7class_resnet101.pth'))
# model.load_state_dict(torch.load('../models/1012_7class_resnet101.pth'))
# model.load_state_dict(torch.load('../models/1015_7class_resnet101.pth'))

model = model.to(device)
model.eval()

# 정확도 계산
correct = 0
total = 0

with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 배치 100 당 진행상황 출력
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} batches - Accuracy so far: {(100 * correct / total):.2f}%")

print(f'Final Accuracy of the network on test images: {(100 * correct / total):.2f}%')