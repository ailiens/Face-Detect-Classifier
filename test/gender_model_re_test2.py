import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from custom_dataset import CustomDataset  # 데이터 로더를 직접 작성해야 함
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import dlib

# dlib의 shape predictor 객체 생성
predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# 얼굴 및 얼굴 특징점을 인식하는 함수
def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        shape = predictor(gray, face)
        for i in range(0, 68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        face_img = cv2.resize(img[y1:y2, x1:x2], (224, 224))
    return face_img


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
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 파라미터 불러오기
model.load_state_dict(torch.load('../models/1010_gender_classification_model_re_under.pth'))
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
