import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
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
    return img, face_img

# 이미지 전처리
transform = transforms.Compose([transforms.ToTensor()])

# 단일 이미지 불러오기
image_path = './image/jang.jpg'  # 실제 이미지 경로로 변경해주세요.
landmarked_image, face_image = detect_face(image_path)
# face_image = detect_face(image_path)
face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

######## 1011 수정
# 랜드마크가 찍힌 원본 이미지 표시
plt.figure()
plt.imshow(cv2.cvtColor(landmarked_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Landmarked Image')
plt.show()

# 잘라낸 얼굴 이미지 표시
plt.figure()
plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Cropped Face')
plt.show()

# 모델 입력을 위한 전처리
input_image = transform(face_image_pil)
input_batch = input_image.unsqueeze(0)

# GPU 설정 및 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
age_model = models.resnet50()
N = 7  # 나이 범위의 수, 실제 모델에 맞게 변경해야 함
age_model.fc = nn.Linear(age_model.fc.in_features, N)
# age_model.load_state_dict(torch.load('../models/7class_1010resnet.pth'))  # 모델 가중치 경로
age_model.load_state_dict(torch.load('../models/7class_1010resnet2.pth'))  # 모델 가중치 경로
age_model = age_model.to(device)
age_model.eval()

# 예측 수행
with torch.no_grad():
    input_batch = input_batch.to(device)
    output = age_model(input_batch)
    _, predicted = torch.max(output, 1)
    print(f'Predicted age range: {predicted.item()}')  # 예측된 나이 범위 출력