# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # 이미지 전처리
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # 단일 이미지 불러오기
# # image_path = 'D:\\data\\images\\111.jpg'  # 이미지 경로를 설정해주세요.
# image_path = 'D:\\face_image_data\\valid\\image\\0899_1985_18_00000036_F.jpg'
#
# image = Image.open(image_path).convert('RGB')
#
# # 이미지 표시
# plt.imshow(image)
# plt.axis('off')
# plt.title('Input Image')
# plt.show()
#
# # 모델 입력을 위한 전처리
# input_image = transform(image)
# input_batch = input_image.unsqueeze(0)  # 배치 차원을 추가
#
# # GPU 설정 및 모델 불러오기
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.resnet34()
# model.fc = nn.Linear(model.fc.in_features, 2)  # 성별은 보통 2개의 클래스로 분류
# model.load_state_dict(torch.load('../models/gender_classification_model.pth'))
# model = model.to(device)
# model.eval()
#
# # 예측 수행
# with torch.no_grad():
#     input_batch = input_batch.to(device)
#     output = model(input_batch)
#     _, predicted = torch.max(output, 1)
#     print(f'Predicted label: {predicted.item()}')

#### 2번째
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# 
# # OpenCV로 얼굴 인식
# def detect_face(image_path):
#     face_cascade = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         face_img = cv2.resize(img[y:y+h, x:x+w], (224, 224))
#     return face_img
# 
# # 이미지 전처리
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# 
# # 단일 이미지 불러오기
# # image_path = './image/test.jpg'
# image_path = 'D:\\face_image_data\\valid\image\\0899_1985_13_00000029_F.jpg'# 이미지 경로를 설정해주세요.
# face_image = detect_face(image_path)
# face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
# 
# # 이미지 표시
# plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('Cropped Face')
# plt.show()
# 
# # 모델 입력을 위한 전처리
# input_image = transform(face_image_pil)
# input_batch = input_image.unsqueeze(0)
# 
# # GPU 설정 및 모델 불러오기
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.resnet34()
# model.fc = nn.Linear(model.fc.in_features, 2)  # 성별은 보통 2개의 클래스로 분류
# model.load_state_dict(torch.load('../models/gender_classification_model.pth'))
# model = model.to(device)
# model.eval()
# 
# # 예측 수행
# with torch.no_grad():
#     input_batch = input_batch.to(device)
#     output = model(input_batch)
#     _, predicted = torch.max(output, 1)
#     print(f'Predicted label: {predicted.item()}')



##### 3번째 
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import dlib  # dlib 라이브러리 추가

# dlib의 shape predictor 객체 생성
predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


# OpenCV와 dlib를 사용하여 얼굴 및 얼굴 특징점 인식
def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dlib detector를 사용하여 얼굴 검출
    faces = detector(gray)
    for face in faces:
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())

        # 얼굴 특징점 찾기
        shape = predictor(gray, face)

        # 예: 눈썹, 눈, 코, 입 등의 특징점에 점 찍기
        for i in range(0, 68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # face_img = cv2.resize(img[y:y + h, x:x + w], (224, 224))
        face_img = cv2.resize(img[y1:y2, x1:x2], (224, 224))



    return face_img


# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 단일 이미지 불러오기
# image_path = './image/test.jpg'
image_path = 'D:\\face_image_data\\train\\image\\0004_1990_03_00000018_F.jpg'# 이미지 경로를 설정해주세요.
face_image = detect_face(image_path)
face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

# 이미지 표시
plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Cropped Face')
plt.show()

# 모델 입력을 위한 전처리
input_image = transform(face_image_pil)
input_batch = input_image.unsqueeze(0)

# GPU 설정 및 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet34()
model.fc = nn.Linear(model.fc.in_features, 2)  # 성별은 보통 2개의 클래스로 분류
model.load_state_dict(torch.load('../models/gender_classification_model.pth'))
model = model.to(device)
model.eval()

# 예측 수행
with torch.no_grad():
    input_batch = input_batch.to(device)
    output = model(input_batch)
    _, predicted = torch.max(output, 1)
    print(f'Predicted label: {predicted.item()}')