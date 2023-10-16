import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import dlib
import cv2


# 디바이스 체크 & 할당
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 클래스 정의
classes = ('0', '1', '2', '3', '4', '5', '6')

# 데이터 전처리 정의
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 모델 불러오기
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False)
model.load_state_dict(torch.load('../models/1011_7class_resnet101.pth'))
model = model.to(device)
model.eval()

# dlib 설정
predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


# 얼굴 및 얼굴 특징점을 인식하는 함수
def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())

        # 좌표값을 약간 확장
        x1 = max(x1 - 50, 0)
        y1 = max(y1 - 50, 0)
        x2 = min(x2 + 50, img.shape[1])
        y2 = min(y2 + 50, img.shape[0])

        shape = predictor(gray, face)
        # 점으로 인해 특징점이 예측에 영향을 미칠 수도 있다는 결과를 확인
        # for i in range(0, 68):
        #     x = shape.part(i).x
        #     y = shape.part(i).y
        #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        face_img = cv2.resize(img[y1:y2, x1:x2], (224, 224))
    return img, face_img


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def predict_age(image_path):
    original_image, face_image = detect_face(image_path)
    image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    # 이미지 시각화
    imshow(val_transform(image))

    # 예측
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)

    print(f"The predicted class label is {classes[predicted]}.")

    return classes[predicted]

image_path = "./image/byun.jpg"  # 예측하고자 하는 이미지의 경로
predicted_class = predict_age(image_path)
