import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from custom_dataset import CustomDataset  # your_file_name은 CustomDataset 클래스가 정의된 파일명
import matplotlib.pyplot as plt
import IPython.display as display
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
    return face_img

#### 이미지 불러오기
# test_image_path = '../test/image/image.jpg'   ### no  >> OK
# test_image_path = '../test/image/ma.jpg'    # OK
# test_image_path = '../test/image/sun.jpg'    ### no
# test_image_path = '../test/image/nana.jpg'    # OK
# test_image_path = '../test/image/nana2.jpg'   # OK
# test_image_path = '../test/image/goong.jpg'    # OK
# test_image_path = '../test/image/jungi.jpg'   ### no
# test_image_path = '../test/image/jo.jpg'   # no 조혜련 태보
# test_image_path = '../test/image/jo2.jpg'   # 조혜련 OK
# test_image_path = '../test/image/yeop.jpg'   # 여장 신동엽 OK
test_image_path = '../test/image/cheol.jpg'   # 김희철 OK

face_image = detect_face(test_image_path)
face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

# 이미지 표시
plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Cropped Face')
plt.show()

# 이미지 전처리를 위한 transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 모델 입력을 위한 전처리
input_image = transform(face_image_pil)
input_batch = input_image.unsqueeze(0)

# GPU설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 모델 불러오기
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

model.load_state_dict(torch.load('../models/1010_gender_classification_model_re_under.pth', map_location=device))
model.eval()

with torch.no_grad():
    input_batch = input_batch.to(device)
    output = model(input_batch)
    _, predicted = torch.max(output, 1)
    print(f'Predicted age range: {predicted.item()}')  # 예측된 성별 출력
    class_labels = ['Male', 'Female']
    predicted_class = class_labels[predicted.item()]
    print(f'Predicted gender: {predicted_class}')




#
# # 이미지를 화면에 표시
# plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
# plt.title(f'Predicted Gender: {predicted_class}')
# plt.axis('off')  # 이미지 축 숨김
# plt.show()
#
# # display.display(display.Image(data=image.squeeze().permute(1, 2, 0).cpu().numpy()))
# # permute(1,2,0): 차원 재배열
# # PyTorch의 이미지 텐서의 차원 순서는(채널, 높이, 너비)
# # 이미지를 일반적으로 표시하거나 저장할 때는 (높이, 너비, 채널) 순서가 더 흔함 (RGB)