from django.shortcuts import render, redirect
from .forms import ImageUploadForm
import cv2
import torch
from PIL import Image
import dlib
from torchvision import transforms
import base64
from torchvision import models

## 전에 사용하던거
# def load_model(model_path, model_type, output_classes):
#     model = model_type(pretrained=False)
#     num_ftrs = model.fc.in_features
#     model.fc = torch.nn.Linear(num_ftrs, output_classes)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# 여기부터
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = 'C:/Users/tjoeun/Documents/TJE_3rd_proj-2/models/'
SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'


def load_gender_model(model_path, model_type, output_classes):
    model = model_type(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, output_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def load_age_model(model_path, model_type):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_type, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    print(f"Loaded age model: {model}")  # Debugging
    return model


# Load models
# gender_model = load_gender_model(MODEL_PATH + 'Final_1010_gender_classification_model_re_under.pth', models.resnet50, 2)
gender_model = load_gender_model(MODEL_PATH + 'gender_classification_model.pth', models.resnet34, 2)

age_model = load_age_model(MODEL_PATH + '1012_7class_resnet101.pth', 'resnet101')


def cv2_to_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode()
    return jpg_as_text


def detect_face_and_predict(image_path, gender_model, age_model):
    print("Detecting faces and predicting attributes...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH + SHAPE_PREDICTOR)
    print(f"Loaded predictor: {predictor}")  # Debugging
    print("Image path: ", image_path)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img_rgb)
    print(f"Detected faces: {dets}")  # Debugging

    all_results = []
    age_range = {0: "10세 미만", 1: "10대", 2: "20대", 3: "30대", 4: "40대", 5: "50대", 6: "60대 이상"}

    for d in dets:
        shape = predictor(img_rgb, d)
        x, y, w, h = (d.left(), d.top(), d.width(), d.height())

        # 좌표값을 약간 확장
        x = max(x - 40, 0)
        y = max(y - 40, 0)
        x2 = min(x + w + 40, img.shape[1])
        y2 = min(y + h + 40, img.shape[0])

        # 얼굴 부분만 크롭
        face_image = img[y:y2, x:x2]
        face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        transform = transforms.Compose([transforms.ToTensor(), ])
        input_image = val_transform(face_image_pil)
        input_batch = input_image.unsqueeze(0).to(device)  # Modified

        results = {}
        with torch.no_grad():
            gender_output = gender_model(input_batch)
            _, gender_predicted = torch.max(gender_output, 1)
            print(f"Gender output shape: {gender_output.shape}")  # Debugging

            age_output = age_model(input_batch)
            _, age_predicted = torch.max(age_output, 1)
            print(f"Age output shape: {age_output.shape}")  # Debugging

            results['gender'] = 'Male' if gender_predicted.item() == 0 else 'Female'
            results['age'] = age_range[age_predicted.item()]

        all_results.append(results)


    landmarked_image_base64 = cv2_to_base64(img)
    cropped_face_image_base64 = cv2_to_base64(face_image)

    return all_results, landmarked_image_base64, cropped_face_image_base64

def index(request):
    context = {}
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            image_path = form.instance.image.path
            context['uploaded_image'] = form.instance.image

            try:
                prediction_results, landmarked_image_base64, cropped_face_image_base64 = detect_face_and_predict(image_path, gender_model, age_model)
                if len(prediction_results) == 0:
                    raise Exception("No faces detected.")

                context['results'] = prediction_results
                context['landmarked_image_base64'] = landmarked_image_base64
                context['cropped_face_image_base64'] = cropped_face_image_base64

                return render(request, 'result.html', context)
            except Exception as e:
                context['error'] = str(e)
                return render(request, 'index.html', context)
    else:
        form = ImageUploadForm()
        context['form'] = form

    return render(request, 'index.html', context)

def result(request):
    return render(request, 'result.html')