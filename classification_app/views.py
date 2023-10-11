from django.shortcuts import render, redirect
from .forms import ImageUploadForm
import cv2
import torch
from PIL import Image
import dlib
from torchvision import models, transforms
from django.http import JsonResponse
import base64

MODEL_PATH = 'C:/Users/tjoeun/Documents/TJE_3rd_proj-2/models/'
SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'

def load_model(model_path, model_type, output_classes):
    model = model_type(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, output_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

gender_model = load_model(MODEL_PATH + 'gender_classification_model.pth', models.resnet34, 2)
age_model = load_model(MODEL_PATH + '7class_1010resnet2.pth', models.resnet50, 7)

def cv2_to_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode()
    return jpg_as_text

def detect_face_and_predict(image_path, gender_model, age_model):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH + SHAPE_PREDICTOR)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img_rgb)
    all_results = []

    for d in dets:
        shape = predictor(img_rgb, d)
        x, y, w, h = (d.left(), d.top(), d.width(), d.height())

        for i in range(0, 68):
            x_lm = shape.part(i).x
            y_lm = shape.part(i).y
            cv2.circle(img, (x_lm, y_lm), 2, (0, 255, 0), -1)

        face_image = img[y:y+h, x:x+w]
        face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        transform = transforms.Compose([transforms.ToTensor(),])
        input_image = transform(face_image_pil)
        input_batch = input_image.unsqueeze(0)

        results = {}
        with torch.no_grad():
            gender_output = gender_model(input_batch)
            _, gender_predicted = torch.max(gender_output, 1)
            results['gender'] = 'Male' if gender_predicted.item() == 0 else 'Female'

            age_output = age_model(input_batch)
            _, age_predicted = torch.max(age_output, 1)
            results['age'] = age_predicted.item()

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