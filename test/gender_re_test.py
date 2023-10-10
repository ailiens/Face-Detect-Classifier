import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from custom_dataset import CustomDataset  # your_file_name은 CustomDataset 클래스가 정의된 파일명
import matplotlib.pyplot as plt
import IPython.display as display


# 이미지 전처리를 위한 transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지를 정규화
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# model.load_state_dict((torch.load('../models/1006_gender_classification_model_re.pth')))
model.load_state_dict(torch.load('../models/1006_gender_classification_model_re.pth', map_location=device))
model.eval()

# test_image_path = '../test/image/image.jpg'   ### no
# test_image_path = '../test/image/ma.jpg'    # OK
# test_image_path = '../test/image/sun.jpg'    ### no
# test_image_path = '../test/image/nana.jpg'    # OK
# test_image_path = '../test/image/nana2.jpg'   # OK
test_image_path = '../test/image/goong.jpg'    # OK
image = Image.open(test_image_path)

image = transform(image).unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)

class_labels = ['Male', 'Female']
predicted_class = class_labels[predicted.item()]
print(f'Predicted gender: {predicted_class}')

# 이미지를 화면에 표시
plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
plt.title(f'Predicted Gender: {predicted_class}')
plt.axis('off')  # 이미지 축 숨김
plt.show()

# display.display(display.Image(data=image.squeeze().permute(1, 2, 0).cpu().numpy()))
# permute(1,2,0): 차원 재배열
# PyTorch의 이미지 텐서의 차원 순서는(채널, 높이, 너비)
# 이미지를 일반적으로 표시하거나 저장할 때는 (높이, 너비, 채널) 순서가 더 흔함 (RGB)