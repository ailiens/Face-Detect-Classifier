from torchvision import transforms
from PIL import Image
from custom_dataset import CustomDataset  # your_file_name은 CustomDataset 클래스가 정의된 파일명
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# img_dir와 json_dir에는 실제 이미지와 JSON 파일이 저장된 디렉토리를 지정해야 합니다.
img_dir = 'D:/face_image_data/train/image'
json_dir = 'D:/face_image_data/train/label'

dataset = CustomDataset(img_dir=img_dir, json_dir=json_dir, transform=transform)

# 데이터셋의 200번째 항목을 로드하여 테스트
image, label = dataset[0]  # Python 인덱스는 0부터 시작하므로 200번째는 199번 인덱스입니다.

gender_str = "남자" if label == 0 else "여자"
print(f"Label: {label} ({gender_str})")

print(f"Image shape: {image.shape}")

# 200번째 이미지와 JSON 파일의 이름을 가져오기
img_name = dataset.file_list[0]
json_name = img_name.replace('.jpg', '.json')

print(f"Image file name: {img_name}")
print(f"JSON file name: {json_name}")

# 이미지를 출력하기 위한 추가 코드
image_to_show = transforms.ToPILImage()(image).convert("RGB")
plt.imshow(image_to_show)
plt.show()
