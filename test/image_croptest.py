from custom_dataset import CustomDataset
from torchvision import transforms
from PIL import ImageDraw
import matplotlib.pyplot as plt

transform = transforms.Compose([
    # 여기에 필요한 전처리를 넣으세요.
])

img_dir = 'D:/face_image_data/train/image'
json_dir = 'D:/face_image_data/train/label'

dataset = CustomDataset(img_dir=img_dir, json_dir=json_dir, transform=transform)

# 200번째 데이터 가져오기
image, label = dataset[199]  # 인덱스는 0부터 시작하므로

# label이 0이면 남자, 1이면 여자
gender = "Male" if label == 0 else "Female"

# 이미지와 레이블 출력
# plt.imshow(transforms.ToPILImage()(image))
plt.imshow(image)
plt.title(f"Gender: {gender}")
plt.show()

