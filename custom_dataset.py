from torch.utils.data import Dataset
from PIL import Image
import json
import os


class CustomDataset(Dataset):
    def __init__(self, img_dir, json_dir, transform=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transform = transform
        self.file_list = os.listdir(img_dir)  # 이미지 파일 목록

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        json_path = os.path.join(self.json_dir, img_name.replace('.jpg', '.json'))

        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size

        with open(json_path, 'r') as f:
            json_data = json.load(f)
            gender = json_data['gender']
            # print(json_data['annotation'][0]['box'])  # 여기를 추가해서 출력해보세요.
            # x, y, w, h = json_data['annotation'][0]['box']  # 예시
            # x, y, w, h = [float(i) for i in json_data['annotation'][0]['box']]
            x = float(json_data['annotation'][0]['box']['x'])
            y = float(json_data['annotation'][0]['box']['y'])
            w = float(json_data['annotation'][0]['box']['w'])
            h = float(json_data['annotation'][0]['box']['h'])

            # bounding box 중심 찾기
            center_x = x + w // 2
            center_y = y + h // 2

            # 새로운 bounding box 좌표 설정 (중심을 유지하고, 너비와 높이를 두 배로)
            new_w = int(w * 2)
            new_h = int(h * 2)
            new_x = int(center_x - new_w // 2)
            new_y = int(center_y - new_h // 2)

            # 이미지의 크기를 넘지 않도록 조정
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w = min(img_width - new_x, new_w)
            new_h = min(img_height - new_y, new_h)

            # 확장된 bounding box로 얼굴 영역 잘라내기
            cropped_image = image.crop((new_x, new_y, new_x + new_w, new_y + new_h))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, 0 if gender == 'male' else 1  # 0 for male and 1 for female

# Undersammpling(binary classification training)
def undersample_DataLoader(img_dir, json_dir, transform, batch_size, shuffle=True):
    from sklearn.utils import resample
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    # 데이터프레임 생성 (파일명과 레이블)
    file_list = os.listdir(img_dir)
    labels = []
    for img_name in file_list:
        json_path = os.path.join(json_dir, img_name.replace('.jpg', '.json'))
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            gender = json_data['gender']
            label = 0 if gender == 'male' else 1
            labels.append(label)

    data = pd.DataFrame({'Image': file_list, 'Label': labels})

    # 언더샘플링
    male_samples = data[data['Label'] == 0]
    female_samples = data[data['Label'] == 1]

    if len(male_samples) >= len(female_samples):
        undersampled_male = resample(male_samples, replace=False, n_samples=len(female_samples), random_state=42)
        undersampled_data = pd.concat([undersampled_male, female_samples])
    else:
        undersampled_female = resample(female_samples, replace=False, n_samples=len(male_samples), random_state=42)
        undersampled_data = pd.concat([male_samples, undersampled_female])

    dataset = CustomDataset(img_dir = img_dir, json_dir = json_dir, transform = transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

## 7진 분류로 변경
class AgeCustomDataset(Dataset):
    def __init__(self, img_dir, json_dir, transform=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transform = transform
        self.file_list = os.listdir(img_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        json_path = os.path.join(self.json_dir, img_name.replace('.jpg', '.json'))

        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size

        with open(json_path, 'r') as f:
            json_data = json.load(f)
            past_age = json_data['age_past']

            # Convert the original age to new category
            if 0 <= past_age <= 9:
                past_age = 0
            elif 10 <= past_age <= 19:
                past_age = 1
            elif 20 <= past_age <= 29:
                past_age = 2
            elif 30 <= past_age <= 39:
                past_age = 3
            elif 40 <= past_age <= 49:
                past_age = 4
            elif 50 <= past_age <= 59:
                past_age = 5
            else:
                past_age = 6

            x = float(json_data['annotation'][0]['box']['x'])
            y = float(json_data['annotation'][0]['box']['y'])
            w = float(json_data['annotation'][0]['box']['w'])
            h = float(json_data['annotation'][0]['box']['h'])

            # Finding the center of bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Setting new bounding box coordinates (keeping the center, width and height multiplied by 2)
            new_w = int(w * 2)
            new_h = int(h * 2)
            new_x = int(center_x - new_w // 2)
            new_y = int(center_y - new_h // 2)

            # Making sure the new bounding box doesn't exceed the image dimensions
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w = min(img_width - new_x, new_w)
            new_h = min(img_height - new_y, new_h)

            # Crop the image using the expanded bounding box
            cropped_image = image.crop((new_x, new_y, new_x + new_w, new_y + new_h))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, past_age
    
#re_image dataset 사용시
# torch offical 기준으로 재작성중인 age_model_train 에 사용하는 용도
# JSON data에서 [age_past] 값을 사용하려고 작성함
class FaceAgeDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform

        self.image_files = os.listdir(image_dir)
        self.json_files = os.listdir(json_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        json_name = os.path.join(self.json_dir, self.json_files[idx])

        image = Image.open(img_name).convert('RGB')
        with open(json_name, 'r') as f:
            label_json = json.load(f)

        label = label_json['age_past']

        # Convert the original age to new category
        if 0 <= label <= 9:
            label = 0
        elif 10 <= label <= 19:
            label = 1
        elif 20 <= label <= 29:
            label = 2
        elif 30 <= label <= 39:
            label = 3
        elif 40 <= label <= 49:
            label = 4
        elif 50 <= label <= 59:
            label = 5
        else:
            label = 6

        if self.transform:
            image = self.transform(image)

        return image, label
    

### 83진 분류 폐기 ㅋㅋ
# class AgeCustomDataset(Dataset):
#     def __init__(self, img_dir, json_dir, transform=None):
#         self.img_dir = img_dir
#         self.json_dir = json_dir
#         self.transform = transform
#         self.file_list = os.listdir(img_dir)
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         img_name = self.file_list[idx]
#         img_path = os.path.join(self.img_dir, img_name)
#         json_path = os.path.join(self.json_dir, img_name.replace('.jpg', '.json'))
#
#         image = Image.open(img_path).convert('RGB')
#         img_width, img_height = image.size
#
#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
#             past_age = json_data['age_past']
#
#             x = float(json_data['annotation'][0]['box']['x'])
#             y = float(json_data['annotation'][0]['box']['y'])
#             w = float(json_data['annotation'][0]['box']['w'])
#             h = float(json_data['annotation'][0]['box']['h'])
#
#             # Finding the center of bounding box
#             center_x = x + w // 2
#             center_y = y + h // 2
#
#             # Setting new bounding box coordinates (keeping the center, width and height multiplied by 2)
#             new_w = int(w * 2)
#             new_h = int(h * 2)
#             new_x = int(center_x - new_w // 2)
#             new_y = int(center_y - new_h // 2)
#
#             # Making sure the new bounding box doesn't exceed the image dimensions
#             new_x = max(0, new_x)
#             new_y = max(0, new_y)
#             new_w = min(img_width - new_x, new_w)
#             new_h = min(img_height - new_y, new_h)
#
#             # Crop the image using the expanded bounding box
#             cropped_image = image.crop((new_x, new_y, new_x + new_w, new_y + new_h))
#
#         if self.transform:
#             cropped_image = self.transform(cropped_image)
#
#         return cropped_image, past_age

