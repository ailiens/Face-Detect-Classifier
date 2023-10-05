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


