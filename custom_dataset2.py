# torch offical 기준으로 재작성중인 age_model_train 에 사용하는 용도
# JSON data에서 [age_past] 값을 사용하려고 작성함
from torch.utils.data import Dataset
from PIL import Image
import json
import os


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
