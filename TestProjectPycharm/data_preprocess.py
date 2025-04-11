from setup import device  # Import device từ setup.py
import torch  # Thêm dòng này
import os
from torchvision import transforms
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Đường dẫn
root_train = "D:\\ThirdYearsInHell\\deeplearning\\Project\\DatasetImageFoodVietNam1111\\DatasetImageFoodVietNam1111\\Train"
root_val = "D:\\ThirdYearsInHell\\deeplearning\\Project\\DatasetImageFoodVietNam1111\\DatasetImageFoodVietNam1111\\Validate"
root_test = "D:\\ThirdYearsInHell\\deeplearning\\Project\\DatasetImageFoodVietNam1111\\DatasetImageFoodVietNam1111\\Test"

Name_food = {0: "BanhChung", 1: "BanhCuon", 2: "BanhMi", 3: "BanhXeo", 4: "BunBoHue"}

# Transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=1.7),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_training_augmentation():
    return A.Compose([
        A.Resize(224, 224, p=1),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=200, width=200, p=0.3),
        A.Affine(rotate=(-15, 15), translate_percent=(0.1, 0.1), scale=(0.8, 1.2), p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1),
            A.CLAHE(p=0.6),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=25, val_shift_limit=20, p=1),
        ], p=0.7),
        A.Sharpen(alpha=(0.25, 0.5), lightness=(0.6, 1.0), p=0.4),
    ])

def get_validation_augmentation():
    return A.Compose([A.Resize(224, 224, p=1)])

# Lấy danh sách ảnh và nhãn
def get_path_images_labels(path):
    images = []
    labels = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if os.path.isdir(label_path):
            for image in os.listdir(label_path):
                images.append(os.path.join(label_path, image))
                labels.append(label)
    return images, labels

# Dataset class
class FoodVNDS(Dataset):
    def __init__(self, image_paths, labels, transform=None, albumentations_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.albumentations_transform = albumentations_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = torch.tensor(self.labels[idx], dtype=torch.long)

            if self.albumentations_transform:
                image_np = np.array(image)
                augmented = self.albumentations_transform(image=image_np)
                image = Image.fromarray(augmented['image'])

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return torch.zeros(3, 224, 224), label

# Load dataset
def get_all_dataset():
    train_paths, train_labels = get_path_images_labels(root_train)
    val_paths, val_labels = get_path_images_labels(root_val)
    test_paths, test_labels = get_path_images_labels(root_test)

    lb = LabelEncoder()
    train_labels = lb.fit_transform(train_labels)
    val_labels = lb.transform(val_labels)
    test_labels = lb.transform(test_labels)

    print(f"Train: {len(train_paths)} ảnh, {len(set(train_labels))} nhãn")
    print(f"Validate: {len(val_paths)} ảnh, {len(set(val_labels))} nhãn")
    print(f"Test: {len(test_paths)} ảnh, {len(set(test_labels))} nhãn")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = get_all_dataset()

# Tạo DataLoader
train_dataset = FoodVNDS(train_paths, train_labels, transform=train_transform, albumentations_transform=get_training_augmentation())
val_dataset = FoodVNDS(val_paths, val_labels, transform=test_transform, albumentations_transform=get_validation_augmentation())
test_dataset = FoodVNDS(test_paths, test_labels, transform=test_transform, albumentations_transform=get_validation_augmentation())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

print("DataLoader created successfully!")