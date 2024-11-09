import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=128, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        # GT : Ground Truth
        self.GT_root = root[:-1] + '_GT/'

        directories = os.listdir(root)

        self.image_paths = []
        self.GT_paths = []

        for directory in directories:
            dir_path = os.path.join(root, directory)
            path_parts = dir_path.split('/')
            path_parts[2] = f'{path_parts[2]}_GT'
            new_dir_path = os.path.join(*path_parts)
            if os.path.isdir(dir_path):
                filenames = os.listdir(dir_path)
                for filename in filenames:
                    basename, ext = os.path.splitext(filename)
                    ext = os.path.splitext(filename)[-1].lower()
                    if ext in ['.jpg', '.jpeg']:
                        self.image_paths.append(os.path.join(dir_path, filename))
                        self.GT_paths.append(os.path.join(new_dir_path, f'{basename}_mask.png'))

        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        # filename = image_path.split('_')[-1][:-len(".jpg")]
        GT_path = self.GT_paths[index]

        GT_init = Image.open(GT_path)
        GT_array = np.array(GT_init)
        GT_unique = np.unique(GT_array)
        # print('GT_unique', GT_unique,GT_path)
        # visualize_gt_regions(GT_array)
        mapping = {0: 0, 44: 1, 88: 0, 100: 2, 144: 3, 200: 0, 244: 0}
        GT_mapped = np.vectorize(lambda x: mapping.get(x, 0))(GT_array)
        GT_mapped = np.array(GT_mapped, dtype=np.uint8)

        GT = Image.fromarray(GT_mapped)
        GT.path = GT_path

        image = Image.open(image_path)
        image.path = image_path

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []
        Transform_GT = []

        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.NEAREST))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
            CropRange = random.randint(250, 270)
            Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
            Transform = T.Compose(Transform)

            image = Transform(image)
            GT = Transform(GT)

            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = image.size[0] - random.randint(0, 20)
            ShiftRange_lower = image.size[1] - random.randint(0, 20)
            image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

            image = Transform(image)

            Transform = []
            Transform_GT = []

        Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256), interpolation=Image.NEAREST))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        Transform_GT.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256), interpolation=Image.NEAREST))
        GT = Transform_GT[0](GT)
        GT = torch.tensor(np.array(GT), dtype=torch.long)
        GT.path = GT_path

        image = Transform(image)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        image.path = image_path

        return image, GT, image_path, GT_path

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    transform = {
        'train': T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    shuffle = True if mode == 'train' else False
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


def visualize_gt_regions(gt_array):
    """
    Visualize different regions in GT image with their corresponding values

    Args:
        gt_array: numpy array of GT image
    """
    unique_values = np.unique(gt_array)
    n_values = len(unique_values)

    # Create a figure with subplots for each unique value
    fig, axs = plt.subplots(1, n_values, figsize=(20, 4))
    fig.suptitle('GT Regions Visualization')

    for idx, value in enumerate(unique_values):
        mask = (gt_array == value)
        blank_image = np.zeros_like(gt_array)
        blank_image[mask] = 255

        axs[idx].imshow(blank_image, cmap='gray')
        axs[idx].set_title(f'Value: {value}')
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()