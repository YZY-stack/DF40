'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''
import os
import sys
sys.path.append('.')

import cv2
import yaml
import random
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from dataset.albu import IsotropicResize
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.sbi_api import SBI_API


class SBIDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # Init SBI
        self.sbi = SBI_API(phase=mode,image_size=config['resolution'])

        # Init data augmentation method
        self.transform = self.init_data_aug_method()

    def __getitem__(self, index):
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]
        if not os.path.exists(real_image_path):
            real_image_path = real_image_path.replace('/Youtu_Pangu_Security_Public/', '/Youtu_Pangu_Security/public/')

        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)
        

        # # Get the parsing mask
        # parsing_mask_path = real_image_path.replace('frames', 'parse_mask')
        # parsing_mask_path_wneck = real_image_path.replace('frames', 'parse_mask_wneck')
        # parsing_mask = cv2.imread(parsing_mask_path, cv2.IMREAD_GRAYSCALE)
        # parsing_mask_wneck = cv2.imread(parsing_mask_path_wneck, cv2.IMREAD_GRAYSCALE)
        # parising_mask_combine = (parsing_mask, parsing_mask_wneck)

        sri_path = real_image_path.replace('frames', 'sri_frames')
        sri_image = self.load_rgb(sri_path)
        sri_image = np.array(sri_image)


        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array
        

        # Generate the corresponding SBI sample
        fake_image, real_image, sri_image, sri_sbi_image = self.sbi(real_image, landmark, sri_image)

        if np.random.random() < 0.35:
            margin = random.uniform(5, 15)
            fake_image = create_bbox_face(fake_image, landmark, margin=margin)
            real_image = create_bbox_face(real_image, landmark, margin=margin)
            sri_image = create_bbox_face(sri_image, landmark, margin=margin)
            sri_sbi_image = create_bbox_face(sri_sbi_image, landmark, margin=margin)
            fake_image = cv2.resize(fake_image, (256, 256))
            real_image = cv2.resize(real_image, (256, 256))
            sri_image = cv2.resize(sri_image, (256, 256))
            sri_sbi_image = cv2.resize(sri_sbi_image, (256, 256))

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image))
        real_image_trans = self.normalize(self.to_tensor(real_image))
        sri_image_trans = self.normalize(self.to_tensor(sri_image))
        sri_sbi_image_trans = self.normalize(self.to_tensor(sri_sbi_image))

        return {"fake": (fake_image_trans, 1), 
                "sri": (sri_image_trans, 2),
                "sri_sbi": (sri_sbi_image_trans, 3),
                "real": (real_image_trans, real_label)}

    def __len__(self):
        return len(self.real_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels = zip(*[data["fake"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])
        sri_images, sri_labels = zip(*[data["sri"] for data in batch])
        sri_sbi_image, sri_sbi_labels = zip(*[data["sri_sbi"] for data in batch])

        # Combine the sri and fake to obtain the final fake images
        fake_images = torch.cat([torch.stack(fake_images, dim=0), torch.stack(sri_images, dim=0), torch.stack(sri_sbi_image, dim=0)], dim=0)
        fake_labels = torch.LongTensor([1] * len(fake_labels) + [2] * len(sri_labels) + [3] * len(sri_sbi_labels))

        # Stack the image, label, landmark, and mask tensors for fake and real data
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        
        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': None,
        }
        return data_dict

    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            additional_targets={'real': 'sbi'},
        )
        return trans
    

def create_bbox_face(image, landmarks, margin=0):
    # Convert landmarks to a NumPy array if not already
    landmarks = np.array(landmarks)

    # Find the minimum and maximum x and y coordinates
    min_x, min_y = np.min(landmarks, axis=0)
    max_x, max_y = np.max(landmarks, axis=0)

    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Find the maximum of the two to get the side length of the square
    max_side = max(width, height)

    # Adjust the bounding box to be a square
    min_x = min_x - ((max_side - width) / 2)
    max_x = min_x + max_side
    min_y = min_y - ((max_side - height) / 2)
    max_y = min_y + max_side

    # Add margin
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    max_x = min(image.shape[1], max_x + margin)
    max_y = min(image.shape[0], max_y + margin)

    # Convert coordinates to integers
    min_x, min_y, max_x, max_y = map(int, [min_x, min_y, max_x, max_y])

    # Crop original image within the bbox
    face = image[min_y:max_y, min_x:max_x]

    return face


if __name__ == '__main__':
    with open('/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/preprocessing/dataset_json'
    config.update(config2)
    train_set = SBIDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break