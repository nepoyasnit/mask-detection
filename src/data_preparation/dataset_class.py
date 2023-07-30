import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import xml.etree.ElementTree as eTree
import cv2
import os
import numpy as np
import torch


class MaskDataset(Dataset):
    def __init__(self, dataset, width, height, classes, img_dir_path, xml_dir_path, transforms=None):
        self.transforms = transforms
        self.img_dir_path = img_dir_path
        self.xml_dir_path = xml_dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.dataset = dataset

    def __getitem__(self, idx):
        image_name = self.dataset[idx]
        image_path = os.path.join(self.img_dir_path, image_name)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        label_filename = image_name[:-4] + '.xml'
        label_filepath = os.path.join(self.xml_dir_path, label_filename)

        boxes, labels = [], []

        tree = eTree.parse(label_filepath)
        root = tree.getroot()
        image_width, image_height = 0, 0

        for i in root.findall('size'):
            image_width = int(i.find('width').text)
            image_height = int(i.find('height').text)

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            x_min_temp = int(member.find('bndbox').find('xmin').text)
            x_max_temp = int(member.find('bndbox').find('xmax').text)
            y_min_temp = int(member.find('bndbox').find('ymin').text)
            y_max_temp = int(member.find('bndbox').find('ymax').text)

            x_min = (x_min_temp / image_width) * self.width
            x_max = (x_max_temp / image_width) * self.width
            y_min = (y_min_temp / image_height) * self.height
            y_max = (y_max_temp / image_height) * self.height

            if x_min > self.width:
                x_min = self.width

            if y_min > self.height:
                y_min = self.height

            if x_max > self.width:
                x_max = self.width

            if y_max > self.height:
                y_max = self.height

            boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        if boxes.shape[0] > 1:
            is_crowd = torch.ones((boxes.shape[0],), dtype=torch.int64)
        else:
            is_crowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'area': area, 'is_crowd': is_crowd}
        image_id = torch.tensor([idx])
        target['image_id'] = image_id

        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']

            target['boxes'] = torch.Tensor(sample['bboxes'])

            return image_resized, target

        else:
            image_resized = np.transpose(image_resized, (2, 0, 1))
            image_resized = torch.from_numpy(image_resized)

            return image_resized, target

    def __len__(self):
        return len(self.dataset)
