import os
import cv2
import torch
import random
import matplotlib.pyplot as plt
from src.train.network import Model
from src.data_preparation.data_splitting import DataSplitter
from src.evaluate.draw_boxes import OutputPlotter
from src.constants import IMG_DIR, NUM_CLASSES, PRED_CLASSES, DETECTION_THRESHOLD


class ModelTester:
    def __init__(self, img_dir, detection_threshold, num_classes, pred_classes):
        self.plotter = OutputPlotter(pred_classes)
        self.num_classes = num_classes
        self.pred_classes = pred_classes
        self.data_splitter = DataSplitter(img_dir=img_dir)
        train_set, valid_set, test_set = self.data_splitter.split()

        self.model = Model(num_classes=num_classes, pretrained=True)

        self.detection_threshold = detection_threshold
        self.test_images = []

        for i in test_set:
            self.test_images.append(os.path.join(img_dir, i))
        self.model.eval()

    def test(self):
        print(f'Test instances: {len(self.test_images)}')

        for i in random.sample(range(len(self.test_images)), 5):
            image = cv2.imread(self.test_images[i])
            orig_image = image.copy()

            image = self.plotter.prepare_image(image)
            with torch.no_grad():
                outputs = self.model(image)

            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            image = self.plotter.draw_outputs(orig_image, outputs, self.detection_threshold)
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            print(f"Image {i+1} done...")
            print('-'*50)
        print('TEST PREDICTIONS COMPLETE')


ModelTester(IMG_DIR, DETECTION_THRESHOLD, NUM_CLASSES, PRED_CLASSES).test()
