import torch
import random
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from src.train.network import Model
from src.data_preparation.data_splitting import DataSplitter
from src.constants import NUM_CLASSES, IMG_DIR, DEVICE, CLASSES, WEIGHTS20_PATH


def test_model():
    data_splitter = DataSplitter(img_dir=IMG_DIR)
    train_set, valid_set, test_set = data_splitter.split()

    model = Model(NUM_CLASSES, pretrained=True)
    model.load_state_dict(torch.load(WEIGHTS20_PATH, map_location=DEVICE))
    model.eval()

    detection_threshold = 0.5
    test_images = []

    for i in test_set:
        test_images.append(os.path.join(IMG_DIR, i))

    print(f'Test instances: {len(test_images)}')

    for i in random.sample(range(len(test_images)), 5):
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(float)
        image = torch.tensor(image, dtype=torch.float)#.cuda()
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            labels = outputs[0]['labels'].data.numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            labels = labels[scores >= detection_threshold]

            draw_boxes = boxes.copy()

            for box, label in zip(draw_boxes, labels):
                label = CLASSES[label]
                color = ()

                if label == CLASSES[1]:
                    color = (0, 0, 255)

                elif label == CLASSES[2]:
                    color = (0, 255, 0)

                elif label == CLASSES[3]:
                    color = (255, 0, 0)

                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            color, 1)
                cv2.putText(orig_image, label,
                            (int(box[0]), int(box[1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.show()
        print(f"Image {i+1} done...")
        print('-'*50)
    print('TEST PREDICTIONS COMPLETE')
