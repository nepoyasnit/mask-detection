import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


class OutputPlotter:
    def __init__(self, pred_classes):
        self.pred_classes = pred_classes

    @staticmethod
    def get_outputs(outputs, output_key):
        return outputs[0][output_key].data.numpy()

    def draw_outputs(self, image, outputs, detection_threshold):
        if len(outputs[0]['boxes']) != 0:
            boxes = self.get_outputs(outputs, 'boxes')
            scores = self.get_outputs(outputs, 'scores')
            labels = self.get_outputs(outputs, 'labels')

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            labels = labels[scores >= detection_threshold]

            image = self.draw_boxes(boxes, labels, image)
            return image

    @staticmethod
    def prepare_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(float)
        image = torch.tensor(image, dtype=torch.float)  # .cuda()
        image = torch.unsqueeze(image, 0)
        return image

    def draw_boxes(self, boxes, labels, image):
        for box, label in zip(boxes, labels):
            label = self.pred_classes[label]
            color = ()

            if label == self.pred_classes[1]:
                color = (0, 0, 255)

            elif label == self.pred_classes[2]:
                color = (0, 255, 0)

            elif label == self.pred_classes[3]:
                color = (255, 0, 0)

            cv2.rectangle(image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 1)

            cv2.putText(image, label,
                        (int(box[0]), int(box[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return image
