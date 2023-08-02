import torch
import cv2
from src.train.network import Model
from src.evaluate.draw_boxes import OutputPlotter
from src.constants import NUM_CLASSES, WEIGHTS20_PATH, DEVICE, PRED_CLASSES, DETECTION_THRESHOLD


class CameraApp:
    def __init__(self, pred_classes=PRED_CLASSES, num_classes=NUM_CLASSES, detection_threshold=DETECTION_THRESHOLD,
                 device=DEVICE, weights_path=WEIGHTS20_PATH):
        self.detection_threshold = detection_threshold
        self.num_classes = num_classes
        self.pred_classes = pred_classes
        self.device = device
        self.weights_path = weights_path
        self.plotter = OutputPlotter(pred_classes=pred_classes)

        self.model = Model(self.num_classes, pretrained=True)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

    def start(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 600)
        cap.set(4, 480)

        is_run = True

        while is_run:
            ret, image = cap.read()
            orig_image = image.copy()
            image = self.plotter.prepare_image(image)
            with torch.no_grad():
                outputs = self.model(image)

            image = self.plotter.draw_outputs(orig_image, outputs, detection_threshold=self.detection_threshold)
            cv2.imshow('Webcam', image)

            if cv2.waitKey(1) == ord('q'):
                is_run = False

        cap.release()
        cv2.destroyAllWindows()
