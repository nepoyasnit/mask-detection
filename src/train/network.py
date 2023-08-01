import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.constants import UNKNOWN_MODEL_EXCEPTION, WEIGHTS20_PATH, DEVICE


class Model:
    def __init__(self, num_classes, model_type='FasterRCNN', pretrained=False):
        self.num_classes = num_classes
        if model_type == 'FasterRCNN':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features

            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        else:
            raise Exception(UNKNOWN_MODEL_EXCEPTION)

        if pretrained:
            self.model.load_state_dict(torch.load(WEIGHTS20_PATH, map_location=DEVICE))

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        return self.model.to(device)

    def eval(self):
        self.model.eval()

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)
