import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.constants import FLIP_PROB, ROTATE90_PROB, MOTION_BLUR_PROB, \
                    MEDIAN_BLUR_PROB, BLUR_LIMIT, BLUR_PROB, \
                    TO_TENSOR_PROB, BBOX_PARAMS


def get_train_transform():
    return A.Compose([
        A.Flip(FLIP_PROB),
        A.RandomRotate90(ROTATE90_PROB),
        A.MotionBlur(p=MOTION_BLUR_PROB),
        A.MedianBlur(blur_limit=BLUR_LIMIT, p=MEDIAN_BLUR_PROB),
        A.Blur(blur_limit=BLUR_LIMIT, p=BLUR_PROB),
        ToTensorV2(p=TO_TENSOR_PROB),
    ], bbox_params=BBOX_PARAMS)
