import os
import torch.cuda
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 4
RESIZE_SIZE = 400
SAVE_MODEL_EPOCH = 2
NUM_EPOCHS = 20
PLT_STYLE = 'ggplot'
MODEL_NAME = 'fasterrcnn_model'
EPOCH_INDICATOR_STR = "\nEPOCH: {epoch} of {epochs}"
TRAIN_LOSS_INDICATOR_STR = "Epoch #{epoch} train loss: {train_loss:.3f}"
VALID_LOSS_INDICATOR_STR = "Epoch #{epoch} validation loss: {valid_loss_hist.value:.3f}"
TIME_INDICATOR_STR = "Took {min:.3f} minutes for epoch {epoch}"

DEVICE = torch.device('cuda') if torch.cuda.is_available() \
                              else torch.device('cpu')

CLASSES = ['background', 'without_mask', 'with_mask', 'mask_weared_incorrect']
NUM_CLASSES = 4

SAVE_MODEL_PATH = os.getenv('SAVE_MODEL_PATH')
IMG_DIR = os.getenv('IMG_DIR')
LABEL_DIR = os.getenv('LABEL_DIR')
WEIGHTS18_PATH = os.getenv('WEIGHTS18_PATH')
WEIGHTS20_PATH = os.getenv('WEIGHTS20_PATH')
