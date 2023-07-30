import torch
import time
import matplotlib.pyplot as plt
from src.train.network import create_model
from src.train.loss_counter import LossCounter
from src.train.network_train import train
from src.train.network_validate import validate
from src.data_preparation.dataset_class import MaskDataset
from src.data_preparation.data_splitting import train_valid_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from src.data_preparation.data_transform import get_train_transform
from src.data_preparation.supportive import collate_fn
from src.constants import SAVE_MODEL_EPOCH, NUM_EPOCHS, NUM_CLASSES, DEVICE, \
    RESIZE_SIZE, CLASSES, IMG_DIR, LABEL_DIR, BATCH_SIZE, SAVE_MODEL_PATH, EPOCH_INDICATOR_STR, \
    TRAIN_LOSS_INDICATOR_STR, VALID_LOSS_INDICATOR_STR, TIME_INDICATOR_STR, PLT_STYLE


def run_training():
    plt.style.use(PLT_STYLE)

    train_set, valid_set, test_set = train_valid_test_split(img_dir=IMG_DIR)

    train_dataset = MaskDataset(train_set, RESIZE_SIZE, RESIZE_SIZE, CLASSES, IMG_DIR, LABEL_DIR)
    trans_train_dataset = MaskDataset(train_set, RESIZE_SIZE, RESIZE_SIZE, CLASSES, IMG_DIR, LABEL_DIR,
                                      get_train_transform())
    valid_dataset = MaskDataset(valid_set, RESIZE_SIZE, RESIZE_SIZE, CLASSES, IMG_DIR, LABEL_DIR)

    train_dataset = ConcatDataset([train_dataset, trans_train_dataset])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(valid_dataset)}\n')

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_loss_hist = LossCounter()
    valid_loss_hist = LossCounter()

    train_itr = 1
    valid_itr = 1

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(NUM_EPOCHS):
        print(EPOCH_INDICATOR_STR.format(epoch=epoch+1, epochs=NUM_EPOCHS))

        train_loss_hist.reset()
        valid_loss_hist.reset()

        start = time.time()
        train_loss, train_loss_hist = train(train_loader, model, optimizer, train_itr, train_loss_list, train_loss_hist)
        valid_loss, valid_loss_hist = validate(valid_loader, model, valid_itr, valid_loss_list, valid_loss_hist)

        print(TRAIN_LOSS_INDICATOR_STR.format(epoch=epoch+1, train_loss=train_loss_hist.value))
        print(VALID_LOSS_INDICATOR_STR.format(epoch=epoch+1, valid_loss=valid_loss_hist.value))
        end = time.time()
        print(TIME_INDICATOR_STR.format(epoch=epoch+1, min=((end - start) / 60)))

        if (epoch + 1) == NUM_EPOCHS:  # save model once at the end
            torch.save(model.state_dict(), SAVE_MODEL_PATH.format(epoch=epoch+1))

        elif (epoch + 1) % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
            torch.save(model.state_dict(), SAVE_MODEL_PATH.format(epoch=epoch+1))
            print('SAVING MODEL COMPLETE...\n')


run_training()
