from tqdm.auto import tqdm
from src.constants import DEVICE
import torch


def validate(valid_dataloader, model, val_itr,
             val_loss_list, val_loss_hist):
    print('Validating...')
    prog_bar = tqdm(valid_dataloader, total=len(valid_dataloader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_itr, val_loss_list, val_loss_hist
