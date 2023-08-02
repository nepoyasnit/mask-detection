from tqdm.auto import tqdm
from src.constants import DEVICE


def train(train_dataloader, model, optimizer, train_itr,
          train_loss_list, train_loss_hist):
    print('Training...')

    prog_bar = tqdm(train_dataloader, total=len(train_dataloader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_itr, train_loss_list, train_loss_hist
