import os
import random


class DataSplitter:
    def __init__(self, img_dir, train_split=0.15, test_split=0.15):
        self.img_dir = img_dir
        self.train_split = train_split
        self.test_split = test_split

    def split(self):
        files = os.listdir(self.img_dir)

        all_img = files

        random.shuffle(all_img)

        len_imgs = len(all_img)

        train_test_split = int((1-self.train_split)*len_imgs)

        train_val_df = all_img[:train_test_split]
        test_df = all_img[train_test_split:]

        len_df = len(train_val_df)

        train_val_split = int((1-self.test_split)*len_df)

        train_df = train_val_df[:train_val_split]
        valid_df = train_val_df[train_val_split:]

        return train_df, valid_df, test_df
