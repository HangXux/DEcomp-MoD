import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import yaml
import torchvision.transforms as transforms


class AAPMCT(Dataset):
    """CT dataset."""
    def __init__(self, root_dir, transform, downsample=False):
        self.path = sorted(glob.glob(os.path.join(root_dir, '*.npy')))
        self.transform = transform
        self.downsample = downsample

    def __getitem__(self, index):
        img = np.load(self.path[index])

        if self.downsample is not False:
            endx = img.shape[0]
            endy = img.shape[1]
            img = img[0:endx-1:2, 0:endy-1:2]

        # img = (img - img.min()) / (img.max() - img.min())  # [0, 1]

        if img.ndim == 2:
            img = img[None, ...]
        img = torch.from_numpy(img).float()

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.path)



if __name__ == '__main__':
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    with open(os.path.join("../configs", 'aapmct_256.yml'), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)


    def parse_args():
        parser = argparse.ArgumentParser(description=globals()["__doc__"])
        parser.add_argument(
            "--exp", type=str, default="D:/DEcomp-MoD/ddpm_AAPMCT")
        args = parser.parse_args()
        return args
    
    def get_dataset(args, config):
        if config.data.random_flip is False:
            tran_transform = None
            test_transform = None
        else:
            tran_transform = transforms.RandomHorizontalFlip(p=0.5)
            test_transform = None

        if config.data.dataset == "AAPMCT":
            train_folder = "{}/train".format(config.data.category)
            test_folder = "{}/test".format(config.data.category)
            dataset = AAPMCT(os.path.join(args.exp, "datasets", train_folder),
                             transform=tran_transform, downsample=config.data.downsample)
            test_dataset = AAPMCT(os.path.join(args.exp, "datasets", test_folder),
                                  transform=test_transform, downsample=config.data.downsample)
        return dataset, test_dataset

    dataset, test_dataset = get_dataset(parse_args(), config)
    print(len(dataset))
    dl = DataLoader(dataset, batch_size=1, shuffle=True)
    for x in dl:
        print(x.shape)
        plt.figure()
        x = x.detach().cpu().numpy().squeeze()
        print(x.shape)
        plt.imshow(x, cmap='gray')
        plt.show()
