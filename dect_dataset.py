import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from physics.ct import CT


class DECT(Dataset):
    """DECT dataset."""

    def __init__(self, root_dir, material='water', thick='1mm', mode='train'):

        self.low = sorted(glob.glob(os.path.join(root_dir, 'CT_energysino_{}_tomo'.format(thick), mode, 'low', '*.npy')))
        self.high = sorted(glob.glob(os.path.join(root_dir, 'CT_energysino_{}_tomo'.format(thick), mode, 'high', '*.npy')))
        self.label = sorted(glob.glob(os.path.join('ddpm_AAPMCT/datasets', material+'_'+thick, mode, '*.npy')))
        self.mode = mode

    def __len__(self):

        return len(self.low)

    def __getitem__(self, index):

        # get low- and high- energy sinogram
        y0 = np.load(self.low[index])
        # y0 = np.transpose(y0)
        y1 = np.load(self.high[index])
        # y1 = np.transpose(y1)

        # get target image
        tgt = np.load(self.label[index])
        # w, h = tgt.shape[0], tgt.shape[1]
        # tgt = tgt[0:w - 1:2, 0:h - 1:2]
        # if self.mode == 'test':
        #     tgt = (tgt - tgt.min()) / (tgt.max() - tgt.min())   # [0, 1]
        # min, max = -1024, 3072
        # tgt = (tgt - min) / (max - min)

        if y0.ndim == 2:
            y0 = y0[None, ...]
        y0 = torch.from_numpy(y0).to(torch.float32)

        if y1.ndim == 2:
            y1 = y1[None, ...]
        y1 = torch.from_numpy(y1).to(torch.float32)

        if tgt.ndim == 2:
            tgt = tgt[None, ...]
        tgt = torch.from_numpy(tgt).to(torch.float32)

        return y0, y1, tgt

if __name__ == '__main__':
    datasets = DECT(root_dir='data', material='water', thick='1mm', mode='train')
    print(len(datasets))
    dl = DataLoader(datasets, batch_size=1, shuffle=True)
    physics = CT(img_width=256, radon_view=90)
    for i, (y0, y1, tgt) in enumerate(dl):
        plt.figure(1), plt.imshow(y0.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.figure(2), plt.imshow(y1.detach().cpu().numpy().squeeze(), cmap='gray')

        # y0 = torch.transpose(y0, 2, 3)
        # y1 = torch.transpose(y1, 2, 3)
        x0 = physics.A_dagger(y0)
        x1 = physics.A_dagger(y1)
        plt.figure(3), plt.imshow(x0.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.figure(4), plt.imshow(x1.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.figure(5), plt.imshow(tgt.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.show()
        print(i)
