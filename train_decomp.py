import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dect_dataset import DECT
import logging
import torch.utils.tensorboard as tb
from models.unet import UNet
from physics.tomo import Operator
from MD import decomp_mod_update, dict2namespace
import yaml


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path
    material = 'water'
    root_dir = 'data'
    save_dir = 'ddpm_AAPMCT'
    model_name = '{}_decomp'.format(material)
    log_path = os.path.join(save_dir, "logs", model_name)
    os.makedirs(log_path, exist_ok=True)
    img_save_path = os.path.join(log_path, "inter_img")
    os.makedirs(img_save_path, exist_ok=True)
    tb_path = os.path.join(save_dir, "tensorboard", model_name)
    os.makedirs(tb_path, exist_ok=True)

    # log
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    file_handler = logging.FileHandler(os.path.join(log_path, 'train_log.txt'))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format=log_format, level=logging.DEBUG, handlers=[file_handler, stream_handler])

    # CT operator
    n = 256
    num_det = int(1.5*n)
    angles = 90
    physics = Operator(img_size=n, angles=angles, num_det=num_det, I0=1e6)

    # load data
    datasets = DECT(root_dir=root_dir, material=material, mode='train')
    dl = DataLoader(datasets, batch_size=1, shuffle=True)

    # training config
    with open(os.path.join("configs", 'aapmct_256.yml'), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    resume_training = False
    n_epochs = 1000
    snapshot_freq = 50
    md_model = decomp_mod_update(config=config, physics=physics, CG=True)
    md_model.to(device)
    optimizer = torch.optim.Adam(md_model.parameters(), lr=1e-4, weight_decay=1e-8)
    criterion = torch.nn.MSELoss().to(device)
    tb_logger = tb.SummaryWriter(log_dir=tb_path)

    start_epoch, step = 0, 0
    if resume_training:
        states = torch.load(os.path.join(log_path, "ckpt.pth"))
        md_model.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        start_epoch = states[2]
        step = states[3]
    else:
        pass

    for epoch in range(start_epoch, n_epochs):
        for i, (y1, y2, tgt) in enumerate(dl):
            md_model.train()
            step += 1

            y1 = y1.to(device)
            y2 = y2.to(device)

            # tgt = physics.A(tgt)
            # tgt = rescale(tgt)
            tgt = tgt.to(device)

            out = md_model(y1, y2)
            # m = physics.A_dagger(out)

            if epoch % 5 == 0 and i == 0:
                # m = physics.A_dagger(out)
                plt.imsave(os.path.join(img_save_path, "sample_epoch_{}.png".format(epoch)),
                           out[0].detach().cpu().numpy().squeeze(), cmap='gray')

                # plt.figure(), plt.imshow(y1[0].detach().cpu().numpy().squeeze(), cmap='gray')
                # plt.figure(), plt.imshow(y2[0].detach().cpu().numpy().squeeze(), cmap='gray')
                # plt.figure(), plt.imshow(out[0].detach().cpu().numpy().squeeze(), cmap='gray')
                # # plt.figure(), plt.imshow(m[0].detach().cpu().numpy().squeeze(), cmap='gray')
                # plt.figure(), plt.imshow(tgt[0].detach().cpu().numpy().squeeze(), cmap='gray')
                # plt.show()

            loss = criterion(out, tgt)

            tb_logger.add_scalar("loss", loss, global_step=step)

            logging.info(
                f"epoch: {epoch + 1}, step: {step}, loss: {loss.item()}"
            )

            optimizer.zero_grad()

            # loss.requires_grad = True
            loss.backward()
            optimizer.step()

            if epoch % snapshot_freq == 0 or epoch == n_epochs-1:
                states = [
                    md_model.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                torch.save(
                    states,
                    os.path.join(log_path, "ckpt_{}.pth".format(epoch)),
                )
                torch.save(states, os.path.join(log_path, "ckpt.pth"))

if __name__ == "__main__":
    main()