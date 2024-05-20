import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from physics.ct import CT
from physics.tomo import Operator
import torch
import argparse


# def norm(input):
#     newimg = (input - input.min()) / (input.max() - input.min())            # [0, 1]
#     return newimg

def transform_ctdata(self, windowWidth, windowCenter, normal=True):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (self - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg

def bone_weighting(input, Ts, Tb):  # 1200, 1600
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] > Tb:
                input[i, j] = 1 * input[i, j]
            elif input[i, j] < Ts:
                input[i, j] = 0 * input[i, j]
            else:
                input[i, j] = (input[i, j] - Ts) / (Tb - Ts) * input[i, j]
    return input

def water_weighting(input, Ts, Tb):
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] > Tb:
                input[i, j] = 0 * input[i, j]
            elif input[i, j] < Ts:
                input[i, j] = 1 * input[i, j]
            else:
                input[i, j] = (Tb - input[i, j]) / (Tb - Ts) * input[i, j]
    return input

def main(args):
    mode = args.mode
    # data_path = 'D:/LIDC/LIDC_img/{}'.format(mode)
    data_path = os.path.join(args.data_path, mode)
    img_path = glob.glob(os.path.join(data_path, '*.npy'))
    bone_save_path = os.path.join(args.bone_save_path, mode)
    os.makedirs(os.path.join(bone_save_path), exist_ok=True)
    water_save_path = os.path.join(args.water_save_path, mode)
    os.makedirs(os.path.join(water_save_path), exist_ok=True)

    # continue_seg = sorted(glob.glob(os.path.join(water_save_path, '*.npy')))

    kvp_list = ['90', '150']   # options are "70" (low) and "120" (high)

    # save path
    energy_sino_save_path = os.path.join(args.energy_sino_save_path)
    low_sino_save_path = os.path.join(energy_sino_save_path, mode, 'low')
    high_sino_save_path = os.path.join(energy_sino_save_path, mode, 'high')
    # low_transmission_path = os.path.join('data/CT_transmission/{}/low'.format(mode))
    # high_transmission_path = os.path.join('data/CT_transmission/{}/high'.format(mode))
    os.makedirs(os.path.join(low_sino_save_path), exist_ok=True)
    os.makedirs(os.path.join(high_sino_save_path), exist_ok=True)
    # os.makedirs(os.path.join(low_transmission_path), exist_ok=True)
    # os.makedirs(os.path.join(high_transmission_path), exist_ok=True)

    pixel_max = 3072
    pixel_min = -1024
    pixel_range = pixel_max - pixel_min
    # scale_factor = 1 / pixel_range

    for i in tqdm(range(len(img_path))):
        img = np.load(img_path[i])
        w, h = img.shape[0], img.shape[1]
        img = img[0:w - 1:2, 0:h - 1:2]

        # plt.figure(), plt.imshow(img, cmap='gray')
        # img = img + (0 - img.min())  # set the min value to 0

        # rescale threshold
        Ts = 200
        Tb = 600
        # minWindow = float(windowCenter) - 0.5 * float(windowWidth)
        new_Ts = float((Ts - pixel_min) / pixel_range)
        new_Tb = float((Tb - pixel_min) / pixel_range)

        # segment
        img_bone = bone_weighting(np.array(img), Ts=new_Ts, Tb=new_Tb)
        # img_water = water_weighting(np.array(img), Ts=new_Ts, Tb=new_Tb)
        img_water = img - img_bone

        # normalize to [0, 1]
        # img_bone = norm(img_bone)
        # img_water = norm(img_water)

        # plt.figure(), plt.imshow(img_bone, cmap='gray')
        # plt.figure(), plt.imshow(img_water, cmap='gray')
        # plt.show()

        # # save material images
        # np.save(os.path.join(water_save_path, 'water_' + str(i)), img_water)
        # np.save(os.path.join(bone_save_path, 'bone_' + str(i)), img_bone)


        # CT geometry
        img_width = 256
        radon_view = 180
        num_det = int(img_width*1.5)
        device = "cuda:0"
        # physics = CT(img_width=img_width, radon_view=radon_view, device=device)

        physics = Operator(img_size=img_width, angles=radon_view, num_det=num_det, I0=1e6)

        # material sinograms
        img_bone = img_bone[None, ...]
        img_water = img_water[None, ...]
        img_bone = torch.from_numpy(img_bone).float().to(device)
        img_water = torch.from_numpy(img_water).float().to(device)
        s_bone = physics.radon(img_bone)
        s_water = physics.radon(img_water)
        s_bone = s_bone.detach().cpu().numpy().squeeze()
        s_water = s_water.detach().cpu().numpy().squeeze()

        # plt.figure(), plt.imshow(s_bone, cmap='gray')
        # plt.figure(), plt.imshow(s_water, cmap='gray')

        # Create energy sinograms
        for k in range(2):
            kvp = kvp_list[k]
            # knock out every other view for kVp switching. First view is 70 kVp, second is 120 kVp, ...
            if kvp == "90":
                sbone = s_bone[1::2, :]
                swater = s_water[1::2, :]

            elif kvp == "150":
                sbone = s_bone[0::2, :]
                swater = s_water[0::2, :]

            modeldata = np.load(os.path.join('../data/spectra', kvp + 'kvp_data.npy'))
            energies = modeldata[0] * 1.  # energy bins
            spectrum = modeldata[1] * 1.  # spectrum
            mu_bone = modeldata[2] * 1.  # bone mass attenuation coefficient
            mu_water = modeldata[3] * 1.  # water mass attenuation coefficient

            nviews, nbins = sbone.shape  # Grab the data dimensions from the Sinogram data

            delta_e = 0.5  # bin width

            transmission = np.zeros([nviews, nbins], "float32")
            for j in range(len(energies)):
                transmission += delta_e * spectrum[j] * np.exp(-mu_bone[j] * sbone - mu_water[j] * swater)

            # create energy sinograms
            energy_sino = -np.log(transmission)

            # energy_sino = energy_sino[None, ...]
            # energy_sino = torch.from_numpy(energy_sino).float().to(device)
            # physics_energy = Operator(img_size=img_width, angles=90, num_det=num_det, I0=1e6)
            # rec = physics_energy.fbp(energy_sino)
            # plt.figure(), plt.imshow(transmission, cmap='gray')
            # plt.figure(), plt.imshow(energy_sino, cmap='gray')
            # plt.figure(), plt.imshow(rec.detach().cpu().numpy().squeeze(), cmap='gray')
            # plt.show()

            # UNCOMMENT if you want to save the data
            if kvp == "90":
                np.save(os.path.join(low_sino_save_path, str(kvp) + '_sino_' + str(i)),
                        energy_sino.astype("float32"))

            elif kvp == "150":
                np.save(os.path.join(high_sino_save_path, str(kvp) + '_sino_' + str(i)),
                        energy_sino.astype("float32"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_path', type=str, default='../ddpm_AAPMCT/datasets/mono_1mm')
    parser.add_argument('--bone_save_path', type=str, default='../ddpm_AAPMCT/datasets/bone_1mm')
    parser.add_argument('--water_save_path', type=str, default='../ddpm_AAPMCT/datasets/water_1mm')
    parser.add_argument('--energy_sino_save_path', type=str, default='../data/CT_energysino_1mm_tomo')
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    args = parser.parse_args()
    main(args)



























    # np.save(os.path.join(bone_save_path, 'bone_sino_' + str(i)), s_bone)
    # np.save(os.path.join(water_save_path, 'water_sino_' + str(i)), s_water)

    # plt.figure(1)
    # plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('(a) original image'), plt.axis('off')
    # plt.subplot(132), plt.imshow(img_bone, cmap='gray'), plt.title('(b) bone image'), plt.axis('off')
    # plt.subplot(133), plt.imshow(img_water, cmap='gray'), plt.title('(c) water image'), plt.axis('off')
    # plt.figure(2), plt.imshow(s_bone, cmap='gray')
    # plt.figure(3), plt.imshow(s_water, cmap='gray')
    # plt.show()


