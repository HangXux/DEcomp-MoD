import numpy as np
import os
# COMMENT out if you don't have tqdm, and use range instead of trange below
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import glob
import astra



kvp = "70"   # options are "70" (low) and "120" (high)

modeldata = np.load(os.path.join('data', kvp+'kvp_data.npy'))
# modeldata = np.load("model_data_"+kvp+"kVp.npy")  # AAPM

energies = modeldata[0]*1.  # energy bins
spectrum = modeldata[1]*1.  # spectrum
mu_bone = modeldata[2]*1.   # bone mass attenuation coefficient
mu_water = modeldata[3]*1.  # water mass attenuation coefficient

# data path
bone_path = sorted(glob.glob(os.path.join('data/CT_materialsino_180/bone/train', '*.npy')))
water_path = sorted(glob.glob(os.path.join('data/CT_materialsino_180/water/train', '*.npy')))

# save path
low_sino_save_path = os.path.join('data/CT_energysino/train/low')
high_sino_save_path = os.path.join('data/CT_energysino/train/high')
low_transmission_path = os.path.join('data/CT_transmission/train/low')
high_transmission_path = os.path.join('data/CT_transmission/train/high')
os.makedirs(os.path.join(low_sino_save_path), exist_ok=True)
os.makedirs(os.path.join(high_sino_save_path), exist_ok=True)
os.makedirs(os.path.join(low_transmission_path), exist_ok=True)
os.makedirs(os.path.join(high_transmission_path), exist_ok=True)

for i in tqdm(range(len(bone_path))):
   # knock out every other view for kVp switching. First view is 80 kVp, second is 50 kVp, ...
   if kvp == "70":
      sbone = np.load(bone_path[i])[1::2, :]
      ssoft = np.load(water_path[i])[1::2, :]

   elif kvp == "120":
      sbone = np.load(bone_path[i])[0::2, :]
      ssoft = np.load(water_path[i])[0::2, :]

   else:
      print("Bad kVp choice")

   nviews, nbins = sbone.shape  # Grab the data dimensions from the Sinogram data

   # # norm sino
   # sbone = (sbone - sbone.min()) / (sbone.max() - sbone.min())
   # ssoft = (ssoft - ssoft.min()) / (ssoft.max() - ssoft.min())

   delta_e = 0.5  # bin width

   # Demonstrate that spectrum is normalized
   spectnorm = (delta_e*spectrum).sum()  # delta_e factor is used for trapezoid sum approximation for integration over E
   print("Spectrum normalization is: ", spectnorm)

   transmission = np.zeros([nviews, nbins], "float32")
   for j in range(len(energies)):
      # transmission += np.exp(-mu_bone[j] * sbone - mu_water[j] * ssoft)
      # tmp = mu_bone[j] * sbone + mu_water[j] * ssoft
      # max_tmp = tmp.max()
      # transmission += delta_e * spectrum[j] * np.exp(-tmp/max_tmp)
      transmission += delta_e * spectrum[j] * np.exp(-mu_bone[j] * sbone - mu_water[j] * ssoft)

   # add poisson noise (astra)
   # I0 = 1e5
   # sinogramCT = I0 * transmission
   # sinogramCT_C = np.zeros_like(sinogramCT)
   # for m in range(sinogramCT_C.shape[0]):
   #    for n in range(sinogramCT_C.shape[1]):
   #       sinogramCT_C[m, n] = np.random.poisson(sinogramCT[m, n])
   # # sinogramCT_C = np.random.poisson(sinogramCT)
   # # to density
   # # sinogramCT_C = sinogramCT_C + np.finfo(np.float32).eps
   # sinogramCT_D = sinogramCT_C / I0
   # eps = 0.0001
   # energy_sino = -np.log(sinogramCT_D+eps)

   energy_sino = -np.log(transmission)
   # energy_sino = astra.add_noise_to_sino(energy_sino, I0=I0)

   # # generate sinogram
   # sino = -np.log(transmission)

   pg = astra.create_proj_geom('parallel', 1.0, int(512*1.5), np.linspace(0, np.pi, 90, False))
   vg = astra.create_vol_geom(512, 512)
   sino_id = astra.data2d.create('-sino', pg, energy_sino)
   proj_id = astra.create_projector('cuda', pg, vg)

   rec_id = astra.data2d.create('-vol', vg)
   cfg = astra.astra_dict('FBP_CUDA')
   cfg['ReconstructionDataId'] = rec_id
   cfg['ProjectionDataId'] = sino_id
   cfg['option'] = {'FilterType': 'Ram-Lak'}
   alg_id = astra.algorithm.create(cfg)
   astra.algorithm.run(alg_id)
   rec = astra.data2d.get(rec_id)

   plt.figure(1), plt.imshow(transmission, cmap='gray'), plt.axis('off')
   plt.figure(2), plt.imshow(energy_sino, cmap='gray'), plt.axis('off')
   plt.figure(3), plt.imshow(rec, cmap='gray')
   plt.show()

   # UNCOMMENT if you want to save the data
   if kvp == "70":
      np.save(os.path.join(low_sino_save_path, str(kvp)+'_sino_'+str(i)), energy_sino.astype("float32"))
      np.save(os.path.join(low_transmission_path, str(kvp)+'_trans_'+str(i)), transmission.astype("float32"))
   else:
      np.save(os.path.join(high_sino_save_path, str(kvp)+'_sino_'+str(i)), energy_sino.astype("float32"))
      np.save(os.path.join(high_transmission_path, str(kvp)+'_trans_'+str(i)), transmission.astype("float32"))
