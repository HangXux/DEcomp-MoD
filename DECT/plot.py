import matplotlib.pyplot as plt
import numpy as np
import astra

low_trans = np.load('data/lowkVpTransmission.npy')
high_trans = np.load('data/highkVpTransmission.npy')
low_sino = np.load('data/lowkvp_sinogram.npy')
high_sino = np.load('data/highkvp_sinogram.npy')

plt.figure(1), plt.imshow(low_sino, cmap='gray')
plt.figure(2), plt.imshow(high_sino, cmap='gray')

pg = astra.create_proj_geom('parallel', 1.0, 750, np.linspace(0, np.pi, 512, False))
vg = astra.create_vol_geom(512, 512)
sino_id = astra.data2d.create('-sino', pg, low_sino)
proj_id = astra.create_projector('cuda', pg, vg)

rec_id = astra.data2d.create('-vol', vg)
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sino_id
cfg['option'] = {'FilterType': 'Ram-Lak'}
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
rec = astra.data2d.get(rec_id)

plt.figure(3), plt.imshow(rec, cmap='gray')
plt.show()

plt.imsave('figure/lowkvp_sino.png', low_sino, cmap='gray')
plt.imsave('figure/highkvp_sino.png', high_sino, cmap='gray')

