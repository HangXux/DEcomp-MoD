import astra
import numpy as np


def my_add_poisson_noise(sinogram_in, I0):

    sinogramRaw = sinogram_in
    max_sinogramRaw = sinogramRaw.max()
    sinogramRawScaled = sinogramRaw / max_sinogramRaw
    # to detector count
    # sinogramCT = I0 * sinogramRawScaled
    sinogramCT = I0 * np.exp(-sinogramRawScaled)
    # add poison noise
    sinogramCT_C = np.zeros_like(sinogramCT)
    for i in range(sinogramCT_C.shape[0]):
        for j in range(sinogramCT_C.shape[1]):
            sinogramCT_C[i, j] = np.random.poisson(sinogramCT[i, j])
    # to density
    sinogramCT_D = sinogramCT_C / I0
    sinogram_out = -max_sinogramRaw * np.log(sinogramCT_D)

    return sinogram_out

def transform_ctdata(self, windowWidth, windowCenter, normal=False):
    """
    注意，这个函数的self.image一定得是float类型的，否则就无效！
    return: trucated image according to window center and window width
    """

    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (self - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg