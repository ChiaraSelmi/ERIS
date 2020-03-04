'''
@author: cs
'''

import numpy as np
from photutils.datasets import make_gaussian_sources_image
from astropy.table import Table
from photutils.centroids import fit_2dgaussian

class SR_Calculator():
    '''
    '''

    def __init__(self):
        """The constructor """

    def XYCoordForPSFIma(self, psf_ima, pixsize, Fn):
        '''
        arg:
            psf_ima = camera frame containing the PSF
            pixsize = [um] pixel size of camera at telescope focal plane
            Fn = F-number at telescope focal plane
        '''

        par = fit_2dgaussian(psf_ima)._parameters
        fit_ampx = par[2] - np.int(par[2])
        fit_ampy = par[3] - np.int(par[3])
        constant = par[0]

        final_bgr = self.bgr_calc(psf_ima, par)
        new_psf_ima = psf_ima - final_bgr
        new_par = fit_2dgaussian(new_psf_ima)._parameters
        new_fit_ampx = par[2] - np.int(par[2])
        new_fit_ampy = par[3] - np.int(par[3])
        new_constant = par[0]

        return par, new_par

    def bgr_calc(self, psf_ima, par, const_for_rmin=6):
        #refine background rejection
        size = np.array([psf_ima.shape[0], psf_ima.shape[1]])
        ima_x = np.arange(size[0], dtype = float)
        ima_y = np.arange(size[1], dtype = float)

        sigma = np.sqrt(par[4]*par[5])
        size2 = (10,10)
        xx = ima_x.reshape(size2)-par[2]
        yy = ima_y.T.reshape(size2)-par[3]
        rr = np.sqrt(xx**2+yy**2)

        cc = np.array([10,10,10,10])
        rmax = np.min((par[2], par[3], size2[0]-par[2]-1,
                       size2[1]-par[3]-1)-cc).astype(int)
        rmin = np.ceil(const_for_rmin * sigma)
        dr = 3
        #rmax = ((rmax-rmin)/dr)*dr+rmin
        nr = ((np.abs(rmax)-np.abs(rmin))/dr).astype(int)
        vr = np.arange(np.abs(rmin), np.abs(rmax), step=dr)

        bgr = np.zeros(nr-1)
        for i in range(0, nr-1):
            idx = np.where(rr<=vr[i+1], rr,  rr>=vr[i]).astype(int)
            bgr[i] = np.mean(psf_ima[idx])

        final_bgr = np.mean(bgr)
        return final_bgr

    def make_psf_ima(self):
        sources = Table()
        sources['amplitude'] = [1000]
        sources['x_mean'] = [30.6]
        sources['y_mean'] = [20]
        sources['x_stddev'] = [2]
        sources['y_stddev'] = [2]
        sources['theta'] = np.radians(30)

        data = make_gaussian_sources_image((100,100), sources)
        data2 = data + 7
        return data2