'''
@author: cs
'''

import os
import numpy as np
from astropy.io import fits as pyfits
from scipy import special
from photutils.datasets import make_gaussian_sources_image
from astropy.table import Table
from photutils.centroids import fit_2dgaussian

class SR_Calculator():
    '''
    '''

    def __init__(self):
        """The constructor """
        self._pixelsize_camera = 15e-6
        self._Fn_camera = 13.64
        self._lambda = 1300e-9

    def XYCoordForPSFIma(self, psf_ima):
        '''
        arg:
            psf_ima = camera frame containing the PSF
            pixsize = [um] pixel size of camera at telescope focal plane
            Fn = F-number at telescope focal plane

        return:
        '''

        par = fit_2dgaussian(psf_ima)._parameters
        #peak off-center in pixel units
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
        '''
        arg:
            psf_ima = camera frame containing the PSF
            par = parameters of 2D Gaussian fit
            const_for_rmin = value to be multiplied at sigma to obtain r_min

        return:
            final_bgr = psf_ima background
        '''
        #refine background rejection
        size = np.array([psf_ima.shape[0], psf_ima.shape[1]])
        ima_x = np.arange(size[0], dtype = float)
        ima_y = np.arange(size[1], dtype = float)

        sigma = np.sqrt(par[4]*par[5])
        xx = np.tile(ima_x, (size[0], 1))-par[2]
        yy = np.tile(ima_y.T, (size[1], 1))-par[3]
        rr = np.sqrt(xx**2+yy**2)

        cc = np.array([10,10,10,10])
        rmax = np.min((par[2], par[3], size[0]-par[2]-1,
                       size[1]-par[3]-1)-cc).astype(int)
        rmin = np.ceil(const_for_rmin * sigma)
        dr = 3
        #rmax = ((rmax-rmin)/dr)*dr+rmin
        nr = ((np.abs(rmax)-np.abs(rmin))/dr).astype(int)
        vr = np.arange(np.abs(rmin), np.abs(rmax), step=dr)

        bgr = np.zeros(nr-1)
        for i in range(0, nr-1):
            idx = np.where(rr>=vr[i], rr, rr<=vr[i+1]).astype(int)
            idx[np.where(idx==1)]=0
            bgr[i] = np.mean(psf_ima[idx])

        #togliere i primi valori
        final_bgr = np.mean(bgr)
        return final_bgr


    def create_psf_diffraction_limited(self, x_coord, y_coord, rmin, rmax):
        ''' create psf diffraction limited for a circular pupil with
        a inner obscuration central zone.

        args:
            wl = lambda
            Fn = F-number at telescope focal plane
            x_coord =
            y_coord =

        returns:
            psf_diff = psf diffraction limited
        '''
        epsilon = rmin / rmax
        r = np.sqrt(x_coord**2 + y_coord**2)/(self._lambda * self._Fn_camera)
        #r = np.linspace(1,20,500)

        psf = (1. / (1- epsilon**2)**2) * ((2. * special.jv(1, np.pi * r))/(np.pi * r) -
                                      epsilon**2 * 2. * special.jv(1, np.pi * r * epsilon)/
                                      (np.pi * r * epsilon))**2

        return psf

    def make_psf_ima(self):
        ''' Create a gaussian distribution to use as test psf_ima
        '''
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

    def read_psf_file_fits(self):
        folder = '/Users/rm/Desktop/Arcetri/ERIS/Python/immaginiperiltestdellamisuradellosr'
        file_name = 'ERIS_IRCAM_2020-01-24T11_31_30.299.fits'
        fits_file_name = os.path.join(folder, file_name)
        hduList = pyfits.open(fits_file_name)
        ima= hduList[0].data
        psf_ima = ima[140:165,300:325]
        return psf_ima







    ### Roba ###

#             psf_list = []
#         for i in range(r.shape[0]):
#             if r[i]== 0:
#                 psf_zero = 1.
#                 psf_list.append(psf_zero)
#             else:
#                 psf = (1. / (1- epsilon**2)**2) * ((2. * special.jv(1, np.pi * r[i]))/(np.pi * r[i]) -
#                                       epsilon**2 * 2. * special.jv(1, np.pi * r[i] * epsilon)/
#                                       (np.pi * r[i] * epsilon))**2
#                 psf_list.append(psf)
# 
#         final_psf = np.array(psf_list)
#         