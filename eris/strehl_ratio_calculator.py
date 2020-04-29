'''
@author: cs
'''

import os
import numpy as np
from astropy.io import fits as pyfits
from scipy import special
from photutils import background
from photutils.datasets import make_gaussian_sources_image
from astropy.table import Table
from photutils.centroids import fit_2dgaussian

class SR_Calculator():
    '''
    '''

    def __init__(self):
        """The constructor """
        self._pixelCut = 20
        self._pixelsize_camera = 15e-6 #15e-6
        self._Fn_camera = 13.64
        self._lambda = 1300e-9

    def principal_main(self, psf_ima):
        ''' Eseguo il procedimento usando le dimenzioni totali del frame
        e alla fine eseguo la normalizzazione sull'immagine ritaziata al PixelCut
        '''
        psf_ima_cut, y_min, x_min = self.ima_cutter(psf_ima, self._pixelCut)
        par_cut, par_psf_ima = self.fit_2dGaussianForPsfImaCut(psf_ima_cut, y_min, x_min)
        xx, yy, rmin, rmax, final_bgr = self.bgr_calc_rings(psf_ima, par_psf_ima)

        psf_ima_no_bgr = psf_ima - final_bgr

        par_psf_ima_no_bgr = self.fit_2dGaussianForPsfImaWithoutBgr(psf_ima_no_bgr)
        psf_diffraction_limited = self.create_psf_diffraction_limited(xx, yy)

        norm_psf_ima_no_bgr = self._normalizePsf(psf_ima_no_bgr, par_psf_ima_no_bgr[2],
                                                 par_psf_ima_no_bgr[3])
        norm_psf_diffraction_limited = self._normalizePsf(psf_diffraction_limited,
                                                          par_psf_ima_no_bgr[2],
                                                          par_psf_ima_no_bgr[3])

        strehl_ratio = np.max(norm_psf_ima_no_bgr)/np.max(norm_psf_diffraction_limited)

        return strehl_ratio, norm_psf_ima_no_bgr, norm_psf_diffraction_limited
    ### PLOT ###
#        x= np.arange(280,350)
#        plot(x, norm_psf_ima_no_bgr[146,280:350], label='psf_no_bgr');plot(x, norm_psf_diffraction_limited[146,280:350], label='psf_dl'); plt.xlabel('Pixel'); plt.title('SR = %f' %strehl_ratio); plt.legend()
#
#        plot(x, aa[146,280:350], label='psf_no_bgr');plot(x, bb[146,280:350], label='psf_dl'); plt.xlabel('Pixel'); plt.legend()
#
#        x= np.arange(40)
#        plot(x, norm_psf_ima_no_bgr[20,0:40], label='psf_no_bgr');plot(x, norm_psf_diffraction_limited[20,0:40], label='psf_dl'); plt.xlabel('Pixel');plt.title('SR = %f' %strehl_ratio); plt.legend()

    def principal_main_cut(self, psf_ima, cut):
        ''' Passo cut=numero di pixel per ritagliare il frame iniziale
        ed eseguo tutto il procedimento su quella dimenzione lì
        '''
        psf_ima_cut, y_min, x_min = self.ima_cutter(psf_ima, cut)
        par_cut = fit_2dgaussian(psf_ima_cut)._parameters
        xx, yy, rmin, rmax, final_bgr = self.bgr_calc_rings(psf_ima_cut, par_cut)
        print(rmin, rmax)
        psf_ima_no_bgr = psf_ima_cut - final_bgr
        psf_diffraction_limited = self.create_psf_diffraction_limited(xx, yy)

        norm_psf_ima_no_bgr = psf_ima_no_bgr/np.sum(psf_ima_no_bgr)
        norm_psf_diffraction_limited = psf_diffraction_limited/np.sum(psf_diffraction_limited)
        strehl_ratio = np.max(norm_psf_ima_no_bgr)/np.max(norm_psf_diffraction_limited)
        return strehl_ratio, norm_psf_ima_no_bgr, norm_psf_diffraction_limited

    def _normalizePsf(self, psf_to_normalize, x_peak, y_peak):
        x_peak = x_peak.astype(int)
        y_peak = y_peak.astype(int)
        psf_ima_cut = psf_to_normalize[y_peak-self._pixelCut:y_peak+self._pixelCut,
                                       x_peak-self._pixelCut:x_peak+self._pixelCut]
        norm_psf_ima = psf_ima_cut/np.sum(psf_ima_cut)
        return norm_psf_ima

    def seq_sr(self, psf_ima):
        srList = []
        pix = np.arange(239)+15
        prova = np.arange(79)+15
        for i in prova:
            print(i)
            sr, n_bgr,n_dl = self.principal_main_cut(psf_ima, i)
            srList.append(sr)
        return np.array(srList), pix
###
    def _proveBkg(self, psf_ima):
        bkg = background.MeanBackground()
        bkg_value = bkg.calc_background(psf_ima)
        psf_ima_no_bgr = psf_ima - bkg_value
        norm1 = psf_ima_no_bgr/np.sum(psf_ima_no_bgr)

        bkg = background.MedianBackground()
        bkg_value = bkg.calc_background(psf_ima)
        psf_ima_no_bgr = psf_ima - bkg_value
        norm2 = psf_ima_no_bgr/np.sum(psf_ima_no_bgr)

        bkg = background.ModeEstimatorBackground()
        bkg_value = bkg.calc_background(psf_ima)
        psf_ima_no_bgr = psf_ima - bkg_value
        norm3 = psf_ima_no_bgr/np.sum(psf_ima_no_bgr)

        return norm1, norm2, norm3
###

    def fit_2dGaussianForPsfImaCut(self, psf_ima_cut, y_min, x_min):
        '''
        arg:
            psf_ima_cut = camera frame containing the PSF

        return:
            par_cut = parameters of 2D Gaussian fit for cut image
            par_psf_ima = parameters of 2D Gaussian fit for total image
        '''

        par_cut = fit_2dgaussian(psf_ima_cut)._parameters
        #peak off-center in pixel units
        fit_ampx = par_cut[2] - np.int(par_cut[2])
        fit_ampy = par_cut[3] - np.int(par_cut[3])
        constant = par_cut[0]

        par_psf_ima = np.copy(par_cut)
        par_psf_ima[2] = par_cut[2] + x_min #300
        par_psf_ima[3] = par_cut[3] + y_min #140
        return par_cut, par_psf_ima

    def fit_2dGaussianForPsfImaWithoutBgr(self, new_psf_ima):
        #new_psf_ima = new_psf_ima[140:165,300:325]
        new_psf_ima, y_min, x_min = self.ima_cutter(new_psf_ima, self._pixelCut)

        new_par = fit_2dgaussian(new_psf_ima)._parameters
        new_fit_ampx = new_par[2] - np.int(new_par[2])
        new_fit_ampy = new_par[3] - np.int(new_par[3])
        new_constant = new_par[0]

        par_psf_ima_no_bgr = np.copy(new_par)
        par_psf_ima_no_bgr[2] = new_par[2] + x_min #300
        par_psf_ima_no_bgr[3] = new_par[3] + y_min #140
        return par_psf_ima_no_bgr

    def bgr_calc_rings(self, psf_ima, par, const_for_rmin=6):
        '''
        arg:
            psf_ima = camera frame containing the PSF
            par = parameters of 2D Gaussian fit
            const_for_rmin = value to be multiplied at sigma to obtain r_min

        return:
            xx = 2d coordinates used to map Gaussian radially
            yy = 2d coordinates used to map Gaussian radially
            rmin = minimum radius used for bgr calculation
            rmax = minimum radius used for bgr calculation
            final_bgr = psf_ima background (int)
        '''
        size = np.array([psf_ima.shape[0], psf_ima.shape[1]])
        ima_x = np.arange(size[0], dtype = float)
        ima_y = np.arange(size[1], dtype = float)

        sigma = np.sqrt(np.abs(par[4]*par[5]))  ###ATTENZIONE###
        xx = np.tile(ima_x, (size[0], 1))-par[2]
        yy = np.tile(ima_y, (size[1], 1)).T-par[3]
        rr = np.sqrt(xx**2+yy**2)

        cc = np.array([10,10,10,10])
        rmax = np.min((par[2], par[3], size[0]-par[2]-1,
                       size[1]-par[3]-1)).astype(int)   #-cc
        rmin = np.ceil(const_for_rmin * sigma)
        dr = 3
        #rmax = ((rmax-rmin)/dr)*dr+rmin
        nr = (np.abs(rmax-rmin)/dr).astype(int)
        vr = np.arange(np.abs(rmin), np.abs(rmax), step=dr)

        bgr = np.zeros(nr-1)
        for i in range(0, nr-1):
            circ1 = (rr>=vr[i]).astype(int)
            circ2 = (rr<=vr[i+1]).astype(int)
            ring = circ1 * circ2
            idx = np.where(ring == 1)
            bgr[i] = np.mean(psf_ima[idx])

        #bgr_cut = self.bgr_cut(bgr)
        final_bgr = np.mean(bgr)
        return xx, yy, rmin, rmax, final_bgr
        ### PLOT ###
#        axs = vr[0:vr.size-1]
#        plot(axs, bgr); plot(axs, bgr, 'o'); plt.xlabel('r [px]'); plt.ylabel('bgr'); plt.title('bgr medio = %f' %np.mean(bgr))


    def create_psf_diffraction_limited(self, x_coord, y_coord):
        ''' create psf diffraction limited for a circular pupil with
        a inner obscuration central zone.

        args:
            self._lambda = lambda
            self._pixsize = [um] pixel size of camera at telescope focal plane
            self._Fn = F-number at telescope focal plane
            x_coord = 2d coordinates to create the psf
            y_coord = 2d coordinates to create the psf
            rmin = diametro del secondario
            rmax = diametro del primario

        returns:
            psf_diff = psf diffraction limited
        '''
        #epsilon = rmin / rmax = 1.2 / 8.2 metri
        epsilon = 0.14
        r = np.sqrt(x_coord**2 + y_coord**2)* self._pixelsize_camera/(self._lambda * self._Fn_camera)
        #r = np.linspace(1,20,500)

        psf = (1. / (1- epsilon**2)**2) * ((2. * special.jv(1, np.pi * r))/(np.pi * r) -
                                      epsilon**2 * 2. * special.jv(1, np.pi * r * epsilon)/
                                      (np.pi * r * epsilon))**2

        return psf

    def ima_cutter(self, image, px):
        y_peak, x_peak = np.where(image == np.max(image))
        y_min = y_peak[0]-px
        x_min = x_peak[0]-px
        image_cut = image[y_min:y_peak[0]+px+1, x_min:x_peak[0]+px+1]
        return image_cut, y_min, x_min

    def read_psf_file_fits(self, file_name):
        folder = '/Users/rm/Desktop/Arcetri/ERIS/Python/immaginiperiltestdellamisuradellosr'
        #file_name = 'ERIS_IRCAM_2020-01-24T11_31_30.299.fits'
        #file_name = 'ERIS_IRCAM_2020-01-24T15_17_12.503.fits'
        #file_name = 'ERIS_IRCAM_2020-01-24T15_21_14.162.fits'
        #file_name = 'ERIS_IRCAM_2020-01-27T12_20_53.915.fits'    viene una sigma negativa
        #file_name = 'ERIS_IRCAM_2020-01-27T12_21_14.940.fits'
        #file_name = 'ERIS_IRCAM_2020-01-27T12_21_25.565.fits'
        fits_file_name = os.path.join(folder, file_name)
        hduList = pyfits.open(fits_file_name)
        psf_ima= hduList[0].data
        #psf_ima_cut = psf_ima[140:165,300:325]
        return psf_ima

    ###
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

    def _esercizio(self):
        ima = np.zeros((512,512))+10
        ima_rum = ima + np.random.poisson(lam=10.0,size=(512,512))
        ima_no_bgr = ima_rum - np.mean(ima_rum)
        en = np.sum(ima_no_bgr)
        return en
    ###
