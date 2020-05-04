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

        ring_or_circle = int 0 per ring, 1 per circle
        '''
        psf_ima_cut, y_min, x_min = self.ima_cutter(psf_ima, self._pixelCut)
        par_cut, par_psf_ima = self.fit_2dGaussianForPsfImaCut(psf_ima_cut, y_min, x_min)
        xx, yy, rr, rmin, rmax, nr, vr = self.bgr_parameters(psf_ima, par_psf_ima)

        final_bgr = self._bgrCircle(psf_ima, rr, nr, vr)
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
        xx, yy, rr, rmin, rmax, nr, vr = self.bgr_parameters(psf_ima_cut, par_cut)
        print(rmin, rmax)
        final_bgr = self._bgrRings(psf_ima, rr, nr, vr)
        psf_ima_no_bgr = psf_ima_cut - final_bgr
        psf_diffraction_limited = self.create_psf_diffraction_limited(xx, yy)

        norm_psf_ima_no_bgr = psf_ima_no_bgr/np.sum(psf_ima_no_bgr)
        norm_psf_diffraction_limited = psf_diffraction_limited/np.sum(psf_diffraction_limited)
        strehl_ratio = np.max(norm_psf_ima_no_bgr)/np.max(norm_psf_diffraction_limited)
        return strehl_ratio, norm_psf_ima_no_bgr, norm_psf_diffraction_limited

    def principal_main_tot(self, psf_ima):
        ''' usa il coefficiente trovato dal fit per normalizzare la psf
        Non torna proprio
        '''
        psf_ima_cut, y_min, x_min = self.ima_cutter(psf_ima, self._pixelCut)
        par_cut, par_psf_ima = self.fit_2dGaussianForPsfImaCut(psf_ima_cut, y_min, x_min)
        xx, yy, rr, rmin, rmax, nr, vr = self.bgr_parameters(psf_ima, par_psf_ima)
        final_bgr = self._bgrRings(psf_ima, rr, nr, vr)
        psf_ima_no_bgr = psf_ima_cut - final_bgr
        par_psf_ima_no_bgr = self.fit_2dGaussianForPsfImaWithoutBgr(psf_ima_no_bgr)

        tot, nq, a, b = self._totAreaCircle(psf_ima, rr, nr, vr)

        psf_diffraction_limited = self.create_psf_diffraction_limited(xx, yy)
        norm_psf_diffraction_limited = self._normalizePsf(psf_diffraction_limited,
                                                          par_psf_ima[2],
                                                          par_psf_ima[3])

        x_peak = par_psf_ima_no_bgr[2].astype(int)
        y_peak = par_psf_ima_no_bgr[3].astype(int)
        psf_ima_cut_to_norm = psf_ima_no_bgr[y_peak-self._pixelCut:y_peak+self._pixelCut,
                                       x_peak-self._pixelCut:x_peak+self._pixelCut]
        norm_psf_ima = psf_ima_cut_to_norm / b
        strehl_ratio = np.max(norm_psf_ima)/np.max(norm_psf_diffraction_limited)
        return strehl_ratio, norm_psf_ima, norm_psf_diffraction_limited

    def _normalizePsf(self, psf_to_normalize, x_peak, y_peak):
        x_peak = x_peak.astype(int)
        y_peak = y_peak.astype(int)
        psf_ima_cut = psf_to_normalize[y_peak-self._pixelCut:y_peak+self._pixelCut,
                                       x_peak-self._pixelCut:x_peak+self._pixelCut]
        norm_psf_ima = psf_ima_cut/np.sum(psf_ima_cut)
        return norm_psf_ima

    def seq_sr(self, psf_ima, pix):
        ''' Caso taglio dopo
        Fa la sequenza di calcolo del valore si sr a seconda del numero di pixel
        '''
        srList = []
        for i in pix:
            print(i)
            self._pixelCut = i
            sr, n_bgr,n_dl = self.principal_main(psf_ima)
            srList.append(sr)
        return np.array(srList), pix
    ###PLOT
    # plot(pix, sh); plt.xlabel('pixel'); plt.ylabel('strehl ratio'); plt.title('SR_max = %f' %np.max(sh))

    def seq_sr_cut(self, psf_ima):
        ''' caso taglio prima
        Fa la sequenza di calcolo del valore si sr a seconda del numero di pixel
        '''
        srList = []
        #pix = np.arange(239)+15
        prova = np.arange(79)+12
        prova2 = np.arange(6)+95
        prova3 = np.append(np.array([105,106,108,109,110,112,113,114,115]), np.arange(121,127))
        #prova4 = np.append(np.array([128,129,131,132,134,137]), np.arange(140,144))
        p = np.append(prova, prova2)
        pp = np.append(p, prova3)

        pix = p
        for i in pix:
            print(i)
            sr, n_bgr,n_dl = self.principal_main_cut(psf_ima, i)
            srList.append(sr)
        return np.array(srList), pix
    ###PLOT
    #  plot(pix, sh); plt.xlabel('pixel'); plt.ylabel('strehl ratio'); plot(pix,x, label='uno'); plt.legend()                                        

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

    def scalino(self, psf_ima):
        mean_list = []
        for i in range(psf_ima.shape[1]):
            a = psf_ima[:,i].mean()
            mean_list.append(a)
        return np.array(mean_list)

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

    def bgr_parameters(self, psf_ima, par, const_for_rmin=6):
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
                        final_bgr[0] = ring method
                        final_bgr[1] = circle method
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
        #rmax = 200 e 250
        nr = (np.abs(rmax-rmin)/dr).astype(int)
        vr = np.arange(np.abs(rmin), np.abs(rmax), step=dr)
        return xx, yy, rr, rmin, rmax, nr, vr
        ### PLOT ###
#        axs = vr[0:vr.size-1]
#        plot(axs, bgr); plot(axs, bgr, 'o'); plt.xlabel('r [px]'); plt.ylabel('bgr'); plt.title('bgr medio = %f' %np.mean(bgr))

    def _bgrRings(self, psf_ima, rr, nr, vr):
        ''' Calcola bgr sugli anelli
        '''
        bgr = np.zeros(nr-1)
        for i in range(0, nr-1):
            circ1 = (rr>=vr[i]).astype(int)
            circ2 = (rr<=vr[i+1]).astype(int)
            ring = circ1 * circ2
            idx = np.where(ring == 1)
            bgr[i] = np.mean(psf_ima[idx])
        #bgr_cut = self.bgr_cut(bgr)
        final_bgr = np.mean(bgr)
        return final_bgr

    def _bgrCircle(self, psf_ima, rr, nr, vr):
        ''' Calcola il bgr su cerchi pieni sempre più grandi
        '''
        bgr = np.zeros(nr-1)
        for i in range(0, nr-1):
            circ2 = (rr<=vr[i+1]).astype(int)
            idx = np.where(circ2 == 1)
            bgr[i] = np.mean(psf_ima[idx])
        final_bgr = np.mean(bgr[40:])
        return final_bgr

    def _totAreaCircle(self, psf_ima, rr, nr, vr):
        ''' Dei cerchi pieni sempre più grandi fa il totale dentro
        il fit dovrebbe caratterizzare il bgr
        '''
        tot = np.zeros(nr-1)
        for i in range(0, nr-1):
            circ2 = (rr<=vr[i+1]).astype(int)
            idx = np.where(circ2 == 1)
            tot[i] = np.sum(psf_ima[idx])
        diff = vr.shape[0] - tot.shape[0]
        axs = vr[0:vr.size-diff]
        n_pix = axs**2
        # fit che restituisce a = media bgr; b = totale psf senza background
        a, b = np.polyfit(n_pix, tot, 1)
        return tot, n_pix, a, b


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
