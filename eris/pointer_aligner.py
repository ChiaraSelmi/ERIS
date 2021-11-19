'''
Authors
  - C. Selmi: written in 2021
'''

import os
import time
import numpy as np
from photutils import centroids
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from eris.type.pointer_align import PointerAlign
from eris.ground import tracking_number_folder
from IPython.display import clear_output


IP = '192.168.1.18'
PORT = 7100 #ma anche 7110

class PointerAligner():
    '''
    https://github.com/ArcetriAdaptiveOptics/pysilico
    '''

    def __init__(self, pysilico, pointerId):
        '''
        Parameters
        ----------
        pysilico: object
            camera for frame acquisition
        poniterId: string
            'NGS' or 'LGS'
        '''
        self._camera = pysilico
        self._idData = PointerAlign(pointerId)
        self._tt = None
        self._dove = None

    def main(self):
        self._dove, self._tt = tracking_number_folder.createFolderToStoreMeasurements(self._idData.folder)
        self._idData.dove = self._dove
        self._idData.tt = self._tt
        y00, y01, image_a = self._dyCalculator()
        #self._saveImage(image_a, 'xTilt.fits')
        print('Move camera away')
        self._pause() #spostare camera
        y11, y10, image_b = self._dyCalculator()
        #self._saveImage(image_b, 'yTilt.fits')

        point0, point1 = self._idData.operationAndPrint(y00, y01, y11, y10)
        self._idData.saveInfo()

        x, y = self._centroid_calculator(image_b)
        doagain = 1
        while doagain ==1:
            for i in range(0, 10):
                self.target_center(point0, x) #1 vite
                time.sleep(1)
            doagain = int(input('Press 1 to repeat, 0 to exit:'))

        doagain = 1
        while doagain ==1:
            for i in range(0, 10):
                self.target_center(point1, x) #2 vite
                time.sleep(1)
            doagain = int(input('Press 1 to repeat, 0 to exit:'))


    def _dyCalculator(self):
        image0 = self.take_images(1)
        coord0 = self._centroid_calculator(image0)
        print('Rotate ±180°')
        self._pause() #rotazione
        image1 = self.take_images(1)
        coord1 = self._centroid_calculator(image1)

        dy = coord1[1]-coord0[1]
        print('Y offset [um]')
        print(dy*self._idData.pixs*1e6) #(dy*self._pixs/self._dl * self._rad2as)

        image = image0 + 2*image1
        self._plot(image)
        return coord0[1], coord1[1], image1


    def target_center(self, point, x):
        '''
        Function that acquires an image and draws a cross on the input point

        Parameters
        ----------
        point: list
            point y coord
        '''
        image = self.take_images(1)
        imm = image.copy()
        ymin = np.int32(np.round(point-5))
        ymax = np.int32(np.round(point+5))
        imm[ymin:ymax, :] = imm[ymin:ymax, :]*4
        roi = 100
        x = np.int32(np.round(x))
        point = np.int32(np.round(point))
        imm1 = imm[point-roi:point+roi, x-roi:x+roi]
        self._plot(imm1)
        
    def _plot(self, imm):
        clear_output(wait=True)
        plt.imshow(imm, origin='lower')
        plt.show()
        plt.pause(0.1)
        #size = 250
        #plt.scatter(point, point, s=size, c='red', marker='+')

    def _saveImage(self, image, name):
        fits_file_name = os.path.join(self._dove, name)
        pyfits.writeto(fits_file_name, image)

    def _pause(self):
        '''this function will pause the script with a default massage'''
        os.system("read -p 'Press Enter to continue...' var")
        return

    def take_images(self, numberOfReturnedImages):
        '''
        Parameters
        ----------
        numberOfReturnedImages: int
            numebers of sequential images

        Returns
        -------
        images: ?
        '''
        self._camera.setExposureTime(self._idData.exposureTimeInMilliSeconds)
        images = self._camera.getFutureFrames(numberOfReturnedImages,
                                     numberOfFramesToAverageForEachImage=1)
                                    #timeout
        return images.toNumpyArray()

    def _centroid_calculator(self, data):
        '''
        Parameters
        ----------
        data: numpy array
            image

        Returns
        -------
        coord: list
            centroid's coordinates
        '''
        #x, y = centroids.centroid_com(data, mask=None) #Calculates the object “center of mass” from 2D image moments
        #x, y = centroids.centroid_1dg(data, error=None, mask=None) #Calculates the centroid by fitting 1D Gaussians to the marginal x and y distributions of the data.
        x, y = centroids.centroid_2dg(data, error=None, mask=None) #Calculates the centroid by fitting a 2D Gaussian to the 2D distribution of the data.
        #https://photutils.readthedocs.io/en/v0.3.1/photutils/centroids.html
        #par = centroids.fit_2dgaussian(data)._parameters #uguale a 2dg
        return [x, y]

    def setExposureTime(self, exposureTimeInSeconds):
        self._idData.exposureTimeInMilliSeconds = exposureTimeInSeconds * 1e3

    def getExposureTime(self):
        return self._idData.exposureTimeInMilliSeconds * 1e-3

