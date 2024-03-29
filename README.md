# ERIS mini tools
Some mini tools for activities on the adaptive optics module

## Installare il pacchetto ERIS da github ##
Per installarlo potendo modificarne il contenuto:
- git clone https://github.com/ChiaraSelmi/ERIS.git
- cd ERIS
- python setup.py install

Per installarlo senza accesso al contenuto:
- pip install git+https://github.com/ChiaraSelmi/ERIS.git

Per scaricare i cambiamenti presenti su github:
- git pull origin master

## Utilizzare il codice di allineamento con puntatore ##
- Far partire la macchina virtuale LGS-Pointer
- Aprire il terminale e digitare conda activate pysilico
- Far partire il server con il comando pysilico_start
- Aprire un altro terminale e digitare conda activate pysilico
- Lanciare python ed eseguire:
  - import pysilico
  - cam = pysilico.camera(IPServer, port)   NOTA:port=7100 IPServer= IP macchina virtuale
  - from eris.pointer_aligner import PointerAligner
  - p = PointerAligner(camera, pointerID) NOTA: pointerID='NGS' or 'LGS'
  - p.main()

## Utilizzare Pysilico Cam ##
- Far partire la macchina virtuale LGS-Pointer
- Aprire un terminale (da dentro una directory perchè il collegamento sul desktop non funziona)
- Dare il comando conda activate pysilico
- Far partire il server con il comando pysilico_start
- Aprire un altro terminare da usare come client e spostarsi nuovamente sull'enviroment pysilico
- Aprire python (ipython --pylab) ed eseguire:
  - import pysilico
  - cam = pysilico.camera(IPServer, port)   NOTA:port=7100 IPServer= IP macchina virtuale
  - cameraFrame = camera.getFutureFrames(numberOfReturnedImages, numberOfFramesToAverageForEachImage)
  - images = cameraFrame.toNumpyArray()

Per modificare le impostazione della camera usare:
- cam.setExposureTime(exposureTimeInMilliSeconds)
- cam.setBinning(binning)
- per tutte le altre funzioni: https://github.com/ArcetriAdaptiveOptics/pysilico/blob/master/pysilico/client/camera_client.py
