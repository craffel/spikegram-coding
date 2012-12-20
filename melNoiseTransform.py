# melNoiseTransform.py
# Given a (noise) signal, get the mel-scale representation, and inverse
#
# Created by Colin Raffel on 12/4/12

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import mfcc
import utility

class melTransformer:
  def __init__( self, fs ):
    self.fs = 1.0*fs
    self.frameSize = 512
    self.nBands = 20.0
    self.MFCCer = mfcc.MFCC( fs, self.frameSize, self.nBands )

  def getMelGram( self, signal, hop=64 ):
    spectrogram = utility.getSpectrogram( signal, frameSize=self.frameSize, hop=hop )
    return np.array( map( self.MFCCer.getMelSpectrum, spectrogram ) )

  def getSignalFromMelGram( self, melGram, hop=64 ):
    hopScale = 4
    randomSignal = 2*np.random.rand( melGram.shape[0]*hop + hop*hopScale ) - 1
    randomSpectrogram = utility.getSpectrogram( randomSignal, hop=hop, frameSize=hop*hopScale )
    #randomSpectrogram = np.random.randn( melGram.shape[0], hop*hopScale/2 + 1  ) + np.random.randn( melGram.shape[0], hop*hopScale/2 + 1 )*1j
    #randomSpectrogram[:, 0] = randomSpectrogram[:, 0].real + 0j
    #randomSpectrogram[:, -1] = randomSpectrogram[:, -1].real + 0j
    spectrogramScaler = np.zeros( randomSpectrogram.shape )
    filters = self.MFCCer.getFilters( hop*hopScale, self.fs, 0, None, 0 )
    for n in xrange( randomSpectrogram.shape[0] ):
      spectrogramScaler[n] = np.sum( filters*melGram[n], axis=1 )
      #spectrogramScaler[n] = np.sum( filters, axis=1 )
    return utility.getSignalFromSpectrogram( randomSpectrogram*spectrogramScaler, hop, np.hanning( hop*hopScale ) )