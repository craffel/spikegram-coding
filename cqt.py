# cqt.py
# Calculate constant-Q transform
#
# Created by Colin Raffel on 10/9/12

import numpy as np
from scipy import sparse
import utility

class ConstantQTransformer:
  def __init__(self, fs, fftLen=None, minFreq=32.7031956626, maxFreq=8372.018089619156, bins=24, thresh=0.0054):
    Q = 1.0/(2.0**(1.0/bins)-1.0)
    K = np.ceil( bins * np.log2(maxFreq/minFreq) )
    if fftLen is None:
      fftLen = 1
      while fftLen < np.ceil(Q*fs/minFreq):
        fftLen = fftLen*2
    tempKernel = np.zeros(fftLen, dtype=complex)
    sparKernel = np.zeros((K, fftLen), dtype=complex)
    
    for k in np.arange(K-1, -1, -1):
      N = np.ceil( Q * fs / (minFreq*2.0**(k/bins)) )
      tempKernel[0:N] = (np.hamming(N)/N) * np.exp(2*np.pi*(1j)*Q*np.arange(N)/N)
      specKernel = np.fft.fft(tempKernel)
      specKernel[np.abs(specKernel)<=thresh] = 0;
      sparKernel[k,:] = specKernel;
    
    sparKernel = np.conj(sparKernel)/fftLen
    self.sparseKernel = sparse.lil_matrix(sparKernel, dtype=complex)

  # Fast constant-Q transform
  def fcqt( self, x ):
    N = self.sparseKernel.shape[1]
    if x.shape[0] > N:
      print "Warning: Signal vector size is bigger than Constant Q kernel, some cropping will result!"
    X = np.fft.fft(x, N)
    return self.sparseKernel.dot(X);

  def getSTCQT( self, data, hop = 512, frameSize = 1024, window = np.hanning( 1024 ) ):
    # Framify the signal
    dataSplit = utility.splitSignal( data, hop, frameSize )
    # Create spectrogram array
    spectrogram = np.zeros( (dataSplit.shape[0], self.sparseKernel.shape[0]), dtype = np.complex )
    # Get spectra
    for n in np.arange( dataSplit.shape[0] ):
      spectrogram[n] = self.fcqt( dataSplit[n] )
    return spectrogram
