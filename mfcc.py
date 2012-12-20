# mfcc.py
# Get mel-frequency cepstral coefficients
#
# Created by Colin Raffel on 11/17/10

import numpy as np
import scipy.signal as signal

def nextpow2(i):
  n = 1
  while n < i:
    n = n * 2
  return n

filters = np.zeros((0, 0))

class MFCC:
  def __init__( self, fs, N, nFilters=40.0, nCC = 13.0 ):
    self.nFilters = nFilters
    self.nCC = nCC
    self.filters = self.getFilters( N, fs )
  
  def getMelSpectrum( self, spectrum ):
    magnitudeSpectrum = np.abs( spectrum )# spectrum.real*spectrum.real + spectrum.imag*spectrum.imag
    return np.dot(magnitudeSpectrum, self.filters)
  
  def getMFCC( self, spectrum ):
    melSpectrum = self.getMelSpectrum( spectrum )
    # This could be optional
    logMelSpectrum = np.log( melSpectrum.clip(1e-5,np.inf) )
    s2dct = self.s2dctmat()
    # Calculate MFCCs and return them
    return np.dot(logMelSpectrum, s2dct.T)/self.nCC

  # Currently we "slope out" some high frequencies, resulting in high frequency loss.
  def getFilters( self, N, fs, lowerFreq=0, upperFreq=None, normalizeFilters=1 ):
    # Build mel filter matrix
    filters = np.zeros((N/2 + 1, self.nFilters), 'd')
    dFreq = fs/(1.0*N)
    if upperFreq > fs/2 or upperFreq is None:
      #raise(Exception, "Upper frequency exceeds Nyquist")
      upperFreq = fs/2
    melMax = self.mel(upperFreq)
    melMin = self.mel(lowerFreq)
    dmelBandwidth = (melMax - melMin)/(self.nFilters + 1)
    # Filter edges, in Hz
    filterEdges = self.melinv(melMin + dmelBandwidth*np.arange(self.nFilters + 2, dtype='d'))
    
    for whichfilt in np.arange(0, self.nFilters):
      # Filter triangles, in DFT points
      leftFreq = np.round(filterEdges[whichfilt]/dFreq)
      centerFreq = np.round(filterEdges[whichfilt + 1]/dFreq)
      rightFreq = np.round(filterEdges[whichfilt + 2]/dFreq)
      # For some reason this is calculated in Hz, though I think
      # it doesn't really matter
      fWidth = (filterEdges[whichfilt + 2]/dFreq  - filterEdges[whichfilt]/dFreq)*dFreq
      if normalizeFilters:
        height = 2.0/fWidth
      else:
        height = 1.0
      
      if centerFreq != leftFreq:
        leftSlope = height/(centerFreq - leftFreq)
      else:
        leftSlope = 0
      freq = leftFreq + 1
      freqs = np.arange(freq, centerFreq, dtype=int)
      filters[freqs, whichfilt] = (freqs - leftFreq)*leftSlope
      freq = centerFreq
      filters[freq, whichfilt] = height
      freq = freq + 1
      if centerFreq != rightFreq:
        rightSlope = height/(centerFreq - rightFreq)
      freqs = np.arange(freq, rightFreq, dtype=int)
      filters[freqs, whichfilt] = (freqs-rightFreq)*rightSlope
    
    return filters
  
  # Get the DCT matrix for calculating the DCT
  def s2dctmat(self):
    melcos = np.empty((self.nCC, self.nFilters), 'double')
    for i in np.arange(0,self.nCC):
      freq = np.pi*float(i)/self.nFilters
      melcos[i] = np.cos(freq * np.arange(0.5, float(self.nFilters)+0.5, 1.0, 'double'))
    melcos[:,0] = melcos[:,0] * 0.5
    return melcos
  
  def mel(self, f):
    return 2595. * np.log10(1. + f / 700.)
  def melinv(self, m):
    return 700. * (np.power(10., m / 2595.) - 1.)