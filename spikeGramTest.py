# spikeGramTest.py
# A test implementation of spike gram coding using matching pursuit
#
# Created by Colin Raffel on 10/30/12

import numpy as np
import matplotlib.pyplot as plt
import utility
import sys
import os
import optparse
import scipy.signal
import multiprocessing as mp
import time

import logging
logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)



# Computes an array of N frequencies uniformly spaced on an ERB scale
# Based on Slaney's Auditory Toolbox implementation
# Verified same output for input params 100, 44100/4, 100.
def ERBSpace( lowFreq, highFreq, N ):
  # Change the following three parameters if you wish to use a different
  # ERB scale.  Must change in MakeERBCoeffs too.
  # Glasberg and Moore Parameters
  EarQ = 9.26449
  minBW = 24.7
  order = 1

  # All of the followFreqing expressions are derived in Apple TR #35, "An
  # Efficient Implementation of the Patterson-Holdsworth Cochlear
  # Filter Bank."  See pages 33-34.
  return -(EarQ*minBW) + np.exp(np.arange(1, N+1)*(-np.log(highFreq + EarQ*minBW) + np.log(lowFreq + EarQ*minBW))/(1.0*N))*(highFreq + EarQ*minBW);

# Computes the filter coefficients for gammatone filters.
# Based on Slaney's Auditory Toolbox implementation.
# Verified same output for params 44100, 16, 100.
def makeERBFilters( fs, numChannels, lowFreq ):

  T = 1.0/fs
  cf = ERBSpace(lowFreq, fs/2, numChannels)

  # Change the followFreqing three parameters if you wish to use a different
  # ERB scale.  Must change in ERBSpace too.
  # Glasberg and Moore Parameters
  EarQ = 9.26449
  minBW = 24.7
  order = 1

  ERB = ((cf/EarQ)**order + minBW**order)**(1.0/order)
  B = 1.019*2*np.pi*ERB

  A0 = T
  A2 = 0
  B0 = 1
  B1 = -2*np.cos(2*cf*np.pi*T)/np.exp(B*T)
  B2 = np.exp(-2*B*T)

  A11 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0
  A12 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) - 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0
  A13 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0
  A14 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) - 2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0

  gain = abs((-2*np.exp(4*1j*cf*np.pi*T)*T + \
              2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) - np.sqrt(3 - 2**(3/2.0))* \
               np.sin(2*cf*np.pi*T))) * \
             (-2*np.exp(4*1j*cf*np.pi*T)*T + \
              2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) + np.sqrt(3 - 2**(3/2.0)) * \
               np.sin(2*cf*np.pi*T)))* \
             (-2*np.exp(4*1j*cf*np.pi*T)*T + \
              2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) - \
               np.sqrt(3 + 2**(3/2.0))*np.sin(2*cf*np.pi*T))) * \
             (-2*np.exp(4*1j*cf*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) + np.sqrt(3 + 2**(3/2.0))*np.sin(2*cf*np.pi*T))) / \
             (-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*np.pi*T) +  \
              2*(1 + np.exp(4*1j*cf*np.pi*T))/np.exp(B*T))**4)

  allfilts = np.ones( cf.shape[0] )
  return np.dstack( (A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain) )[0]

# Process an input waveform with a gammatone filter bank
# Based on Slaney's Auditory Toolbox implementation
def ERBFilterBank(x, fcoefs ):
  A0  = fcoefs[:,0]
  A11 = fcoefs[:,1]
  A12 = fcoefs[:,2]
  A13 = fcoefs[:,3]
  A14 = fcoefs[:,4]
  A2  = fcoefs[:,5]
  B0  = fcoefs[:,6]
  B1  = fcoefs[:,7]
  B2  = fcoefs[:,8]
  gain= fcoefs[:,9]
  
  output = np.zeros( (gain.shape[0], x.shape[0]) )
  for chan in np.arange( gain.shape[0] ):
    y1 = scipy.signal.lfilter(np.array([A0[chan]/gain[chan], A11[chan]/gain[chan], \
               A2[chan]/gain[chan]]), \
              np.array([B0[chan], B1[chan], B2[chan]]), x);
    y2 = scipy.signal.lfilter(np.array([A0[chan], A12[chan], A2[chan]]), \
              np.array([B0[chan], B1[chan], B2[chan]]), y1);
    y3 = scipy.signal.lfilter(np.array([A0[chan], A13[chan], A2[chan]]), \
              np.array([B0[chan], B1[chan], B2[chan]]), y2);
    y4 = scipy.signal.lfilter(np.array([A0[chan], A14[chan], A2[chan]]), \
              np.array([B0[chan], B1[chan], B2[chan]]), y3);
    output[chan, :] = y4

  return output

# Converts a matrix of ERB filters from makeERBFilters to kernels
# It would be cool to make this non-ERB dependent, to try other filters.
def ERBFiltersToKernels( fcoefs, threshold = .001 ):
  # Create impulse
  impulse = np.zeros( 10000 )
  impulse[0] = 1.0
  # Get impulse responses
  impulseResponses = ERBFilterBank( impulse, fcoefs )
  # Dictionary for gammatone kernels
  kernelDictionary = {}
  for n in np.arange( impulseResponses.shape[0] ):
    impulseResponse = impulseResponses[n]
    impulseResponsePeak = np.max( impulseResponse )
    # Find index of last value greater than the threshold
    trim = 0
    for m in np.arange( impulseResponse.shape[0] ):
      if impulseResponse[m] > threshold*impulseResponsePeak:
        trim = m
    # Trim the impulse response to this value and store in the dictionary
    kernelDictionary[n] = impulseResponse[:trim]
    # Normalize
    kernelDictionary[n] /= np.sqrt( np.sum( kernelDictionary[n]**2 ) )
  return kernelDictionary

# Encodes an input signal "x" with elements from "dictionary"
def matchingPursuit( dictionary, x, threshold = .1 ):
  
  # Find the biggest kernel
  biggestKernelSize = 0
  for n in dictionary:
    if dictionary[n].shape[0] > biggestKernelSize:
      biggestKernelSize = dictionary[n].shape[0]
  
  # Residual of the encoding process, initialize to x with zero padding
  residual = np.append( np.append( np.zeros( biggestKernelSize), x ), np.zeros(biggestKernelSize) )
  # Number of kernels
  nKernels = len( dictionary )
  # Make sure input signal is longer than all kernels
  for n in np.arange( nKernels ):
    assert( dictionary[n].shape[0] < residual.shape[0] )
  
  # Hold onto the cross-correlations at each iteration
  correlations = np.zeros( (nKernels, residual.shape[0]) )
  # Keep track of the iterations
  currentIteration = 0
  # Max # of iterations
  maxIterations = 50000
  # Where we'll be storing the kernels, scales, and offsets
  scales = np.zeros( maxIterations )
  kernels = np.zeros( maxIterations )
  offsets = np.zeros( maxIterations )
  
  # For keeping track of how long we've run
  lastTime = time.time()
  # For checking whether we've converged
  currentResidualMax = np.inf

  threshold = threshold*np.max( np.abs( x ) )
  
  # Parallelize cross-correlation computation...
  def __consumer(in_Q, out_Q):
    while True:
      # Until you get an empty queue exception
      try:
        i = in_Q.get(True, 1)
        out_Q.put( (i, np.correlate( residual, dictionary[i], 'same') ) )
      except:
        break
    out_Q.close()

  # Until the max of the residual is smaller than our threshold value
  while currentResidualMax > threshold:
    
    '''# Create queues to hold input indices and output index/correlation pairs
    in_Q = mp.Queue()
    out_Q = mp.Queue()
    for i in np.arange( nKernels ):
      in_Q.put( i )

    # Number of cores!
    nCores = 4
    for i in np.arange( nCores ):
      mp.Process( target=__consumer, args=(in_Q, out_Q) ).start()
          
    for j in np.arange( nKernels ):
      (i, v) = out_Q.get( True )
      correlations[i] = v
    '''
    
    # The old way of doing it
    for n in xrange( nKernels ):
      # On first iteration, do the whole correlation
      if currentIteration == 0:
        correlations[n] = np.correlate( residual, dictionary[n], 'same' )
      # On subsequent iterations, only update the part of the correlation that has changed
      else:
        # Where does the change in the residual start?
        changeStart = offsets[currentIteration - 1]
        # Look up the previous kernel subtracted from the residual - its size is where the change in the residual ends
        changeEnd = changeStart + dictionary[kernels[currentIteration - 1]].shape[0]
        # What's the size of current kernel to be correlated?
        kernelSize = dictionary[n].shape[0]
        # Where should we start correlating this kernel for no edge effects?
        correlationStart = changeStart - 2*kernelSize
        # This should never happen, but just in case
        if correlationStart < 0:
          correlationStart = 0
        # Correlate with no edge effects
        correlation = np.correlate( residual[correlationStart:changeEnd + kernelSize - 1], dictionary[n], 'valid' )
        # [:correlation.shape[0]] is an indexing hack to avoid shape mismatches
        correlations[n, correlationStart + kernelSize/2:changeEnd + kernelSize/2][:correlation.shape[0]] = correlation

    '''
    randIndex = np.random.randint( nKernels )
    print np.max( np.abs( np.correlate( residual, dictionary[randIndex], 'same' ) - correlations[randIndex] ) )
    plt.plot( np.correlate( residual, dictionary[0], 'same' ) )#[zoomMe - 100:zoomMe+100] )
    plt.plot( correlations[0] )#[zoomMe - 100:zoomMe + 100] )
    plt.show()'''
            
    # Get the kernel index and sample offset and store them
    bestKernelAndOffset = np.unravel_index( np.argmax( np.abs( correlations ) ), correlations.shape )
    kernels[currentIteration] = bestKernelAndOffset[0]
    offsets[currentIteration] = bestKernelAndOffset[1]
    # Get the kernel scale
    scales[currentIteration] = correlations[kernels[currentIteration], offsets[currentIteration]]

    # Get the kernel that turned out to be the best
    kernel = dictionary[kernels[currentIteration]]*scales[currentIteration]
    # The actual offset is the correlation offset - (the kernel size/2)
    offsets[currentIteration] -= kernel.shape[0]/2
    # Construct the shifted kernel to subtract out of the residual
    #shiftedKernel = np.zeros( residual.shape[0] )
    #shiftedKernel[offsets[currentIteration]:offsets[currentIteration] + kernel.shape[0]] = kernel[:shiftedKernel.shape[0] - offsets[currentIteration]]

    '''if currentIteration % 1000 == 0:
      plt.subplot( 211 )
      plt.plot( residual )
      plt.plot( np.arange( offsets[currentIteration], offsets[currentIteration] + kernel.shape[0] ), kernel )'''
    # Subtract out the shfited kernel to get the new residual
    #residual -= shiftedKernel
    residual[offsets[currentIteration]:offsets[currentIteration] + kernel.shape[0]] -= kernel
    '''if currentIteration % 1000 == 0:
      plt.subplot( 212 )
      plt.plot( residual )
      plt.show()'''
    currentResidualMax = np.max( np.abs( residual ) )
    print "Iteration {}, time = {:.3f}, residual = {:.3f}, scale = {:.3f}".format( currentIteration, time.time() - lastTime, currentResidualMax, np.mean( kernel**2 ) )
    lastTime = time.time()
    # Next iteration...
    currentIteration += 1
    if currentIteration > maxIterations - 1:
      break

  # Trim return arrays
  scales = scales[:currentIteration]
  kernels = kernels[:currentIteration]
  offsets = offsets[:currentIteration]
  # Initialize return signal
  returnSignal = np.zeros( residual.shape[0] )
  # Sum in the kernels
  for n in np.arange( currentIteration ):
    kernel = dictionary[kernels[n]]
    returnSignal[offsets[n]:offsets[n] + kernel.shape[0]] += scales[n]*kernel
  # Trim the zero padding
  returnSignal = returnSignal[biggestKernelSize:-1]
  residual = residual[biggestKernelSize:-1]
  return returnSignal, residual, scales, kernels, offsets

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: %s audio.wav" % (sys.argv[0])
    sys.exit()
  audioData, fs = utility.getAudioData( sys.argv[1] )
  kernelDictionary = ERBFiltersToKernels( makeERBFilters( fs, 200, 100 ) )
  reconstructedSignal, residual, scales, kernels, offsets = matchingPursuit( kernelDictionary, audioData )
  plt.subplot(211)
  plt.plot( audioData )
  plt.subplot(212)
  plt.plot( reconstructedSignal )
  plt.plot( residual )
  plt.show()
  utility.writeWav( reconstructedSignal, fs, os.path.splitext( sys.argv[1] )[0] + "Reconstructed.wav" )