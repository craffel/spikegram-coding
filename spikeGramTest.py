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
import ERBFilters
import fastAbsArgMax

# Encodes an input signal "x" with elements from "dictionary
def matchingPursuit( dictionary, x, amplitudeThreshold=.01, scaleThreshold=0, maxIterations=50000 ):
  
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
  # Where we'll be storing the kernels, scales, and offsets
  scales = np.zeros( maxIterations )
  kernels = np.zeros( maxIterations )
  offsets = np.zeros( maxIterations )
  
  # For keeping track of how long we've run
  lastTime = time.time()
  # For checking whether we've converged
  currentResidualMax = np.inf

  amplitudeThreshold = amplitudeThreshold*np.max( np.abs( x ) )
  
  # Until the max of the residual is smaller than our amplitudeThreshold value
  while currentResidualMax > amplitudeThreshold:
    
    # On first iteration, do the whole correlation
    if currentIteration == 0:
      for n in xrange( nKernels ):
        if currentIteration == 0:
          correlations[n] = scipy.signal.fftconvolve( residual, dictionary[n][::-1], 'same' )
    # On subsequent iterations, only re-correlate where it has changed
    else:
      # Where does the change in the residual start?
      changeStart = offsets[currentIteration - 1]
      # Look up the previous kernel subtracted from the residual - its size is where the change in the residual ends
      changeEnd = changeStart + dictionary[kernels[currentIteration - 1]].shape[0]
      for n in xrange( nKernels ):
        # What's the size of current kernel to be correlated?
        kernelSize = dictionary[n].shape[0]
        # Where should we start correlating this kernel for no edge effects?
        correlationStart = changeStart - 2*kernelSize
        # This should never happen, but just in case
        if correlationStart < 0:
          correlationStart = 0
        # Correlate with no edge effects
        #correlation = np.correlate( residual[correlationStart:changeEnd + kernelSize - 1], dictionary[n], 'valid' )
        correlation = scipy.signal.fftconvolve( residual[correlationStart:changeEnd + kernelSize - 1], dictionary[n][::-1], 'valid' )
        # [:correlation.shape[0]] is an indexing hack to avoid shape mismatches
        correlations[n, correlationStart + kernelSize/2:changeEnd + kernelSize/2][:correlation.shape[0]] = correlation
            
    # Get the kernel index and sample offset and store them
    bestKernelAndOffset = fastAbsArgMax.fastAbsArgMax( correlations )#np.unravel_index( np.argmax( np.abs( correlations ) ), correlations.shape )
    kernels[currentIteration] = bestKernelAndOffset[0]
    offsets[currentIteration] = bestKernelAndOffset[1]
        
    # Get the kernel scale
    scales[currentIteration] = correlations[kernels[currentIteration], offsets[currentIteration]]
    # Get the kernel that turned out to be the best
    kernel = dictionary[kernels[currentIteration]]*scales[currentIteration]
    # The actual offset is the correlation offset - (the kernel size/2)
    offsets[currentIteration] -= kernel.shape[0]/2
    # Subtract out the shfited kernel to get the new residual
    residual[offsets[currentIteration]:offsets[currentIteration] + kernel.shape[0]] -= kernel

    currentResidualMax = np.max( np.abs( kernel ) )
    print "Iteration {}, time = {:.3f}, scale = {:.3f}".format( currentIteration, time.time() - lastTime, currentResidualMax/amplitudeThreshold )
    lastTime = time.time()
    if np.abs(scales[currentIteration]) < scaleThreshold:
      break
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
  # Create impulse
  impulse = np.zeros( 10000 )
  impulse[0] = 1.0
  kernelDictionary = ERBFilters.ERBFiltersToKernels( impulse, ERBFilters.makeERBFilters( fs, 200, 100 ) )
  reconstructedSignal, residual, scales, kernels, offsets = matchingPursuit( kernelDictionary, audioData )
  plt.subplot(211)
  plt.plot( audioData )
  plt.subplot(212)
  plt.plot( reconstructedSignal )
  plt.plot( residual )
  plt.show()
  utility.writeWav( reconstructedSignal, fs, os.path.splitext( sys.argv[1] )[0] + "Reconstructed.wav" )