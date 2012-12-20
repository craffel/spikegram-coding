# plotSpikeGram.py
# Plot a spike gram (what did you expect?)
#
# Created by Colin Raffel on 12/14/12

import numpy as np
import matplotlib.pyplot as plt
import sys

def plotSpikeGram( scalesKernelsAndOffsets, nKernels=200, markerSize = .0001 ):
  for scale, kernel, offset in scalesKernelsAndOffsets:
    # Put a dot at each spike location.  Kernels on y axis.  Dot size corresponds to scale
    plt.plot( offset, nKernels-kernel, 'k.', ms=markerSize*np.abs( scale ) )
  plt.title( "Spikegram" )
  plt.xlabel( "Time (samples)" )
  plt.ylabel( "Kernel" )
  plt.axis( [0.0, np.max(scalesKernelsAndOffsets[:, 2]), 0.0, nKernels-1] )
  plt.show()

def plotSpikeDensity( scalesKernelsAndOffsets, window=256 ):
  offsets = scalesKernelsAndOffsets[:, 2]
  nWindows = int(np.floor( np.max(offsets)/(1.0*window) ))
  spikeCounts = np.zeros( nWindows )
  for n in xrange( nWindows ):
    spikeCounts[n] = np.sum( np.logical_and(offsets > n*window, offsets < (n+1)*window) )
  plt.bar( np.arange( nWindows ), spikeCounts, 1.0 )
  plt.show()