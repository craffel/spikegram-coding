import multiprocessing as mp
import numpy as np
from spikeGramTest import *
import time

import logging
logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

class correlator( mp.Process ):
  
  def __init__( self ):
    self.in_Q = mp.Queue()
    self.out_Q = mp.Queue()
    Super( correlator, self ).__init__( target=self.__consumer )
    Super( correlator, self ).start()

  def __consumer( self ):
    while True:
      try:
        i, residual, kernel = self.in_Q.get(True, 1)
        self.out_Q.put( (i, np.correlate( residual, kernel, 'same') ) )
      except:
        time.sleep(.1)

def go( dictionary, residual):

  # Number of cores!
  nCores = 4
  for i in np.arange( nCores ):
    correlator( target=__consumer, args=(in_Q, out_Q) ).start()

  currentResidualMax = 0
  threshold = -1
  
  correlations = np.zeros( (nKernels, residual.shape[0]) )
  
  # Until the max of the residual is smaller than our threshold value
  while currentResidualMax > threshold:
    
    for i in np.arange( nKernels ):
      in_Q.put( i )
    
    # Number of cores!
    nCores = 4
    for i in np.arange( nCores ):
      mp.Process( target=__consumer, args=(in_Q, out_Q) ).start()
    
    for j in np.arange( nKernels ):
      (i, v) = out_Q.get( True )
      correlations[i] = v


nKernels = 64
kernelDictionary = ERBFiltersToKernels( makeERBFilters( 44100, 64, 100 ) )
residual = np.random.rand( 427770 )

go( kernelDictionary, residual )