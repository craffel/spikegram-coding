import multiprocessing as mp
import numpy as np
import time
import sys

import logging
logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

def go( dictionary, residual):

  currentResidualMax = 0
  threshold = -1
  
  correlations = np.zeros( (nKernels, residual.shape[0]) )

  # Create queues to hold input indices and output index/correlation pairs
  in_Q = mp.Queue( nKernels )
  out_Q = mp.Queue( nKernels )

  # Parallelize cross-correlation computation...
  def __consumer():
    while True:
      # Until you get an empty queue exception
      try:
        i, residual, kernel = in_Q.get()
        print mp.current_process().name + " about to do " + str(i)
        out_Q.put( (i, np.correlate( residual, kernel, 'same') ) )
        print mp.current_process().name + " did " + str(i)      
      except:
        print "__consumer " + str(sys.exc_info()[0])
        time.sleep(.1)

  
  nCores = 8
  for i in np.arange( nCores ):
    mp.Process( target=__consumer ).start()

  # Until the max of the residual is smaller than our threshold value
  while currentResidualMax > threshold:
    
    for i in np.arange( nKernels ):
      out_Q.put( (i, residual) )
      out_Q.get()
      in_Q.put( (i, residual, dictionary[i]) )
    
    nCorrelationsCalculated = 0
        
    while nCorrelationsCalculated < nKernels:
      try:
        (i, v) = out_Q.get()
        correlations[i] = v
        nCorrelationsCalculated += 1
        print "Got correlation " + str( nCorrelationsCalculated )
      except:
        print "filler " + str(sys.exc_info()[0])
        time.sleep(.1)
    
    print correlations
    residual = np.random.rand( 427770 )
    

nKernels = 64
kernelDictionary = {}
for n in np.arange( nKernels ):
  kernelDictionary[n] = np.random.rand( np.random.randint( 300, 3000 ) )
residual = np.random.rand( 427770 )

go( kernelDictionary, residual )