# separateHarmonicPercussive.py
# Use median filtering to extract tonal and percussive components of an audio signal
#
# Created by Colin Raffel on 10/25/10

import numpy as np
import medianFilter
from time import time
    
class HarmonicPercussiveSeparator:
  def __init__( self, spectrogram, wienerPower = 3, medianFilterSize = 17 ):
    self.spectrogram = spectrogram
    self.wienerPower = wienerPower
    self.medianFilterSize = medianFilterSize
    self.separate()
  
  def separate( self ):
    magnitudeSpectrogram = np.abs( self.spectrogram )
    #harmonicFilter = self.medianFilterEachRow( magnitudeSpectrogram.T ).T
    harmonicFilter = np.array( magnitudeSpectrogram.T, order = 'C' )
    medianFilter.filterEachRow( harmonicFilter, self.medianFilterSize )
    harmonicFilter = harmonicFilter.T
    #percussiveFilter = self.medianFilterEachRow( magnitudeSpectrogram )
    percussiveFilter = magnitudeSpectrogram
    medianFilter.filterEachRow( percussiveFilter, self.medianFilterSize )

    harmonicFilter = harmonicFilter**self.wienerPower
    percussiveFilter = percussiveFilter**self.wienerPower
    harmonicMask = harmonicFilter/(harmonicFilter + percussiveFilter + np.finfo(float).eps)
    percussiveMask = percussiveFilter/(harmonicFilter + percussiveFilter + np.finfo(float).eps)

    self.harmonicSpectrogram = self.spectrogram*harmonicMask
    self.percussiveSpectrogram = self.spectrogram*percussiveMask
    
  def medianFilterEachRow( self, image ):
    outputImage = np.array( image )
    for n in np.arange( image.shape[0] ):
      row = outputImage[n]
      medianFilter.filter( row, self.medianFilterSize )
    return outputImage    
  
# Run function as script
if __name__ == "__main__":
  import sys
  import os
  import utility
  if len(sys.argv) < 2:
    print "Usage: %s filename.mp3|filename.wav" % sys.argv[0]
    sys.exit(-1)
  # Wav or mp3?
  basename, extension = os.path.splitext( sys.argv[1] )
  if extension == '.mp3':
    print "Getting mp3 data ..."
    audioData, fs = utility.getMp3Data( sys.argv[1] )
  elif extension == '.wav':
    print "Getting wav data ..."
    audioData, fs = utility.getWavData( sys.argv[1] )
  else:
    print "Not .wav or .mp3."
    sys.exit(-1)
  
  hop = 1024
  frameSize = 4096
  spectrogram = utility.getSpectrogram( audioData, hop=hop, frameSize=frameSize )
  print "Separating harmonic and percussive components ..."
  start = time()
  seperator = HarmonicPercussiveSeparator( spectrogram )
  end = time()
  print "Took ", end - start
  utility.plotSpectrogram( spectrogram )
  utility.plotSpectrogram( seperator.harmonicSpectrogram )
  utility.plotSpectrogram( seperator.percussiveSpectrogram )
  #utility.writeWav( utility.getSignalFromSpectrogram( seperator.harmonicSpectrogram, hop, np.hanning( frameSize ) ), fs, sys.argv[1] + "harm.wav", 1 )
  #utility.writeWav( utility.getSignalFromSpectrogram( seperator.percussiveSpectrogram, hop, np.hanning( frameSize ) ), fs, sys.argv[1] + "perc.wav", 1 )