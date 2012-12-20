# harmonicPercussiveEncoding
# Try coding the harmonic and percussive components separately
#
# Created by Colin Raffel on 11/17/12

import numpy as np
import utility
import spikeGramTest
import ERBFilters
import separateHarmonicPercussive
import sys
import os
import melNoiseTransform

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: %s audio.wav" % (sys.argv[0])
    sys.exit()
  audioData, fs = utility.getAudioData( sys.argv[1] )
  
  hop = 1024
  frameSize = 4096
  spectrogram = utility.getSpectrogram( audioData, hop=hop, frameSize=frameSize )
  seperator = separateHarmonicPercussive.HarmonicPercussiveSeparator( spectrogram )
  harmonicSignal = utility.getSignalFromSpectrogram( seperator.harmonicSpectrogram, hop, np.hanning( frameSize ) )
  percussiveSignal = utility.getSignalFromSpectrogram( seperator.percussiveSpectrogram, hop, np.hanning( frameSize ) )

  impulse = np.zeros( 10000 )
  impulse[0] = 1.0
  kernelDictionary = ERBFilters.ERBFiltersToKernels( impulse, ERBFilters.makeERBFilters( fs, 200, 100, 5000 ) )
  reconstructedHarmonicSignal, residual, scales, kernels, offsets = spikeGramTest.matchingPursuit( kernelDictionary, harmonicSignal, 16, 0, (2000.0*audioData.shape[0])/fs )
  
  melTransformer = melNoiseTransform.melTransformer( fs )
  N = np.min( [percussiveSignal.shape[0], residual.shape[0]] )
  reconstructedNoiseSignal = melTransformer.getSignalFromMelGram( melTransformer.getMelGram( percussiveSignal[:N] + residual[:N] ) )

  reconstructedNoiseSignal /= np.max( np.abs( reconstructedNoiseSignal ) )
  reconstructedHarmonicSignal /= np.max( np.abs( reconstructedHarmonicSignal ) )

  N = np.min( [reconstructedHarmonicSignal.shape[0], reconstructedNoiseSignal.shape[0]] )
  basename =os.path.splitext( sys.argv[1] )[0]
  utility.writeWav( reconstructedHarmonicSignal[:N] + reconstructedNoiseSignal[:N], fs, basename + "ReconstructedHarmPerc.wav" )
  utility.writeWav( harmonicSignal, fs, basename + "Harm.wav" )
  utility.writeWav( percussiveSignal, fs, basename + "Perc.wav" )
  utility.writeWav( reconstructedHarmonicSignal, fs, basename + "ReconstructedHarm.wav" )
  utility.writeWav( reconstructedNoiseSignal, fs, basename + "ReconstructedPerc+HarmResidual.wav" )
  utility.writeWav( residual, fs, basename + "HarmResidual.wav" )

  scalesKernelsAndOffsets = np.zeros( (scales.shape[0], 3) )
  scalesKernelsAndOffsets[:, 0] = scales
  scalesKernelsAndOffsets[:, 1] = kernels
  scalesKernelsAndOffsets[:, 2] = offsets
  np.save( basename + 'ReconstructedHarmPercScalesKernelsAndOffsets.npy', scalesKernelsAndOffsets )