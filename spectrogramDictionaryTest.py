# spectrogramDictionaryTest.py
# Some tests on how dictionaries work on spectrograms
#
# Created by Colin Raffel on 10/8/12

import numpy as np
import matplotlib.pyplot as plt
import utility
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import sys
import os
import cqt

def getDictionary( spectrogram ):
  # Use scikits.learn's patching function to get spectral patches (arbitrarily sized!)
  patchSize = ( 8, 36 )
  data = extract_patches_2d( spectrogram, patchSize )
  # Code from here is from sk learn example
  data = data.reshape(data.shape[0], -1)
  data -= np.mean(data, axis=0)
  data /= np.std(data, axis=0)
  nAtoms = 64
  dictionary = MiniBatchDictionaryLearning(n_atoms=nAtoms, alpha=1, n_iter=500)
  components = dictionary.fit( data ).components_
  for i, comp in enumerate(components[:nAtoms]):
    plt.subplot(np.sqrt(nAtoms), np.sqrt(nAtoms), i + 1)
    plt.imshow(comp.reshape(patchSize).T, interpolation='nearest', cmap=plt.cm.gray_r, origin='lower', aspect='auto')
    plt.xticks(())
    plt.yticks(())
  plt.show()
  return dictionary, components

def reconstructFromDictionary( spectrogram, dictionary, components ):
  # Use scikits.learn's patching function to get spectral patches (arbitrarily sized!)
  patchSize = ( 8, 36 )
  data = extract_patches_2d( spectrogram, patchSize )
  # Code from here is from sk learn example
  data = data.reshape(data.shape[0], -1)
  data -= np.mean(data, axis=0)
  data /= np.std(data, axis=0)
  dictionary.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=2)
  code = dictionary.transform(data)
  patches = np.dot(code, components)
  patches = patches.reshape( len(data), *patchSize )
  return reconstruct_from_patches_2d( patches, spectrogram.shape )


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: %s goodAudio.wav badAudio.wav" % (sys.argv[0])
    sys.exit()

# Get a dictionary from the good audio data
goodAudio, fs = utility.getAudioData( sys.argv[1] )
# The whole signal is way too long!
goodAudio = goodAudio[:44100]
# Get spectrogram
#GoodAudio = utility.getSpectrogram( goodAudio, window = np.sqrt( np.hanning( 1024 ) ) )
constantQTransformer = cqt.ConstantQTransformer( fs )
GoodAudio = constantQTransformer.getSTCQT( goodAudio )
# Do thresholding
#threshold = .01*np.max(GoodAudio) + .99*np.min(GoodAudio)
#GoodAudio[GoodAudio < threshold] = 0
# Get the dictionary trained on this audio
dictionary, components = getDictionary( np.abs( GoodAudio )   )

# Reconstruct the bad audio using the dictionary
if len(sys.argv) > 2:
  badAudio, fs = utility.getAudioData( sys.argv[2] )
  # Shorten here too.
  badAudio = badAudio[:44100]
  #BadAudio = utility.getSpectrogram( badAudio )
  BadAudio = constantQTransformer.getSTCQT( badAudio )
  ReconstructedAudio = reconstructFromDictionary( np.abs( BadAudio ), dictionary, components )
  # Simulate phase
  simulatedPhase = np.angle( BadAudio )
else:
  ReconstructedAudio = reconstructFromDictionary( GoodAudio, dictionary, components )
  # Simulate phase
  simulatedPhase = np.angle( GoodAudio )

ReconstructedAudio = ReconstructedAudio*( np.cos( simulatedPhase ) + 1j*np.sin( simulatedPhase ) )

plt.subplot( 2, 1, 1 )
plt.imshow(np.abs( GoodAudio.T ), interpolation='nearest', origin='lower', cmap=plt.cm.gray_r, aspect='auto' )
plt.subplot( 2, 1, 2 )
plt.imshow(np.abs( ReconstructedAudio.T ), interpolation='nearest', origin='lower', cmap=plt.cm.gray_r, aspect='auto' )
plt.show()

'''reconstructedAudio = utility.getSignalFromSpectrogram( ReconstructedAudio, window = np.sqrt( np.hanning( 1024 ) ) )
# Write out wav
utility.writeWav( reconstructedAudio, fs, os.path.splitext( sys.argv[-1] )[0] + "Reconstructed.wav" )'''