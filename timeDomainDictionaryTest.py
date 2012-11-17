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
import optparse

def getDictionary( audioData ):
  # Split the signal into snippets
  data = utility.splitSignal( audioData, 64, 1024 )
  #data = data*(np.hanning( 1024 ) + 1e-100)
  # Code from here is from sk learn example
  data = data.reshape(data.shape[0], -1)
  data -= np.mean(data, axis=0)
  data /= np.max(np.abs( data ), axis=0)
  nAtoms = 400
  dictionary = MiniBatchDictionaryLearning(n_atoms=nAtoms, alpha=1, n_iter=500)
  components = dictionary.fit( data ).components_
  for i, comp in enumerate(components[:nAtoms]):
    plt.subplot(np.sqrt(nAtoms), np.sqrt(nAtoms), i + 1)
    plt.plot( comp )
    plt.xticks(())
    plt.yticks(())
  plt.figure()
  return dictionary, components

def reconstructFromDictionary( audioData, dictionary, components ):
  # Split the signal into snippets
  data = utility.splitSignal( audioData, 64, 1024 )
  # Code from here is from sk learn example
  data = data.reshape(data.shape[0], -1)
  data -= np.mean(data, axis=0)
  data /= np.max(np.abs( data ), axis=0)
  dictionary.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=2)
  code = dictionary.transform( data )
  for n in np.arange( code.shape[1] ):
    plt.plot( code[:,n]/np.max( np.abs( code[:,n] ) ) + 2*n)
  plt.figure()
  patches = np.dot( code, components )
  #patches = patches*(np.hanning( 1024 ) + 1e-100)
  return utility.unsplitSignal( patches, 64, 1024 )

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: %s goodAudio.wav -r badAudio.wav -o outputDirectory" % (sys.argv[0])
    sys.exit()

parser = optparse.OptionParser()
parser.add_option("-r", "--reconstruct", action="store", type="string", dest="reconstructFilename", help="The file to reconstruct" )
parser.add_option("-o", "--output", action="store", type="string", dest="outputDirectory", help="Output directory for dictionary atoms")
(options, args) = parser.parse_args( sys.argv )

# Get a dictionary from the good audio data
goodAudio, fs = utility.getAudioData( sys.argv[1] )
# Get the dictionary trained on this audio
dictionary, components = getDictionary( goodAudio )

# Write out the dictionary?
if options.outputDirectory is not None:
  outputDirectory = options.outputDirectory
  if not os.path.exists( outputDirectory ):
    os.makedirs( outputDirectory )
  for n, component in enumerate( components ):
    utility.writeWav( component/np.max( np.abs(component)), fs, os.path.join( outputDirectory, str( n ) + ".wav" ) )

# Reconstruct the bad audio using the dictionary
if options.reconstructFilename is not None:
  badAudio, fs = utility.getAudioData( sys.argv[2] )
else:
  badAudio = goodAudio

reconstructedAudio = reconstructFromDictionary( badAudio, dictionary, components )

ax1 = plt.subplot( 3, 1, 1 )
plt.plot( badAudio )
N = badAudio.shape[0] if badAudio.shape[0] < reconstructedAudio.shape[0] else reconstructedAudio.shape[0]
plt.subplot( 3, 1, 2, sharex=ax1, sharey=ax1 )
plt.plot( reconstructedAudio[:N] )
plt.subplot( 3, 1, 3, sharex=ax1, sharey=ax1 )
plt.plot( badAudio[:N] - reconstructedAudio[:N] )
plt.ylim( (-1, 1) )
plt.figure()

ax1 = plt.subplot( 3, 1, 1 )
ax1.imshow( np.log( np.abs( utility.getSpectrogram( badAudio ).T ) + 10e-100), origin='lower', aspect='auto' )
plt.subplot( 3, 1, 2, sharex=ax1, sharey=ax1 )
plt.imshow( np.log( np.abs( utility.getSpectrogram( reconstructedAudio[:N] ).T ) + 10e-100), origin='lower', aspect='auto' )
plt.subplot( 3, 1, 3, sharex=ax1, sharey=ax1 )
plt.imshow( np.log( np.abs( utility.getSpectrogram( badAudio[:N] - reconstructedAudio[:N] ).T ) + 10e-100), origin='lower', aspect='auto' )
plt.show()

# Write out wav
utility.writeWav( reconstructedAudio, fs, os.path.splitext( sys.argv[-1] )[0] + "Reconstructed.wav" )