# utility.py
# Various functions useful for song analysis
#
# Created by Colin Raffel on 10/7/10

import numpy as np
import wave
import csv
import os
import struct
import mad
import scipy.io.wavfile as wavfile

def getWavData( wavFile ):
  '''# Get wav data
  wav = wave.open (wavFile, "r")
  (nChannels, sampleWidth, frameRate, nFrames, compressionType, compressionName) = wav.getparams()
  frames = wav.readframes( nFrames*nChannels )
  out = struct.unpack_from("%dh" % nFrames*nChannels, frames )
  
  # Convert 2 channles to numpy arrays
  if nChannels == 2:
    left = np.array( list( [out[i] for i in range( 0, len( out ), 2 )] ) )
    right = np.array( list( [out[i] for i in range( 1, len( out ), 2 )] ) )
  else:
    left = np.array( out )
    right = left
  audioData = (left + right)/2.0
  # Normalize
  audioData = audioData/np.max(np.abs(audioData))
  return audioData, frameRate
  '''
  # Get wav data
  fs, audioData = wavfile.read(wavFile)
  # Convert to mono
  if (len(audioData.shape) > 1) and (audioData.shape[1] > 1):
    audioData = np.mean( audioData, axis=1 )
  # Normalize
  audioData = (32767.0*audioData)/np.max(np.abs(audioData))
  return audioData, fs

def getMp3Data( mp3File ):
  # Prepare mp3 file object
  mf = mad.MadFile(mp3File)
  # Get PCM data
  audioData = mp3ToPCM( mf )
  # Convert to mono
  audioData = (audioData[::2] + audioData[1::2])/2.0
  # Store fs
  fs = mf.samplerate()
  # Normalize
  audioData = audioData/np.max(np.abs(audioData))
  return audioData, fs

def mp3ToPCM( mf ):
  # Get sample rate
  fs = mf.samplerate()
  # Initialize audioData array.  total_time is in milliseconds and rounded (eg 49000)
  # and there are two channels, so to get samples, add 1 second and multiply by 2
  audioData = np.zeros((mf.total_time()/1000)*2*fs)
  # Where in audioData are we?
  dataPointer = 0;
  # Read in the first buffer to get the length
  buffy = mf.read();
  # Length in per-channel samples
  buffyLen = len(buffy)/2
  # Read through file
  while 1:
    # Reached the end
    if buffy is None:
      break
    if dataPointer + buffyLen > audioData.shape[0]:
      break
    # Convenient, fast function!  Reads in 16 bit values from the buffer
    buffy = np.frombuffer(buffy, 'h')
    # Store this buffer in the audioData
    audioData[dataPointer:(dataPointer+buffyLen)] = buffy
    # Incrememt data pointer
    dataPointer = dataPointer + buffyLen
    # Read next buffer
    buffy = mf.read();
  return audioData

def writeWav( audioData, fs, filename, normalize = 1 ):
  if normalize:
    audioData = audioData/np.max( np.abs( audioData ) )
  audioData = np.array( audioData*32767, dtype=np.int16 )
  wavfile.write( filename, fs, audioData )

# Wrapper for all audio file types
def getAudioData( audioFile ):
  # Check that the file exists
  if not os.path.exists( audioFile ):
    print "%s doesn't exist." % audioFile
    return np.array([]), 0
  basename, extension = os.path.splitext( audioFile )
  if extension == '.mp3':
    return getMp3Data( audioFile )
  elif extension == '.wav':
    return getWavData( audioFile )
  else:
    print "%s is not a .wav or .mp3." % audioFile
    return np.array([]), 0

# Get files of a certain type in a directory, recursively
def getFiles( path, extension ):
  fileList = []
  for root, subdirectories, files in os.walk( path ):
    for file in files:
      # Only get files of the given type
      if os.path.splitext(file)[1] == extension:
        fileList.append(os.path.join(root, file))
  return fileList

# Split a signal into frames
def splitSignal( data, hop, frameSize ):
  nFrames = np.floor( (data.shape[0] - frameSize)/(1.0*hop) )
  # Pre-allocate matrix
  dataSplit = np.zeros( (nFrames, frameSize) )
  for n in np.arange(nFrames):
    dataSplit[n] = data[n*hop:n*hop+frameSize]
  return dataSplit

# Get spectrogram of signal
def getSpectrogram( data, **kwargs ):
  hop = kwargs.get('hop', 512)
  frameSize = kwargs.get('frameSize', 1024)
  window = kwargs.get('window', np.hanning(frameSize))
  # Framify the signal
  dataSplit = splitSignal( data, hop, frameSize )
  # Create spectrogram array
  spectrogram = np.zeros( (dataSplit.shape[0], dataSplit.shape[1]/2 + 1), dtype = np.complex )
  # Get spectra
  for n in np.arange( dataSplit.shape[0] ):
    spectrogram[n] = np.fft.rfft( window*dataSplit[n] )
  return spectrogram

# Plot the magnitude and phase of a spectrogram
def plotSpectrogram( spectrogram ):
  import matplotlib.pyplot as plt
  plt.subplot(211)
  plt.imshow( 20*np.log10( np.abs( spectrogram ).T + np.max(np.abs(spectrogram))*.0001 ), origin='lower', aspect='auto', interpolation='nearest', cmap=plt.cm.gray_r )
  plt.title( 'Log(Magnitude)' )
  plt.ylabel( 'Frequency bin' )
  plt.xlabel( 'Frame' )
  plt.colorbar()
  plt.subplot(212)
  plt.title( 'Phase' )
  plt.imshow( np.unwrap( np.angle( spectrogram ).T, axis=0 ), origin='lower', aspect='auto', interpolation='nearest', cmap=plt.cm.gray_r )
  plt.ylabel( 'Frequency bin' )
  plt.xlabel( 'Frame' )
  plt.colorbar()
  plt.show()

def getSignalFromSpectrogram( spectrogram, hop = 512, window = np.ones( 1024 ) ):
  # Get number of frames in the spectrogram
  nFrames = spectrogram.shape[0]
  # Size of the frame is the spectrum size, minus 1 (DC bin), times two (symmetric spectrum)
  frameSize = (spectrogram.shape[1] - 1)*2
  # Allocate output signal
  outputSignal = np.zeros(nFrames*hop + frameSize)
  for n in np.arange(nFrames):
    # Take IFFT of each spectrum and sum into output
    outputSignal[n*hop:n*hop + frameSize] = outputSignal[n*hop:n*hop + frameSize] + window*np.real(np.fft.irfft(spectrogram[n]))
  return outputSignal

# Get subdirectories for a given directory
def getSubdirectories( path ):
  dirList = []
  for root, subdirectories, files in os.walk( path ):
    for dir in subdirectories:
      dirList.append( os.path.join( root, dir ) )
  return dirList

# Convert midi note to Hz
def midiToHz( midiNote ):
  return 440.0*(2.0**((midiNote - 69)/12.0))

# Convert bin in FFT to Hz
def binToHz( bin, N, fs ):
  return fs*bin/(N*1.0)

# Return bins in an FFT which are close to a Hz value
def hzToBins( hz, N, fs, tolerance = 0.02 ):
  # Range near bins in tolerance range
  binRange = np.arange( (1.0 - tolerance)*hz*N/fs, (1.0 + tolerance)*hz*N/fs )
  # Convert arange to integer indices
  bins = np.array( np.round( binRange ), dtype = np.int )
  return bins