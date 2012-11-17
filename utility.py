# utility.py
# Various functions useful for song analysis
#
# Created by Colin Raffel on 10/7/10

import numpy as np
import wave
import csv
import os
import struct
import platform
if platform.system() is not 'Windows':
  import mad

def getWavData( wavFile ):
  # Get wav data
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
    if len( buffy ) != buffyLen:
      print "Warning: MP3 buffer size was different than expected.  Trying to recover."
      if len( buffy ) > buffyLen:
        audioData = np.append( audioData, np.zeros( audioData.shape[0]*(len( buffy )/buffyLen)*4 - audioData.shape[0] ) )
      buffyLen = len( buffy )
    # Store this buffer in the audioData
    audioData[dataPointer:(dataPointer+buffyLen)] = buffy
    # Incrememt data pointer
    dataPointer = dataPointer + buffyLen
    # Read next buffer
    buffy = mf.read();
  if dataPointer + buffyLen < audioData.shape[0]:
    audioData = audioData[:dataPointer + buffyLen]
  return audioData

def writeWav( audioData, fs, filename, normalize = 0 ):
  if normalize:
    audioData = audioData/np.max( np.abs( audioData ) )
  audioData = np.array( audioData*32767, dtype=np.int16 )
  audioDataBuffer = struct.pack( 'h'*audioData.shape[0], *tuple( audioData ) )
  file = wave.open(filename, 'wb')
  file.setparams( (1, 2, fs, audioData.shape[0], 'NONE', 'noncompressed') )
  file.writeframes( audioDataBuffer )
  file.close()
  #wavfile.write( filename, fs, audioData )

# Wrapper for all audio file types
def getAudioData( audioFile ):
  # Check that the file exists
  if not os.path.exists( audioFile ):
    print "%s doesn't exist." % audioFile
    return np.array([]), 0
  basename, extension = os.path.splitext( audioFile )
  if platform.system() is not 'Windows' and extension == '.mp3':
    return getMp3Data( audioFile )
  elif extension == '.wav':
    return getWavData( audioFile )
  else:
    print "%s is not a recognized audio file." % audioFile
    return np.array([]), 0

# Get files of a certain type in a directory, recursively
def getFiles( path, extension ):
  # If just a file was supplied
  if not os.path.isdir( path ):
    if os.path.splitext( path )[1] == extension:
      return [path]
    else:
      return []
  else:
    fileList = []
    for root, subdirectories, files in os.walk( path ):
      for file in files:
        # Only get files of the given type
        if os.path.splitext( file )[1] == extension:
          fileList.append( os.path.join( root, file ) )
    return fileList

# Split a signal into frames
def splitSignal( data, hop, frameSize ):
  nFrames = np.floor( (data.shape[0] - frameSize)/(1.0*hop) )
  # Pre-allocate matrix
  dataSplit = np.zeros( (nFrames, frameSize) )
  for n in np.arange(nFrames):
    dataSplit[n] = data[n*hop:n*hop+frameSize]
  return dataSplit

def unsplitSignal( dataSplit, hop, frameSize ):
  nFrames = dataSplit.shape[0]
  data = np.zeros( nFrames*(hop+1) )
  for n in np.arange( nFrames ):
    data[n*hop:n*hop+frameSize] += dataSplit[n]
  return data/np.max( np.abs( data ) )

# Get spectrogram of signal
def getSpectrogram( data, hop = 512, frameSize = 1024, window = np.ones( 1024 ) ):
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
  plt.imshow( np.log( np.abs( spectrogram ).T + 10e-100), origin='lower', aspect='auto' )
  plt.title( 'Log(Magnitude)' )
  plt.ylabel( 'Frequency bin' )
  plt.xlabel( 'Frame' )
  plt.subplot(212)
  plt.title( 'Phase' )
  plt.imshow( np.angle( spectrogram ).T, origin='lower', aspect='auto' )
  plt.ylabel( 'Frequency bin' )
  plt.xlabel( 'Frame' )
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

def getAnnotations( fileList, annotationDirectory ):
  annotationList = []
  for file in fileList:
    path, file = os.path.split( file )
    annotationPath = os.path.join( path, annotationDirectory )
    annotationList.append( os.path.join( annotationPath, file + '.txt' ) )
  return annotationList

def getOnsets( file ):
  return np.genfromtxt( file )

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

# Convert Hz to MIDI note (as a float!)
def hzToMIDI( frequency ):
  return 12.0*np.log2( frequency/440.0 ) + 69

# Convert MIDI note to bin number (as a float!) 
def midiToBin( midiNote, N, fs ):
  return hzToBin( midiToHz( midiNote ), N, fs )

# Convert bin in FFT to Hz
def binToHz( bin, N, fs ):
  return fs*bin/(N*1.0)

# Convert hz to closest FFT bin (as a float!)
def hzToBin( frequency, N, fs ):
  return frequency*N/(fs*1.0)

# Return bins in an FFT which are close to a Hz value
def hzToBins( hz, N, fs, tolerance = 0.02 ):
  # Range near bins in tolerance range
  binRange = np.arange( (1.0 - tolerance)*hz*N/fs, (1.0 + tolerance)*hz*N/fs )
  # Convert arange to integer indices
  bins = np.array( np.round( binRange ), dtype = np.int )
  return bins

# Return the next greatest power of 2 for any integer
def nextPowerOf2( value ):
  return np.int( 2**np.ceil( np.log2( value ) ) )

# Return MIDI note name (as a string) for a MIDI note number
def midiNoteToString( note ):
  notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B' ]
  return notes[ int( note ) % 12 ] + str( int( note )/12 - 5 )