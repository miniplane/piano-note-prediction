import train

import wave
import numpy as np
import math
import struct
import random
import os
import glob
from array import array
import matplotlib.pyplot as plt



def create_wav(data=None):
    # debug function for saving training and test pcm data as wav files

    file = "test.wav"
    framerate = train.samplerate

    if data == None:
        data, framerate = read_wav(files[0], 400)

    w = wave.open(file, mode="wb")

    w.setframerate(framerate)
    w.setnchannels(1) # mono
    w.setsampwidth(3)

    buf = bytes()

    for val in data:
        buf += struct.pack('i', val)[1:4]

    w.writeframes(buf)
    w.close()



def get_training_filenames():
    # get all .wav files from the notes dir

    trainingfiles = []

    path = os.path.abspath("notes")
    os.chdir(path)

    for file in glob.glob("*.wav"):
        trainingfiles.append("notes/"+file)

    os.chdir("..")

    return sorted(trainingfiles)



def read_wav(filename, sec):
    # parse wav and return frames and framerate

    print ( "reading " + filename )
    wvf = wave.open(filename)
    nframes = wvf.getnframes()
    framerate = wvf.getframerate()

    if(sec > 0):
        nframes = min(nframes, sec*framerate)

    pcm = wvf.readframes(nframes)       # list of frames
    data = array('i')

    for i in range(0,nframes*3,3):
        data.append(struct.unpack('<i', b'\x00'+ pcm[i:i+3])[0])    

    return data, framerate



def compute_spectrogram(wvd, startidx, sec, framerate, numfreq, overlap):
    # compute values for plotting a spectogram of a recording

    N = 2*numfreq
    incr = math.ceil( (1-overlap / 100.0)*N )
    spec = []
    times = []
    freqs = [ ((i+1)*(framerate/2.0) / numfreq) for i in range(numfreq) ]
    i = startidx

    while i+N < startidx + sec*framerate:

        if  i+N >= len(wvd):
            break

        data = np.asarray( wvd[i:i+N] )
        freqval = np.fft.rfft(data)
        ffts = []	
        
        for k in range(numfreq):
            ffts.append(np.abs( freqval[k] ))

        spec.append(ffts)
        times.append( i / framerate)
        i = i + incr
        
    return np.asarray(spec), np.asarray(times), np.asarray(freqs)



def compute_fft(wvd, startidx, sec, framerate, numfreq, overlap):
    # compute fast fourier transformation of a recording

    fft, t, f = compute_spectrogram( wvd, startidx, sec, framerate, numfreq, overlap )
    return fft



def plot_wav(x, samplerate, Sxx, t, f, time):
    # plot wave file to a diagram
    # debug function

    print("plot wav")

    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, x)

    plt.subplot(212)
    plt.pcolormesh(t, f, np.transpose( Sxx ))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()

    print(x)



if __name__ == '__main__':
    # prints a spectogram

    print("read in file")
    x, samplerate = read_wav("examples/summertime.wav",400)
    print("compute_spectrogram")
    Sxx, t, f = compute_spectrogram(x, 0, 400, samplerate, 2048, 0)

    print("test")
    x = np.asarray(x)
    time = np.arange(len(x)) / samplerate
    plot_wav(x, samplerate, Sxx, t, f, time)


