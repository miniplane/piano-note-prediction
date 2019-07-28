
import wav
import train
import onset_detection

import numpy as np
import random
import math


class DataGenerator:

    numfreq = 1024
    overlap = 0
    samplerate = 44100
    time = 0.1              # length of the note starting from onset, used for generation
                            # increase for better accuracy but slower training
    minpar = 1
    maxpar = 2
    source_pcms = []        # collected pcm data of source files
    onsetsall = []          # ith index is array with time of onsets in file i
    TrainX = []
    TrainY = []
    TestX = []
    TestY = []

    def __init__(self):
        self.source_pcms = []        # collected pcm data of source files
        self.onsetsall = []          # ith index is array with time of onsets in file i
        self.TrainX = []
        self.TrainY = []
        self.TestX = []
        self.TestY = []
        random.seed()

    def create_adddata(self, onsets, train):
        # create a single note or chord

        ndat = math.floor(self.time*self.samplerate)    # number of frames in that time
        pcm_slice = [0]*ndat                            # create "silent" wav sample with length time
        Y = [0]*len(onsets)                     

        for pitch in range(len(onsets)):
            # generate a note or notes

            if onsets[pitch]>=0:

                for i in range(ndat):
                    try:        
                        pcm_slice[i]+=self.source_pcms[pitch][onsets[pitch]+i]
                    except: return

                Y[pitch] = 1                    # indicates a note on this pitch

        for i in range(ndat):
            # normalize samples by dividing them through the number of occuring notes
            pcm_slice[i] = int(pcm_slice[i] / (len(onsets)-onsets.count(-1)))

        # compute fft from pcm_slice
        fft1 = wav.compute_fft(pcm_slice, 0, self.time, self.samplerate, 32, 0)
        fft2 = wav.compute_fft(pcm_slice, 0, self.time, self.samplerate, 256, 0)
        fft  = wav.compute_fft(pcm_slice, 0, self.time, self.samplerate, self.numfreq, self.overlap)
        X = np.concatenate((fft1.flatten(), fft2.flatten(), fft.flatten()))

        if train:
            self.TrainX.append(X)
            self.TrainY.append(Y)
        else:
            self.TestX.append(X)
            self.TestY.append(Y)



    def create_samples(self, onsetlists, train, numblends):
        # create training or test samples from a subet of samples

        nlists = np.shape(onsetlists)[0]

        for k in range(nlists):                             # add all pure wavs
            for i in range(len(onsetlists[k])):
                onsets = [-1] * nlists
                onsets[k] = onsetlists[k][i]
                self.create_adddata(onsets, train)

        if nlists == 1: return
        
        for i in range(numblends):                          # number of blended samples to be created
            
            if i%1000 == 0:
                print(i)

            onsets = [-1]*nlists                           
            uplim = min((nlists+1),(self.maxpar+1))
            num = random.choice([i for i in range(self.minpar,uplim)])  # random number of wavs to blend over each other
            wavs = random.sample(range(nlists), num)                    # random list of wavs to blend over
            
            for k in wavs:                                              # loop over them
                onsets[k] = random.sample(onsetlists[k], 1)[0]          # pick one sample from the wav
            
            self.create_adddata(onsets, train)                          # add the blended samples



    def create_data(self):
        # create the training and test data

        self.TrainX = []
        self.TrainY = []
        self.TestX = []
        self.TestY = []

        onsetstrain = []
        onsetstest  = []

        ndat = np.shape(self.source_pcms)[0]

        for k in range(ndat):
            length = len(self.onsetsall[k])             # num of notes per file
            numtrain = math.floor(length*0.75)          # use 75% of notes

            onsets = self.onsetsall[k]
            random.shuffle(onsets) 
            onsetstrain.append(onsets[:numtrain])       
            onsetstest.append(onsets[numtrain:]) 

        self.create_samples(onsetstrain, True, 5000)         # create the training data, including blends
        self.create_samples(onsetstest, False, 1000)         # create test data
        
        self.TrainX = np.array(self.TrainX)
        self.TrainY = np.array(self.TrainY)
        self.TestX = np.array(self.TestX)
        self.TestY = np.array(self.TestY)

        return self.TrainX, self.TrainY, self.TestX, self.TestY



    def load_wavs(self):
        # load the raw training data files
        
        trainingfiles = wav.get_training_filenames()
        print (trainingfiles)

        for pitch in trainingfiles:
            pcm_data, self.samplerate = wav.read_wav(pitch, 400)
            onsets = onset_detection.detect_onset(pcm_data, self.samplerate) # onsets in file
            self.source_pcms.append(pcm_data)
            self.onsetsall.append(onsets)

        return [i for i in range(0, len(trainingfiles))]

