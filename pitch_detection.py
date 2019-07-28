
import onset_detection
import wav

import os.path
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model



def extract_notes(pcm_data, onset_times, samplerate):
    # filter all notes from an audio file

    time = 0.1
    numfreq = 1024
    overlap = 0

    notes = []

    for onset in onset_times:

        pcm_slice = pcm_data[onset:onset+math.floor(time*samplerate)]
    
        fft1 = wav.compute_fft(pcm_slice, 0, time, samplerate, 32, 0)
        fft2 = wav.compute_fft(pcm_slice, 0, time, samplerate, 256, 0)
        fft  = wav.compute_fft(pcm_slice, 0, time, samplerate, numfreq, overlap)

        X = np.concatenate((fft1.flatten(), fft2.flatten(), fft.flatten())) # fft slice

        notes.append(X)

    return np.array(notes)



def load_wav(wav_file):
    # load audio file
    
    pcm_data, samplerate = wav.read_wav(wav_file, 400)
    sums, onset_times = onset_detection.detect_onset(pcm_data, samplerate, False, True)
    return pcm_data, onset_times, samplerate, sums



def evaluate_model(filename):
    # apply trained model on an audio file to predict pitches

    if os.path.isfile('model/piano_model.h5'):
        print("load model")
        model = load_model('model/piano_model.h5')
    else:
        print("no model available")
        return

    pcm_data, onset_times, samplerate, sums = load_wav(filename)
    notes = extract_notes(pcm_data, onset_times, samplerate)
    model.compile(loss='mean_squared_error', optimizer='adamax')

    trainingfiles = wav.get_training_filenames()

    for i in range(len(trainingfiles)):
        trainingfiles[i] = trainingfiles[i].split("/")[1].split(".")[0]

    pred = model.predict(notes)
    predicted_notes = []

    for i in range(len(pred)):

        best_guess = 0
        num = 0
        nums = []

        for e, f in enumerate(pred[i]):
            
            if best_guess < f:
                best_guess = f
                num = e

            if f > 0.85:
                nums.append(e)

        results = [trainingfiles[i] for i in nums]
        predicted_notes.append(results)
        print("Best guess:", "   ".join(results))

    plot_pitches(sums, samplerate, onset_times, predicted_notes)



def plot_pitches(sums, samplerate, onset_times, predicted_notes):
    # create a visual representation of the onset_times
    # debug function

    print("plot onset_times")
    sums = np.asarray(sums)
    time = np.arange(len(sums)) / samplerate

    # draw a plot of the compressed audio samples
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, sums)

    # draw a vertical line at each detected onset
    for e, i in enumerate(onset_times):
        xpos = i/samplerate
        for u, j in enumerate(predicted_notes[e]):
            plt.text(xpos, 1.3-((u+1)*0.1), j)
        plt.plot([xpos, xpos], [0, 1], 'k-', lw=1)

    plt.show()



if __name__ == '__main__':
    if (len(sys.argv) > 1):
        evaluate_model(sys.argv[1])
