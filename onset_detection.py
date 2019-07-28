import wav

import numpy as np
import matplotlib.pyplot as plt
import queue
import sys


def detect_onset(buffer, samplerate, plot=False, pitchdetection=False, silence=0.01):
    # detect onsents of notes in a wav file

    print("detecting onsets")

    ndat = np.shape(buffer)[0]
    sums = []
    window_values = []
    onset_times = []
    windowsize = 1000
    skip = 0
    maximum = 0
    wsum = 0

    for i in range(len(buffer)):
        # calculate sum of values in a window for each sample, and add them to a list

        newval = abs(compression(buffer[i]))
        window_values.append(newval)
        wsum += newval

        if len(window_values) > windowsize:
            wsum -= window_values.pop(0)

        normalized_wsum = wsum / windowsize
        maximum = max(maximum, normalized_wsum)
        sums.append(normalized_wsum + 1)


    for i in range(0, len(sums)):
        # find the onsets

        sums[i] /= maximum   # scale value of sample based on the max value

        if (skip > 0):
            skip -= 1
            continue
        if (sums[i] > silence):
            if (i > 5000) and (sums[i] / sums[i-5000] > 2):
                onset_times.append(i)
                skip = 10000

    print("onsets found:", onset_times)

    if plot:
        plot_onsets(sums, samplerate, onset_times)

    if pitchdetection:
        return(sums, onset_times)

    return(onset_times)


def compression(sample, rate = 2):
    # prepare data for onset detection by making large values larger and small values smaller

    MAX = 2**23     # maximal possible value, 24 bit signed = 2^23

    sign = 1

    if (sample < 0):
        sign = -1

    normalized = abs(sample / MAX)
    normalized = 1 - ((1 - normalized)**rate)

    return sign * normalized * MAX


def plot_onsets(sums, samplerate, onset_times):
    # create a visual representation of the onsets
    # debug function

    print("plot onsets")
    sums = np.asarray(sums)
    time = np.arange(len(sums)) / samplerate

    # draw a plot of the compressed audio samples
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, sums)

    # draw a vertical line at each detected onset
    for i in onset_times:
        xpos = i/samplerate
        plt.plot([xpos, xpos], [0, 1], 'k-', lw=1)

    plt.show()



if __name__ == '__main__':

    if (len(sys.argv) > 1):
        pcm_data, samplerate = wav.read_wav(sys.argv[1], 400)
        detect_onset(pcm_data, samplerate, True)
