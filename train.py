
import wav
import generate_data

import os.path
import math

import keras

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation

TrainX = []
TrainY = []
TestX = []
TestY = []

categories = []
model_loaded = False



def find_discriminator(pred_index, predictions, actual_values):
    # find cutoff value with minimal errors

    minsum = 9999999999999      # minimal error difference
    mincut = 0                  # cutoff value with minimal error rate, basically weights of model
    
    for l in range(20,100):
        cut = l / 100
        sum_incorrect = 0       # counter for incorrect guesses

        for i in range(len(predictions)):
            
            predicted_value = 1

            if predictions[i][pred_index] < cut: 
                predicted_value = 0

            sum_incorrect += abs(predicted_value - actual_values[i][pred_index])

        if sum_incorrect <= minsum:
            minsum = sum_incorrect
            mincut = cut

    return len(predictions), minsum, mincut



def evaluate_model(model):
    # evaluate and print model's performance

    global TrainX, TrainY, TestX, TestY

    pred = model.predict(TestX)
    for i in range(len(pred)):
        print("Predictions:" + str(['{:f}'.format(f) for f in pred[i]]) + " " + str(['{:1.0f}'.format(f) for f in TestY[i]]) )
    
    sumwrong = 0
    sumpred = 0
    mincuts = []

    for i in range(len(pred[0])):

        npred, minsum, mincut = find_discriminator(i, pred, TestY)
        print(str(npred) + " " + str(minsum) + " " + str(mincut))
        
        sumwrong += minsum
        sumpred += npred
        mincuts.append(mincut)

    print("\ntotal trained: " + str(sumpred) + ", total wrong: " + str(sumwrong) + ", error rate: " + str(sumwrong / sumpred))
    return sumwrong / sumpred



def safe_model(model, accuracy):
    # save model if it performs better than the previously best model

    global model_loaded

    with open("prev_accuracy.txt", 'r+') as file:

        content = file.readlines()
        content = "".join([str(x.strip()) for x in content])

        if not content or (float(content) > accuracy) or model_loaded==False:
            
            model.save('model/piano_model.h5')
            print("model saved as model/piano_model.h5")

            file.seek(0)
            file.truncate()
            file.write(str(accuracy))

        elif float(content) < accuracy:
            print("accuracy worse than in the previous run. model not saved.")



def train_model(categories, numepochs, model):

    print('training')
    global TrainX, TrainY, TestX, TestY, model_loaded

    if not model_loaded:
        model.add(Dense(6000, input_dim=len(TrainX[0])))
        model.add(Activation('relu'))
        model.add(Dense(1000))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(len(categories)))
        model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adamax')
    model.fit(TrainX, TrainY, batch_size=256, nb_epoch=numepochs)
    return model



def begin_training(iterations):

    global TrainX, TrainY, TestX, TestY, categories, model_loaded

    generator = generate_data.DataGenerator()
    categories = generator.load_wavs()
    print(categories)

    for i in range(iterations):

        TrainX, TrainY, TestX, TestY = generator.create_data()
        epoch_num = 50

        # load previous model if one exists
        if os.path.isfile('model/piano_model.h5'):
            print("Load previous model")
            model = load_model('model/piano_model.h5')
            model_loaded = True
            model = train_model(categories, epoch_num, model)
        else:
            print("Create new model")
            model = Sequential()
            model = train_model(categories, epoch_num, model)

        accuracy = evaluate_model(model)
        safe_model(model, accuracy)
        del model  



if __name__ == '__main__':
    begin_training(5)
