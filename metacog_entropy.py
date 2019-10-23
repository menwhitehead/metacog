from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

import numpy as np
from misc_functions import *

INPUT_BINARY_LIMIT = 20
OUTPUT_SIZE = 1
MINIBATCH_SIZE = 32
EPOCHS = 100
TRAINING_PERCENT = 0.8


def getSinePattern(n):
    result = np.sin(n)
    result = normalize(result, -1, 1)
    binary_input = convertToBinary(n, INPUT_BINARY_LIMIT)
    return binary_input, result
    
def getLogPattern(n):
    result = math.log(n, 2)
    result = normalize(result, 0, 10)  # don't allow logs over 10 for now
    binary_input = convertToBinary(n, INPUT_BINARY_LIMIT)
    return binary_input, result
    
def getSquarePattern(n):
    result = math.exp(n, 2)
    result = normalize(result, 0, 100)  # don't allow logs over 10 for now
    binary_input = convertToBinary(n, INPUT_BINARY_LIMIT)
    return binary_input, result

def getDataset(n, pattern_function):
    data = []
    labels = []
    for i in range(1, n):
        binary_input, one_hot_factors_vector = pattern_function(i)
        data.append(binary_input)
        labels.append(one_hot_factors_vector)
    return data, labels

def grabAModel():
    inputs = Input(shape=(INPUT_BINARY_LIMIT,))
    output_1 = Dense(64, activation='relu')(inputs)
    predictions = Dense(OUTPUT_SIZE, activation='relu')(output_1)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='mse',)
    return model


def getModelActivations(model, data):
    layer_output_functions = []
    for layer_num in range(1, len(model.layers)-1):
        layer_output_functions.append(K.function([model.layers[0].input],
                                      [model.layers[layer_num].output]))
    activations = []                  
    for func in layer_output_functions:                        
        layer_output = func(data)[0]
        activations.append(layer_output)
    return np.array(activations)
    


def distributionMaker(dist1, dist2):
    intervals = 100
    bs = []
    curr = -1
    for i in range(intervals):
        bs.append(curr)
        curr += 2.0 / intervals
    hist1 = np.histogram(dist1, bins=bs)[0] / float(len(bs))
    hist1[hist1 == 0] = 0.0001
    hist2 = np.histogram(dist2, bins=bs)[0] / float(len(bs))
    hist2[hist2 == 0] = 0.0001
    return hist1, hist2
    
def compareActs(acts1, acts2):
    distance_sum = 0.0
    rows, cols = acts1.shape
    for col in range(cols):
        col_act1 = acts1[:, col]
        col_act2 = acts2[:, col]
        dist1, dist2 = distributionMaker(col_act1, col_act2)
        distance_sum += entropy(dist1, dist2)
    return distance_sum / cols



if __name__ == "__main__":
    log_data, log_labels = getDataset(100, getLogPattern)
    log_data = np.array(log_data)
    log_labels = np.array(log_labels)
    
    sine_data, sine_labels = getDataset(100, getSinePattern)
    sine_data = np.array(sine_data)
    sine_labels = np.array(sine_labels)
    
    number_tests = 100
    diffs = [0.0, 0.0, 0.0]
    for i in range(number_tests):
        model1 = grabAModel()
        model2 = grabAModel()
        model3 = grabAModel()

        model1.fit(log_data, log_labels, validation_split=1.0-TRAINING_PERCENT, epochs=EPOCHS)
        model2.fit(log_data, log_labels, validation_split=1.0-TRAINING_PERCENT, epochs=EPOCHS)
        model3.fit(sine_data, sine_labels, validation_split=1.0-TRAINING_PERCENT, epochs=EPOCHS)

        acts1 = getModelActivations(model1, log_data)[0]
        acts2 = getModelActivations(model2, log_data)[0]
        acts3 = getModelActivations(model3, sine_data)[0]
        
        diff1v2 = compareActs(acts1, acts2)
        diff1v3 = compareActs(acts1, acts3)
        diff2v3 = compareActs(acts2, acts3)
        diffs[0] += diff1v2
        diffs[1] += diff1v3
        diffs[2] += diff2v3
    print diffs



# Given an existing sine function
# Learn the ANN mapping from arbitrary inputs to corresponding sine outputs

# And then given a new unknown function (cosine)
# Compare network activations? Find activation similarities/distances?
#  to relate the 
# previously trained sine function approximator 
# to the new learning of cosine.  But you already have a
# known, correct sine function that you can use instead.
# Then you have evidence that using your known sine function
# can get you a zero-error solution to the cosine approximation

# Metacognition by having one network look at the activations
# of another in order to "see" how the other is thinking
# and then map related problems somehow?  Or use known functions based
# on that analysis of the other's congition? Dunno.




