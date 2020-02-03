from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
import numpy as np

from misc_functions import *
from metacog_datasets import *
from metacog_buildingblock_functions import *
from metacog_globals import *


def grabAModel():
    inputs = Input(shape=(INPUT_SIZE,))
    output_1 = Dense(HIDDEN_SIZE, kernel_initializer=WEIGHT_INIT, activation='relu')(inputs)
    dropout_1 = Dropout(DROPOUT_RATE)(output_1)
    predictions = Dense(OUTPUT_SIZE, activation='relu')(dropout_1)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def getModelActivationFunctions(model):
    layer_output_functions = []
    for layer_num in range(1, len(model.layers)-1):
        layer_output_functions.append(K.function([model.layers[0].input],
                                      [model.layers[layer_num].output]))
    return layer_output_functions

def getModelActivations(layer_output_functions, data):
    activations = []
    for func in layer_output_functions:
        layer_output = func(data)[0]
        activations.append(layer_output)
    return np.array(activations)

