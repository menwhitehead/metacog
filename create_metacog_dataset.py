from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
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

def distributionMaker(values, intervals=META_HIST_SIZE):
    # print "VALUES:", values
    norm_values = normalize(values)
    # print "NORM VALUES:", norm_values
    hist = np.array(np.histogram(norm_values, bins=META_HIST_SIZE)[0], dtype="float")
    hist[hist == 0] = META_HIST_ZERO_PROB
    norm_hist = normalize(hist, 0, 1)
    return norm_hist

def getColumnDistributions(acts):
    result = []
    rows, cols = acts.shape
    for col in range(cols):
        col_act = acts[:, col]
        #result = np.concatenate((result, distributionMaker(col_act)))
        # print "COL ACT:", col_act
        result.append(distributionMaker(col_act))
        # print "DIST:", result[-1]
        # print sum(result[-1])
    return np.concatenate(result)

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


def originalDatasetCreation():
        log_data, log_labels = getFullDataset(np.arange(1, 100), np.log)
        sqrt_data, sqrt_labels = getFullDataset(np.arange(1, 100), np.sqrt)
        LOG_LABEL = np.array([1, 0])
        SQRT_LABEL = np.array([0, 1])
        datasets = [(log_data, log_labels, LOG_LABEL), (sqrt_data, sqrt_labels, SQRT_LABEL)]

        meta_dataset = []
        meta_dataset_labels = []
        current_pattern = 0
        for i in range(META_NUMBER_MODELS):
            print "Working on model #", i
            model = grabAModel()
            print "Getting activation functions..."
            model_activation_functions = getModelActivationFunctions(model)
            print "Got 'em"
            ds = datasets[i%2]

            # Pre-train to avoid random beginnings
            model.fit(ds[0], ds[1], epochs=PRETRAINING_EPOCHS)

            for epoch in range(EPOCHS):
                # print "\tEPOCH #", epoch
                model.fit(ds[0], ds[1], epochs=1)

                acts = getModelActivations(model_activation_functions, ds[0])[0]
                dists = getColumnDistributions(acts)
                input_vector = np.concatenate((dists, convertToOneHot(PRETRAINING_EPOCHS+epoch, INPUT_ONE_HOT_LIMIT)))

                meta_dataset.append(input_vector)
                meta_dataset_labels.append(ds[2])

        meta_dataset = np.array(meta_dataset)
        meta_dataset_labels = np.array(meta_dataset_labels)
        print meta_dataset.shape, meta_dataset_labels.shape
        createH5Dataset(META_DATASET_FILENAME, meta_dataset, meta_dataset_labels)


def pathDatasetCreation():
        log_data, log_labels = getFullDataset(np.arange(1, 100), np.log)
        sqrt_data, sqrt_labels = getFullDataset(np.arange(1, 100), np.sqrt)
        LOG_LABEL = np.array([1, 0])
        SQRT_LABEL = np.array([0, 1])
        datasets = [(log_data, log_labels, LOG_LABEL), (sqrt_data, sqrt_labels, SQRT_LABEL)]

        meta_dataset = []
        meta_dataset_labels = []
        current_pattern = 0
        for i in range(META_NUMBER_MODELS):
            print "Working on model #", i
            model = grabAModel()
            print "Getting activation functions..."
            model_activation_functions = getModelActivationFunctions(model)
            print "Got 'em"
            ds = datasets[i%2]

            # Pre-train to avoid random beginnings
            model.fit(ds[0], ds[1], epochs=PRETRAINING_EPOCHS)

            for epoch in range(EPOCHS):
                # print "\tEPOCH #", epoch
                model.fit(ds[0], ds[1], epochs=1)

                acts = getModelActivations(model_activation_functions, ds[0])[0]

                # Keep a sequence of activations?
                # Normalize each activation sequence and create a histogram of them
                # just like with the single activation values?
                dists = getColumnDistributions(acts)
                input_vector = np.concatenate((dists, convertToOneHot(PRETRAINING_EPOCHS+epoch, INPUT_ONE_HOT_LIMIT)))

                meta_dataset.append(input_vector)
                meta_dataset_labels.append(ds[2])

        meta_dataset = np.array(meta_dataset)
        meta_dataset_labels = np.array(meta_dataset_labels)
        print meta_dataset.shape, meta_dataset_labels.shape
        createH5Dataset(META_DATASET_FILENAME, meta_dataset, meta_dataset_labels)


if __name__ == "__main__":
    pathDatasetCreation()




