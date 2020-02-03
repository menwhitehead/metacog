from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import scipy.misc
import numpy as np
import cv2
import os.path
import os

from misc_functions import *
from metacog_datasets import *
from metacog_buildingblock_functions import *
from metacog_globals import *
from metacog_models import *



def getWeightsFullHistory(datasets):
    # Every weight goes in here.  Can you believe it?
    # Every function class, individual model, epoch, hidden node
    # all weights are stored in one giant history
    giant_numpy_history = np.zeros((META_NUMBER_FUNCTION_CLASSES,
                                    META_NUMBER_MODELS,
                                    EPOCHS,
                                    HIDDEN_SIZE))

    for dataset_number in range(len(datasets)):
        ds = datasets[dataset_number]
        p = np.random.permutation(len(ds[0]))

        X = ds[0][p]
        y = ds[1][p]
        for model_number in range(META_NUMBER_MODELS):
            model = grabAModel()
            for epoch in range(EPOCHS):
                print "Working on dataset %d, model %d, epoch %d" % (dataset_number, model_number, epoch)
                model.fit(X, y, epochs=1)
                weights = model.get_weights()
                # print model.summary()
                # print weights[0]#.shape
                # print weights[1]#.shape
                # print weights[2]#.shape  # BIAS?
                # print weights[3]#.shape  # BIAS?
                giant_numpy_history[dataset_number, model_number, epoch, :] = weights[1]
            K.clear_session()

    return giant_numpy_history




if __name__ == "__main__":
    datasets = getDatasets()
    history = getWeightsFullHistory(datasets)
    np.save(WEIGHT_HISTORY_FILENAME, history)

    history = getWeightsFullHistory(datasets)
    np.save(WEIGHT_HISTORY_VALIDATION_FILENAME, history)










