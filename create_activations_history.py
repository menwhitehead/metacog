from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
import numpy as np

from misc_functions import *
from metacog_datasets import *
from metacog_buildingblock_functions import *
from metacog_globals import *
from metacog_models import *


def getActivationsFullHistory(datasets):
    # Every activation goes in here.  Can you believe it?
    # Every function class, individual model, epoch, problem #, hidden node
    # activation is stored in one giant history
    giant_numpy_history = np.zeros((META_NUMBER_FUNCTION_CLASSES,
                                    META_NUMBER_MODELS,
                                    EPOCHS,
                                    INPUT_NUMBER_PROBLEMS,
                                    HIDDEN_SIZE))

    for dataset_number in range(len(datasets)):
        ds = datasets[dataset_number]
        p = np.random.permutation(len(ds[0]))

        X = ds[0][p]
        y = ds[1][p]
        for model_number in range(META_NUMBER_MODELS):
            model = grabAModel()
            model_activation_functions = getModelActivationFunctions(model)
            for epoch in range(EPOCHS):
                print "Working on dataset %d, model %d, epoch %d" % (dataset_number, model_number, epoch)
                model.fit(X, y, epochs=1)
                acts = getModelActivations(model_activation_functions, X)[0]
                giant_numpy_history[dataset_number, model_number, epoch, :, :] = acts

    return giant_numpy_history



if __name__ == "__main__":
    datasets = getDatasets()
    history = getActivationsFullHistory(datasets)
    np.save(ACTIVATION_HISTORY_FILENAME, history)










