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

import tsfresh as tsf
import tsfresh.utilities.dataframe_functions as tsfutils
import pandas as pd

from misc_functions import *
from metacog_datasets import *
from metacog_buildingblock_functions import *
from metacog_globals import *
from metacog_models import *


def distributionMaker(values, intervals=META_HIST_SIZE):
    norm_values = normalize(values)
    hist = np.array(np.histogram(norm_values, bins=META_HIST_SIZE)[0], dtype="float")
    hist[hist == 0] = META_HIST_ZERO_PROB
    norm_hist = normalize(hist, 0, 1)
    return norm_hist


def getColumnDistributions(acts):
    result = []
    rows, cols = acts.shape
    for col in range(cols):
        col_act = acts[:, col]
        result.append(distributionMaker(col_act))
    return np.concatenate(result)


# Create histograms of model weights for each epoch
def createHistogramDataset(all_weights, output_filename):
    datasets = getDatasets()
    meta_dataset = []
    meta_dataset_labels = []
    current_pattern = 0
    for func_num in range(META_NUMBER_FUNCTION_CLASSES):
        ds = datasets[func_num%2]
        for epoch_num in range(EPOCHS):
            acts = all_weights[func_num, :, epoch_num, :]
            dists = getColumnDistributions(acts)
            input_vector = np.concatenate((dists, convertToOneHot(PRETRAINING_EPOCHS+epoch_num, INPUT_ONE_HOT_LIMIT)))
            meta_dataset.append(input_vector)
            meta_dataset_labels.append(ds[2])
    meta_dataset = np.array(meta_dataset)
    meta_dataset_labels = np.array(meta_dataset_labels)
    print meta_dataset.shape, meta_dataset_labels.shape
    createH5Dataset(output_filename, meta_dataset, meta_dataset_labels)



def createRecurrentDataset(all_weights, output_filename):
    datasets = getDatasets()
    meta_dataset = []
    meta_dataset_labels = []
    current_pattern = 0
    for func_num in range(META_NUMBER_FUNCTION_CLASSES):
        ds = datasets[func_num%2]
        for epoch_num in range(EPOCHS):
            acts = all_weights[func_num, :, epoch_num, :]
            dists = getColumnDistributions(acts)
            input_vector = np.concatenate((dists, convertToOneHot(PRETRAINING_EPOCHS+epoch_num, INPUT_ONE_HOT_LIMIT)))
            meta_dataset.append(input_vector)
            meta_dataset_labels.append(ds[2])
    meta_dataset = np.array(meta_dataset)
    meta_dataset_labels = np.array(meta_dataset_labels)
    print meta_dataset.shape, meta_dataset_labels.shape
    createH5Dataset(output_filename, meta_dataset, meta_dataset_labels)




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



def allModelsHistogramDataset():
    log_data, log_labels = getFullDataset(np.arange(1, 101), np.log)
    sqrt_data, sqrt_labels = getFullDataset(np.arange(1, 101), np.sqrt)
    LOG_LABEL = np.array([1, 0])
    SQRT_LABEL = np.array([0, 1])
    datasets = [(log_data, log_labels, LOG_LABEL), (sqrt_data, sqrt_labels, SQRT_LABEL)]

    meta_dataset = []
    meta_dataset_labels = []
    current_pattern = 0

    # Create a bunch of models for the first dataset
    all_models = []
    all_activation_functions = []

    for dataset in datasets:
        dataset_models = []  # all the models for this dataset
        dataset_activations = []
        for i in range(META_NUMBER_MODELS/len(datasets)):
            print "Working on model #", i
            model = grabAModel()
            print "Getting activation functions..."
            model_activation_functions = getModelActivationFunctions(model)
            dataset_activations.append(model_activation_functions)
            print "Got 'em"
            # ds = datasets[0]  # only the first function

            # Pre-train to avoid random beginnings
            # model.fit(dataset[0], dataset[1], epochs=PRETRAINING_EPOCHS)
            dataset_models.append(model)
        all_models.append(dataset_models)
        all_activation_functions.append(dataset_activations)


    for epoch in range(EPOCHS):
        for i in range(len(all_models)):
            dataset_models = all_models[i]
            ds = datasets[i]
            dataset_acts = []
            for j in range(len(dataset_models)):  #model in dataset_models:
                print "j:", j
                model = dataset_models[j]
                activations = all_activation_functions[i][j]
                model.fit(ds[0], ds[1], epochs=1)
                acts = getModelActivations(activations, ds[0])[0]
                dataset_acts.append(acts)
            dataset_acts = np.vstack(np.array(dataset_acts))
            print dataset_acts.shape
            # sys.eit()
            dists = getColumnDistributions(dataset_acts)
            # print dists.shape
            # sys.exit()
            input_vector = np.concatenate((dists, convertToOneHot(PRETRAINING_EPOCHS+epoch, INPUT_ONE_HOT_LIMIT)))
            meta_dataset.append(input_vector)
            meta_dataset_labels.append(ds[2])



    meta_dataset = np.array(meta_dataset)
    meta_dataset_labels = np.array(meta_dataset_labels)
    print meta_dataset.shape, meta_dataset_labels.shape
    createH5Dataset(META_DATASET_FILENAME, meta_dataset, meta_dataset_labels)





def generateTimeSeriesFeaturesDataset(total_activation_history):

    # STACK UP ALL ACTIVATION HISTORIES WITH LABELS
    # TO FIND USEFUL FEATURES!!!!!
    final_time_series = []

    for activation_history, label in total_activation_history:
        print activation_history.shape

        epochs, problems, nodes = activation_history.shape

        for problem in range(problems):
            problem_frames = []
            for node in range(nodes):
                problem_frames.append(activation_history[:, problem, node])
            np_frames = np.array(problem_frames)
            # np_frames = np.transpose(np_frames)

            print "STACKED: ", np_frames.shape

            d = pd.DataFrame(np_frames)
            st = d.stack()
            st.index.rename(['id', 'time'], inplace=True)
            st = st.reset_index()
            print st
            # sys.exit()
            #feats = tsf.extract_features(st, column_id="id", column_sort="time")
            # feats = tsf.feature_extraction.feature_calculators.binned_entropy(st, 10)
            feats = tsf.feature_extraction.feature_calculators.sample_entropy(st)
            #tsfutils.impute(feats)
            print feats
            #feature_values = feats.values[:, 0:3]
            #print feature_values
            sys.exit()
        #print feats['0__variance_larger_than_standard_deviation']
    # print feature_values
    # print feature_values.shape
    sys.exit(1)




def pathDatasetCreation():
        log_data, log_labels = getFullDataset(np.arange(1, 100), np.log)
        sqrt_data, sqrt_labels = getFullDataset(np.arange(1, 100), np.sqrt)
        LOG_LABEL = np.array([1, 0])
        SQRT_LABEL = np.array([0, 1])
        datasets = [(log_data, log_labels, LOG_LABEL), (sqrt_data, sqrt_labels, SQRT_LABEL)]

        meta_dataset = []
        meta_dataset_labels = []
        current_pattern = 0
        total_activation_history = []
        for i in range(META_NUMBER_MODELS):
            print "Working on model #", i
            model = grabAModel()
            print "Getting activation functions..."
            model_activation_functions = getModelActivationFunctions(model)
            print "Got 'em"
            ds = datasets[i%2]

            # Pre-train to avoid random beginnings
            #model.fit(ds[0], ds[1], epochs=PRETRAINING_EPOCHS)

            # A complete history of all the model's internal activations
            activation_history = np.zeros((EPOCHS, len(ds[0]), HIDDEN_SIZE))
            for epoch in range(EPOCHS):
                # print "\tEPOCH #", epoch
                model.fit(ds[0], ds[1], epochs=1)

                acts = getModelActivations(model_activation_functions, ds[0])[0]
                print acts.shape
                activation_history[epoch] = acts

                # Keep a sequence of activations?
                # Normalize each activation sequence and create a histogram of them
                # just like with the single activation values?
                # dists = getColumnDistributions(acts)
                # input_vector = np.concatenate((dists, convertToOneHot(PRETRAINING_EPOCHS+epoch, INPUT_ONE_HOT_LIMIT)))
                #
                # meta_dataset.append(input_vector)
                # meta_dataset_labels.append(ds[2])

                #sequences = generateSequenceDataset(activation_history)
                # for sequence in sequences:
                #     meta_dataset.append(sequence)
                #     meta_dataset_labels.append(ds[2])

            total_activation_history.append((activation_history, ds[2]))

        time_series_features = generateTimeSeriesFeaturesDataset(total_activation_history)
        # time_series_features = generateActivationChainsDataset(total_activation_history)


        meta_dataset = np.array(meta_dataset)
        meta_dataset_labels = np.array(meta_dataset_labels)
        print meta_dataset.shape, meta_dataset_labels.shape
        createH5Dataset(META_DATASET_FILENAME, meta_dataset, meta_dataset_labels)




if __name__ == "__main__":
    all_weights = np.load(WEIGHT_HISTORY_FILENAME)
    createHistogramDataset(all_weights, META_DATASET_FILENAME)

    all_weights = np.load(WEIGHT_HISTORY_VALIDATION_FILENAME)
    createHistogramDataset(all_weights, META_DATASET_VALIDATION_FILENAME)










