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
# on that analysis of the other's cognition? Dunno.

# IDEAS
# - Use activation paths as input to the meta-network
# - Use hardwired starting weights to make learning match from original training

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

def grabMetacogModel():
    inputs = Input(shape=(META_INPUT_SIZE,))
    output_1 = Dense(META_HIDDEN_SIZE, activation='relu')(inputs)
    dropout_1 = Dropout(META_DROPOUT_RATE)(output_1)
    output_2 = Dense(META_HIDDEN_SIZE, activation='relu')(dropout_1)
    dropout_2 = Dropout(META_DROPOUT_RATE)(output_2)
    predictions = Dense(META_NUMBER_FUNCTION_CLASSES, activation='softmax')(dropout_2)
    model = Model(inputs=inputs, outputs=predictions)
    opt = Adam(learning_rate=META_LEARNING_RATE)
    # opt = SGD(lr=META_LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model



if __name__ == "__main__":
    meta_dataset, meta_dataset_labels = loadH5Dataset(META_DATASET_FILENAME)
    meta_dataset_validation, meta_dataset_validation_labels = loadH5Dataset(META_DATASET_VALIDATION_FILENAME)
    metacog = grabMetacogModel()
    #metacog.fit(meta_dataset, meta_dataset_labels, batch_size=META_BATCH_SIZE, validation_split=.25, epochs=META_EPOCHS)
    metacog.fit(meta_dataset, meta_dataset_labels,
                batch_size=META_BATCH_SIZE,
                validation_data=(meta_dataset_validation, meta_dataset_validation_labels),
                epochs=META_EPOCHS,
                shuffle="batch")
    #print metacog.predict(meta_dataset)





