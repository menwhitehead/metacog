import numpy as np
from misc_functions import *
from metacog_globals import *


def getRawDataset(values, function):
    print type(values)
    return np.apply_along_axis(function, 0, values)
        
def getFullDataset(inputs, function):
    outputs = getRawDataset(inputs, function)
    norm_inputs = np.array(normalize(inputs))
    norm_outputs = np.array(normalize(outputs))
    
    # EXPAND IF NECESSARY
    norm_inputs = np.expand_dims(norm_inputs, -1)
    norm_outputs = np.expand_dims(norm_outputs, -1)
    
    return norm_inputs, norm_outputs

def getDataset(n, pattern_function):
    data = []
    labels = []
    for i in range(1, n):
        binary_input, one_hot_factors_vector = pattern_function(i)
        data.append(binary_input)
        labels.append(one_hot_factors_vector)
    return data, labels

def getSinePattern(n):
    result = np.sin(n)
    result = normalize(result, -1, 1)
    binary_input = convertToOneHot(n, INPUT_ONE_HOT_LIMIT)
    # binary_input = convertToBinary(n, INPUT_BINARY_LIMIT)
    return binary_input, result

def getLogPattern(n):
    result = math.log(n, 2)
    result = normalize(result, 0, 10)  # don't allow logs over 10 for now
    binary_input = convertToOneHot(n, INPUT_ONE_HOT_LIMIT)
    # binary_input = convertToBinary(n, INPUT_BINARY_LIMIT)
    return binary_input, result

def getSquarePattern(n):
    result = math.exp(n, 2)
    result = normalize(result, 0, 100)  # don't allow logs over 10 for now
    binary_input = convertToBinary(n, INPUT_BINARY_LIMIT)
    return binary_input, result
