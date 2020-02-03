import numpy as np
from misc_functions import *
from metacog_globals import *


def getDatasets(start_range=FUNCTIONS_RANGE_START,
                end_range=FUNCTIONS_RANGE_END,
                number_problems=FUNCTIONS_NUMBER_PROBLEMS,
                functions=ELEMENTARY_FUNCTIONS):
    datasets = []
    input_range = np.linspace(start_range, end_range, number_problems)
    function_count = 0
    for function in functions:
        func_data, func_labels = getFullDataset(input_range, function)
        datasets.append((func_data, func_labels, convertToOneHot(function_count, len(functions))))
        function_count += 1
    return datasets

def getDatasetsOLD():
    log_data, log_labels = getFullDataset(np.arange(1, 1+INPUT_NUMBER_PROBLEMS), np.log)
    sqrt_data, sqrt_labels = getFullDataset(np.arange(1, 1+INPUT_NUMBER_PROBLEMS), np.sqrt)
    exp_data, exp_labels = getFullDataset(np.linspace(1, 10, INPUT_NUMBER_PROBLEMS), np.exp)

    LOG_LABEL = np.array([1, 0, 0])
    SQRT_LABEL = np.array([0, 1, 0])
    EXP_LABEL = np.array([0, 0, 1])
    datasets = [(log_data, log_labels, LOG_LABEL),
                (sqrt_data, sqrt_labels, SQRT_LABEL),
                (exp_data, exp_labels, EXP_LABEL)]
    return datasets


def getRawDataset(values, function):
    #print type(values)
    return np.apply_along_axis(function, 0, values)

def getFullDataset(inputs, function):
    outputs = getRawDataset(inputs, function)
    norm_inputs = np.array(normalize(inputs, 0, 1))
    norm_outputs = np.array(normalize(outputs, 0, 1))

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
    result = normalize(result, 0, 100)
    binary_input = convertToBinary(n, INPUT_BINARY_LIMIT)
    return binary_input, result




if __name__=="__main__":
    ds = getDatasets()
    print ds
