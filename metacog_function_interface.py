import numpy as np
from misc_functions import *
from metacog_globals import *



def getRandomPolynomialFunction():
    return lambda x: random.randint(1, 25) * x**2 + random.randint(1, 25) * x + random.randint(1, 25)

def getPolynomialFunctions(n):
    functions = []
    for i in range(n):
        functions.append(getRandomPolynomialFunction())
    return functions


# Convert a list of inputs from floating point values each to a single
# normalized input node value
def single_input(inputs):
    return np.expand_dims(inputs, -1)

def single_output(outputs):
    return np.expand_dims(outputs, -1)


def createDataset(function, inputs, input_conversion_function, output_conversion_function):
    outputs = np.apply_along_axis(function, 0, inputs)
    converted_inputs = input_conversion_function(inputs)
    converted_outputs = output_conversion_function(outputs)

    norm_inputs = np.array(normalize(converted_inputs, 0, 1))
    norm_outputs = np.array(normalize(converted_outputs, 0, 1))
    return norm_inputs, norm_outputs

def getDatasets(start_range=FUNCTIONS_RANGE_START,
                end_range=FUNCTIONS_RANGE_END,
                number_problems=FUNCTIONS_NUMBER_PROBLEMS,
                functions=ELEMENTARY_FUNCTIONS):
    datasets = []
    input_range = np.linspace(start_range, end_range, number_problems)
    function_count = 0
    for function, input_conversion_function, output_conversion_function in functions:
        func_data, func_labels = createDataset(function, input_range,
                                               input_conversion_function,
                                               output_conversion_function)
        datasets.append((func_data, func_labels, convertToOneHot(function_count, len(functions))))
    return datasets




if __name__=="__main__":
    ds = getDatasets()
    print ds
