import random

def getRandomPolynomialFunction():
    return lambda x: random.randint(0, 25) * x**5 + random.randint(0, 25) * x**4 + random.randint(0, 25) * x**3 + random.randint(0, 25) * x**2 + random.randint(0, 25) * x + random.randint(0, 25)

def getPolynomialFunctions(n):
    functions = []
    for i in range(n):
        functions.append(getRandomPolynomialFunction())
    return functions


