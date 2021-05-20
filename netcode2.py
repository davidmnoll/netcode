
import random
import numpy

class NoTrainingSetGeneratorError(Exception):
    pass

class NeuralNetNode:

    def __init__(self, prev_layer=None, next_layer=None):
        self.last_inputs = None
        self.weights = {}
        self.inputs = []

    def __call__(self, *args):
        # TODO: write activation function
        activation = 0.
        return activation

    def register_input(self, node):
        self.inputs.append(node)
        b = random.uniform(-1,1)
        while not b:
            b = random.uniform(-1,1)
        self.weights[node] = b


class NeuralNet:

    def __init__(self, dims):
        pass

class TrainableFunction:

    def __init__(self, **kwargs):
        self.n = 0
        self._training_set_generator = None
        if kwargs.get('tests', False):
            self._training_set_generator = self._test_based_training_set_generator(kwargs.get('tests'))
        if kwargs.get('data', False):
            data = kwargs.get('tests')
            if (data.get('y', False)):
                if (data.get('x', False)):
                    if (len(data['y']) == len(data['x'])):
                        self._training_set_generator = self._data_based_training_set_generator()
                    else:
                        raise NoTrainingSetGeneratorError("training set features and labels don't match")
                else:
                    raise NoTrainingSetGeneratorError("no training set features provided")
            else:
                raise NoTrainingSetGeneratorError("no training set labels provided")
        if kwargs.get('copy_function'):
            self._training_set_generator = self._function_based_training_set_generator(kwargs.get('tests'))
        if self._training_set_generator is None:
            raise NoTrainingSetGeneratorError("no training set generation method provided")

    def __call__(self, *args, **kwargs):
        pass

    def _test_based_training_set_generator(self, test):
        pass

    def _data_based_training_set_generator(self):
        pass

    def _function_based_training_set_generator(self, function):
        def training_set_function():
            pass
        return training_set_function


    def _train_step(self):
        x, y_actual = self._training_set_generator
        y_pred = self(*x)
        self.feedback(*y_actual)
        self.n = self.n + 1



