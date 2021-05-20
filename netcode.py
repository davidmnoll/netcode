
from pprint import pprint as pp
import random
import numpy as np

def decorators(*args, **kwargs):
    def inner(func):
        '''
           do operations with func
        '''
        return func

    return inner  # this is the fun_obj mentioned in the above content


def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)



class TrainableFunction():

    class Perceptron():

        def __repr__(self):
            return self.name+": \n-weights:"+str(self.weights)+"\n-biases: "+str(self.biases)

        def __init__(self, name):
            self.name = name
            self.weights = []
            b = random.uniform(-1,1)
            while not b:
                b = random.uniform(-1,1)
            self.biases = []
            self.output = None
            self.prev_layer = []
            self.next_layer = []
            # self.inputs = []
            self.input_weights = []
            self.input_biases = []
            self.activation_fn = sigmoid
            self.activation_fn_deriv = deriv_sigmoid
            self.learning_rate = .01

        def get_activation(self, global_inputs):
            if self.prev_layer:
                if len(self.biases) != len(self.prev_layer):
                    self.biases = [random.uniform(-1,1) for i in range(0, len(self.prev_layer))]
                inputs = [x.get_activation(global_inputs) for x in self.prev_layer]
                x = sum(self.biases[i] + (self.weights[i] * input_val) for i, input_val in enumerate(inputs))
            else:
                if len(self.biases) != len(global_inputs):
                    self.biases = [random.uniform(-1,1) for i in range(0, len(global_inputs))]
                if not self.input_weights:
                    self.input_weights = [random.uniform(-1,1) for i in range(0, len(global_inputs))]
                z_s = []
                for i, input_val in enumerate(global_inputs):
                    bias = self.biases[i] 
                    z = bias + (self.input_weights[i] * input_val)
                    z_s.append(z)
                x = sum(z_s)
            self.output = self.activation_fn(x)
            return self.output


        def link_to_previous_layer(self, prevs):
            # pp(prevs)
            self.prev_layer = prevs
            for i, node in enumerate(self.prev_layer):
                self.weights.append(random.uniform(-1, 1))

        def backprop(self, error, global_inputs):
            pp(self)
            pp('backprop')
            pp(error)
            pp(self.prev_layer)
            c = error #actual - output
            if (self.prev_layer):
                for i, p in enumerate(self.prev_layer):
                    pp("hidden")
                    pp(i)
                    pp(p.output)
                    pp(self.weights)
                    delta_weight = error * self.activation_fn_deriv(p.output)
                    pp(delta_weight)
                    self.weights[i] = self.weights[i] + ( self.learning_rate * delta_weight )
                    p.backprop(error * self.weights[i], global_inputs)
                    #adjust bias
                    delta_bias = error
                    self.biases[i] = self.biases[i] + error
                    #adjust weight
            else:
                for i, w in enumerate(self.input_weights):
                    pp("input")
                    pp(i)
                    pp(w)
                    pp(self.weights)
                    delta_weight = error * self.activation_fn_deriv( w * global_inputs[i])
                    pp(delta_weight)
                    self.input_weights[i] = self.input_weights[i] + ( self.learning_rate * delta_weight )
                    #adjust bias
                    delta_bias = error
                    self.biases[i] = self.biases[i] + error
                    #adjust weight

        def loss(self, expected, actual):
            return (expected - actual)**2


    def __call__(self, *args):
        return self.output_perc.get_activation(args)



    def __init__(self, training, dimensions):
        self.weights = [[]]
        self.training = training
        self.last_call = None
        self.training = training()
        (first_in, first_out) = next(self.training)
        self.inputs = []
        self.outputs = []
        self.hiddens = []
        self.learning_rate = .1
        self.average_losses = []
        self.n = 0
        self.call_n = 0
        self.last_outputs = None

        for i in range(0, len(first_in)):
            self.inputs.append(TrainableFunction.Perceptron("input:"+str(i)))
        for i, num in enumerate(dimensions):
            if len(self.hiddens) <= i:
                self.hiddens.append([])
            new_layer = []
            for j in range(0, num):
                new_layer.append(TrainableFunction.Perceptron("hidden"+str(i)+":"+str(j)))
                if i == 0:
                    new_layer[j].link_to_previous_layer(self.inputs)
                else:
                    new_layer[j].link_to_previous_layer(self.hiddens[i-1][0])
            self.hiddens[i].append(new_layer)
        for i in range(0, len(first_out)):
            self.outputs.append(TrainableFunction.Perceptron("output:"+str(i)))
            self.outputs[i].link_to_previous_layer(self.hiddens[-1][0])
        self.__call__(*first_in)
        self.train(first_out)

    def __call__(self, *args):
        self.call_n = self.call_n + 1
        self.last_call = args
        self.last_outputs = [x.get_activation(args) for x in self.outputs]
        return self.last_outputs

    def train(self, y):
        self.n = self.n + 1
        deltas = []

        for i, out_neur in enumerate(self.outputs):
            if len(self.average_losses) < len(self.outputs):
                self.average_losses.append(0)
            error = out_neur.output - y[i]
            self.average_losses[i] = (self.average_losses[i] * (self.n - 1) ) + self.loss(self.last_outputs[i], y[i])
            deltas.append(out_neur.backprop(error, self.last_call))
        return deltas

    def loss(self, expected, actual):
        return (expected - actual)**2

    def train_loop(self, n):
        for i in range(1, n):
            inputs, outputs = next(self.training)
            if i % 1 == 0:
                pp("cycle"+str(i))
                print("inputs: "+str(inputs))
                result = self.__call__(*inputs)
                print("result: "+str(result))
                print("outputs: "+str(outputs))
            self.train(outputs)


def training_data():
    while True:
        int1 = random.random()/10 
        int2 = random.random()/10
        input_tup = (int1,int2)
        yield (input_tup, [int1 + int2])


if __name__ == "__main__":
    trained_add = TrainableFunction(training_data, [3,2])
    # pp(trained_add(9,3))
    trained_add.train_loop(10)

    pp(trained_add)
    pp(trained_add(3, 2))



