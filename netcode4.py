# https://medium.com/spidernitt/breaking-down-neural-networks-an-intuitive-approach-to-backpropagation-3b2ff958794c
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# http://cs229.stanford.edu/notes2020spring/cs229-notes-deep_learning.pdf

import math
import random
from pprint import pformat as pf
from pprint import pprint as pp


class Perceptron:

    def __init__(self, name=None):
        self.name = name
        self.inputs = {}
        self.input = None
        #logistic curve by default
        self.transfer = lambda x: 1.0 / (1.0 + math.exp(-x))
        #partial derivative of logistic curve
        self.d_transfer = lambda x: x * (1.0 - x)
        #error function
        self.err = lambda x, y: ((y - x)**2) / 2
        #partial derivative of error function
        self.d_err = lambda x, y: (x - y)
        self.learning_rate = 0.05

        self.bias = 0
        while not self.bias:
            self.bias = random.uniform(-1,1)

    def __repr__(self):
        if self.name:
            if self.inputs:
                return self.name + "<" + ",".join([ node.__repr__() for node, info in self.inputs.items()]) + ">"
            else:
                return self.name
        else:
            return super.__repr__(self)

    def get_output(self):
        if self.input is None:
            x = 0
            for node, info in self.inputs.items():
                x = x + node.get_output() * info["weight"]
            return self.transfer(self.bias + x)

        else:
            return self.input

    def _train_internal(self, from_downstream):
        '''
        pd = partial derivative = d[x]/d[y]
        self = current perceptron
        downstream = moving away from input, toward output node
        upstream = moving toward input, away from output node
        out = output
        err = error
        net= "net input" / activation / output before transfer function

        notes:
            - d[err_self]/d[out_self] == d[err_total]/d[out_self]

        during output layer:
            from_downstream = d[err_self]/d[out_self] = pd of [error of this output neuron] wrt. [output of this output perceptron]
            to_upstream = d[err_self]/d[out_upstream] = pd of [error of this output neuron] wrt. [output of upstream perceptron]
                = d[err_self]/d[out_self] * d[out_self]/d[net]          *    d[net]/d[out_upstream]
                = (output - acutal)       * (output * ( 1 - output ) )  *    weight between self & upstream perceptron
                = from_downstream * pd of [output of this neuron] wrt. ["net input" of neuron i.e. output before transfer function]
            d_err_d_node_weight =
            new weight =

        during hidden layer:
            from_downstream =
                = d[err_outputperceptron]/d[out_outputperceptron] * d[out_outputperceptron]/d[net_outputperceptron] * d[net_outputperceptron]/d[out_self]
                = pd of [error of output perceptron] wrt. [output of this hidden perceptron]
            to_upstream = d[err_outputperceptron] = pd of [error of output perceptron] wrt. [net input of this perceptron]
                = d[err_outputperceptron]/d[net_ou] * d[out_self]/d[net]          *    d[net]/d[out_upstream]
                = from_downstream       * (output * ( 1 - output ) )  *    weight between self & upstream perceptron
                = from_downstream * pd of [output of this neuron] wrt. ["net input" of hidden neuron i.e. output before transfer function]
            d_err_d_node_weight =
            new weight =


        '''

        output = self.get_output()
        to_upstream_partial = from_downstream * self.d_transfer(output)
        for node, info in self.inputs.items():
            node_output = node.get_output()
            node._train_internal(to_upstream_partial * info["weight"])
            # pp("previous: " + str(info["weight"]) )
            d_err_d_node_weight = to_upstream_partial * self.d_transfer(node_output) * node_output
            self.inputs[node]["weight"] = info["weight"] - ( d_err_d_node_weight * self.learning_rate )
            # pp("new: " + str(info["weight"]) )

    def train(self, target):
        output = self.get_output()
        pp(self)
        pp("target")
        pp(target)
        pp("old")
        pp(output)
        d_err_d_out = self.d_err(output, target) # partial derivative of error of this neuron wrt. output of this neuron
        self._train_internal(d_err_d_out)
        pp("new")
        pp(self.get_output())

    def set_input(self, x):
        if not self.inputs:
            self.input = x

    def add_input_node(self, name=None):
        p = Perceptron(name)
        self.init_synapse(p)
        return p

    def attach_input(self, node):
        self.init_synapse(node)

    def init_synapse(self, p):
        if p not in self.inputs:
            self.inputs[p] = {"weight": random.uniform(-1,1)}
        else:
            self.inputs[p]["weight"] = random.uniform(-1, 1)
        while not self.inputs[p]["weight"]:
            self.inputs[p]["weight"] = random.uniform(-1,1)

if __name__ == "__main__":
    out1 = Perceptron("o1")
    out2 = Perceptron("o2")
    hidden1 = out1.add_input_node("h1")
    hidden2 = out1.add_input_node("h2")
    out2.attach_input(hidden1)
    out2.attach_input(hidden2)
    pp(out1)
    pp(out2)
    pp(hidden1)
    pp(hidden2)
    hidden1.set_input(1)
    hidden2.set_input(1)
    out1.train(2)
    out2.train(4)
    out1.train(2)
    out2.train(4)
    out1.train(2)
    out2.train(4)
    out1.train(2)
    out2.train(4)
    out1.train(2)
    out2.train(4)
