# https://medium.com/spidernitt/breaking-down-neural-networks-an-intuitive-approach-to-backpropagation-3b2ff958794c
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#   --- error?, d_transfer(output) should be d_transfer(net)?
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

    def get_activation(self):
        if self.input is None:
            x = 0
            for node, info in self.inputs.items():
                x = x + node.get_output() * info["weight"]
            return self.bias + x

        else:
            return self.input

    def get_output(self):
        if self.input is None:
            x = 0
            for node, info in self.inputs.items():
                x = x + node.get_output() * info["weight"]
            return self.transfer(self.bias + x)

        else:
            return self.input

    def _train_internal(self, delta_prev):
        '''
        J = E_tot = total error
        o = net activation of final layer = final output?
        z = net = net input = activation of first layer = bias + sum ( weights * previous outputs )
        a = out = node output = transfer_fn( bias + sum ( weights * previous outputs ) )
        dJ/d[a_i] = d[E_tot]/d[a_i] = d[E_i]/d[a_i] = d_err(a_i, target) = (a_i - target) for squared error function ???
            ... the part. deriv. of total error wrt. single output  == part. deriv. of single output error wrt. single output

        da/dz = d[out]/d[net] = d_transfer( z )
        do/da = d[final activation]/d[output from node] = weight of node
        dJ/do = d[E_i]/d[a_i] for output node = d_err(o, target)
        dJ/dz = dJ/do * do/da * da/dz = dJ/do * w * d_transfer(a)
        dJ/dw = delta_weight = dJ/dz * dz/dw = dJ/dz * input = delta * previous layer output

        delta_first = delta for output layer = dJ/do = d_err(o, target)
        delta_weight[i]_first = dJ/do * a[i] = delta_first * a[i]
        delta_bias = delta_first = dJ/do
        delta_second =  dJ/do * do/da * da/dz
            = delta_first * weight[btwn output & this] * d_transfer(z[this node activation])
        delta_weight[i]_second = delta_second * dz/dw = delta_second * input[i]
        delta_nth = dJ/dz[n-1] * dz[n-1]/da[n] * da[n]/dz[n]
            = delta_[n-1]th * weight[btwn [n-1]th & this] * d_transfer(z[this node activation])
        delta_weight[i]_nth = delta_nth * a[i][from [n+1]th layer node] = delta_nth * output from n+1 layer
        delta_bias = delta_nth
        '''
        # output = self.get_output()
        self.bias = self.bias - delta_prev
        for node, info in self.inputs.items():
            node_output = node.get_output()
            node_activation = node.get_activation()
            delta_weight = delta_prev * node_output
            new_delta = delta_prev * info["weight"] * self.d_transfer(node_activation)
            node._train_internal(new_delta)
            # pp("previous: " + str(info["weight"]) )
            self.inputs[node]["weight"] = info["weight"] - ( delta_weight * self.learning_rate )
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
    out1.train(-2)
    out2.train(-4)
    out1.train(-2)
    out2.train(-4)
    out1.train(-2)
    out2.train(-4)
    out1.train(-2)
    out2.train(-4)
    out1.train(-2)
    out2.train(-4)
