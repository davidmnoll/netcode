# https://medium.com/spidernitt/breaking-down-neural-networks-an-intuitive-approach-to-backpropagation-3b2ff958794c
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

import math
import random
from pprint import pformat as pf
from pprint import pprint as pp
class Perceptron:

    def __init__(self):
        self.inputs = {}
        self.outputs = [] #TODO: remove outputs & only use inputs
        self.activation = 0
        self.activated = False
        self.learning_rate = .05
        self.bias = random.uniform(-1,1)
        while not self.bias:
            self.bias = random.uniform(-1,1)
        self.target = None
        self.d_totalerr_d_output_qty = 0

    # Transfer neuron activation
    def transfer(self, activation):
        return self.logistic_function(activation)

    def transfer_derivative(self, output):
        return self.logistic_partial_derivative(output)

    def node_error(self, output, target):
        return self.squared_error(output, target)

    def total_error_derivative(self, output, target):
        return self.squared_error_derivative(output, target)

    def squared_error(self, output, target):
        return (1/2) * ( (target - output) ** 2)

    def squared_error_derivative(self, output, target):
        return (output - target)

    def d_totalerr_d_nodeweight(self, output, target, node):
        dt_do = self.d_totalerr_d_output(output, target)
        do_di = self.d_output_d_totalinput(output)
        di_dw = self.d_totalinput_d_nodeweight(node)
        return dt_do * do_di * di_dw

    def add_d_totalerr_d_output(self, output, target):
        self.d_totalerr_d_output_qty = self.d_totalerr_d_output_qty + self.d_totalerr_d_output(output, target)


    def d_totalerr_d_output(self, output, target):
        return self.squared_error_derivative(output, target)

    def d_output_d_totalinput(self, node_output):
        self.transfer_derivative(node_output)

    def d_totalinput_d_nodeweight(self, node):
        # d (weight * node_activation + bias) / d (weight) = node_activation
        return self.inputs[node]["weight"]

    def logistic_partial_derivative(self, x):
        # derivatitve of 1.0 / (1.0 + math.exp(-x))
        return x(1-x)

    def logistic_function(self, x):
        return 1.0 / (1.0 + math.exp(-x))


    def train(self):
        for node, info in self.inputs:
            delta = self.d_totalerr_d_nodeweight(self.output, self.target, node)
            weight = info["weight"]
            new_weight = weight - (delta * self.learning_rate)
            self.inputs[node]["weight"] = new_weight
            node.update(self.node_error(self.transfer(self.activation), self.target))

    def add_input(self):
        p = Perceptron()
        self.attach_dendrite_to_axon(p)
        return p

    def accept_attachment_to_axon(self, node):
        self.outputs.append(node)

    def accept_detachment_from_axon(self, node):
        self.outputs.remove(node)

    def fire_axon(self, amt):
        if self.outputs:
            for output in self.outputs:
                output.fire_dendrite(self, amt)
        else:
            pp(self)
            print ( "activated - " + str(self.activation))
            if self.target:
                print ( "expected - " + str(self.target))
                self.train()

    def fire_dendrite(self, node, input_amt):
        self.activation = self.activation + (self.inputs[node]["weight"] * input_amt) + self.bias
        self.inputs[node]["activated"] = True
        if all([value["activated"] for key, value in self.inputs.items()]):
            self.fire_axon(self.transfer(self.activation))

    def attach_dendrite_to_axon(self, node):
        self.inputs[node] = { "weight": random.uniform(-1,1), "activated": False }
        while not self.inputs[node]["weight"]:
            self.inputs[node]["weight"] = random.uniform(-1,1)
        node.accept_attachment_to_axon(self)

    def detach_dendrite_from_axon(self, node):
        del(self.inputs[node]["weight"])
        node.accept_detachment_from_axon(self)

    def expect_answer(self, amt):
        self.target = amt



if __name__ == "__main__":
    out1 = Perceptron()
    out2 = Perceptron()
    hidden1 = out1.add_input()
    hidden2 = out1.add_input()
    out2.attach_dendrite_to_axon(hidden1)
    out2.attach_dendrite_to_axon(hidden2)
    pp(out1)
    pp(out2)
    pp(hidden1)
    pp(hidden1)
    out1.expect_answer(1)
    out2.expect_answer(1)
    hidden1.fire_axon(2)
    hidden2.fire_axon(4)
