from netcode.perceptron.perceptron import Perceptron
from typing import List

class Network:
  
  def __init__(self, shape: List[int], learning_rate):
    self.nodes = []
    for i in range(len(shape) - 2):
      self.nodes.append([])
      for j in range(shape[-1-i]):
        self.nodes[i].append(Perceptron("layer" + str(i) + "-" + str(j)))
        if i > 0:
          for k in range():
            self.nodes[i - 1][k].attach_input(self.nodes[i][j])
    self.learning_rate = learning_rate
    self.total_errors = []
    self.shape = shape

  def get_output(self):
    return [node.get_output() for node in self.outputNodes]
  

  def train(self, data_generator):
    epochs = 10000
    tolerance = 0.05
    # Train the network and return True if error is below the tolerance
    total_errors = [0 for i in range(self.shape[-1])]
    for epoch in range(epochs):
      for data, target in next(data_generator):
        predictions = []
        errors = []
        for i in range(len(data)):
          self.nodes[0].set_input(data[i])

        for output in self.nodes[-1]:
          predictions.append(output.get_output())

        for i, target_bit in enumerate(target):
          errors.append(predictions[i] - target_bit)
          total_errors[i] += errors[i] ** 2

        if (epoch % 1000 == 0):
          self.total_errors.append(total_errors)
          print(total_errors)

        for i in range(len(target)):
          self.nodes[-1][i].train(target[i] * 1.0)
    for total_error in total_errors:
      if total_error < tolerance:
        return True
      else:
        raise Exception("Network did not converge to a solution.")
