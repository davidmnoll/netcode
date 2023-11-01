
import unittest
from netcode.trainable import Trainable
from random import uniform
from typing import Tuple


class TestTrainableFunction(unittest.TestCase):

    
  def test_trainable_xor_problem(self):

    dataset = [
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0)
    ]


    @Trainable(epochs=10000, tolerance=0.05, shape=[2,2,1])
    def test1(input: Tuple[bool, bool]) -> bool:
      yield dataset[uniform(0, 4)]
    
    # Test the network's predictions
    for data, target in dataset:
      prediction = test1(data)
      self.assertEqual(prediction, target, f"Failed for input: {data}. Expected {target}, but got {prediction}.")