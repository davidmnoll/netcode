
import unittest
from netcode.perceptron.perceptron import Perceptron

class TestPerceptronBackpropagation(unittest.TestCase):

    def setUp(self):
        # This method sets up the neural network for the tests.
        self.output_neuron = Perceptron("output")
        self.hidden_neurona1 = self.output_neuron.add_input_node("hiddena1")
        self.hidden_neurona2 = self.output_neuron.add_input_node("hiddena2")
        self.hidden_neurona3 = self.output_neuron.add_input_node("hiddena3")

        self.hidden_neuronb1 = self.hidden_neurona1.add_input_node("hiddenb1")
        self.hidden_neuronb2 = self.hidden_neurona1.add_input_node("hiddenb2")

        self.hidden_neurona2.attach_input(self.hidden_neuronb1)
        self.hidden_neurona2.attach_input(self.hidden_neuronb2)

        self.hidden_neurona3.attach_input(self.hidden_neuronb1)
        self.hidden_neurona3.attach_input(self.hidden_neuronb2)




        self.dataset = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0)
        ]
        
    def train_network(self, epochs=10000, tolerance=0.05):
        # Train the network and return True if error is below the tolerance
        for epoch in range(epochs):
            total_error = 0
            for data, target in self.dataset:
                self.hidden_neuronb1.set_input(data[0])
                self.hidden_neuronb2.set_input(data[1])
                self.output_neuron.train(target * 1.0)
                prediction = self.output_neuron.get_output()
                total_error += (prediction - target) ** 2
            # if (epoch % 1000 == 0):
            #     print(total_error)
            if total_error < tolerance:
                return True
        return False
    
    def test_xor_problem(self):
        training_successful = self.train_network()
        self.assertTrue(training_successful, "Neural network did not converge to a solution.")
        
        # Test the network's predictions
        for data, target in self.dataset:
            self.hidden_neuronb1.set_input(data[0])
            self.hidden_neuronb2.set_input(data[1])
            prediction = round(self.output_neuron.get_output())
            self.assertEqual(prediction, target, f"Failed for input: {data}. Expected {target}, but got {prediction}.")