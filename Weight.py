import random

class Weight:
    """
    Represents a connection weight between two neurons in the neural network.
    
    Attributes:
        from_neuron: The source neuron of the connection
        to_neuron: The target neuron of the connection
        weight: The strength/weight of the connection (initialized randomly)
    """
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = random.uniform(-3, +3)
    
    def activate(self):
        """
        Propagate the signal from source neuron to target neuron.
        Multiplies the source neuron's value by this connection's weight
        and adds it to the target neuron's value.
        """
        self.to_neuron.value += self.from_neuron.value * self.weight
