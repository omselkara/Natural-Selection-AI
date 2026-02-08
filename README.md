# Natural Selection AI 🧬

A Python-based neural network library that evolves through natural selection and genetic algorithms. This library implements a neuroevolution system where neural networks evolve and improve over generations through mutation, crossover, and selection mechanisms inspired by biological evolution.

## 🌟 Features

- **Evolutionary Neural Networks**: Networks that evolve and adapt through generations
- **Genetic Algorithms**: Implementation of selection, crossover, and mutation operators
- **Dynamic Network Topology**: Networks can add/remove neurons and connections during evolution
- **Flexible Architecture**: Support for custom input, hidden, and output layer configurations
- **Population-Based Training**: Evolves populations of neural networks simultaneously
- **Save/Load Functionality**: Persist trained networks and resume evolution
- **Fitness-Based Selection**: Automatic selection of best performing networks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.6 or higher
- NumPy

### Install Dependencies

```bash
pip install numpy
```

### Clone the Repository

```bash
git clone https://github.com/omselkara/Natural-Selection-AI.git
cd Natural-Selection-AI
```

## ⚡ Quick Start

Here's a simple example to get you started:

```python
from Network import Network

# Create a network with 3 inputs, 5 hidden neurons, 2 outputs, and population of 100
network = Network(input=3, hidden=5, output=2, pop=100)

# Training loop
for generation in range(1000):
    for genome in network.genomes:
        # Your evaluation logic here
        inputs = [0.5, 0.3, 0.8]  # Example input
        outputs = genome.activate(inputs)
        
        # Assign fitness score based on performance
        genome.score = evaluate_performance(outputs)  # Your fitness function
    
    # Select best performers and create next generation
    best = network.select()

# Save the trained network
network.save("trained_model.npz")
```

## 🏗️ Architecture

The library consists of several key components:

### Core Classes

1. **Network**: Main class managing the population of genomes
2. **Genome**: Represents an individual neural network (a solution candidate)
3. **Layer**: Manages neurons in input, hidden, or output layers
4. **Neuron**: Individual processing unit with bias and connections
5. **Weight**: Connection between neurons with associated weight value
6. **Activation**: Activation function implementations (tanh)
7. **Selector**: Selection algorithms for choosing parents

### Network Structure

```
Input Layer → Hidden Layers → Output Layer
     ↓            ↓              ↓
  Neurons      Neurons        Neurons
     ↓            ↓              ↓
  Weights ←→   Weights ←→    Weights
```

## 📖 Usage Guide

### Creating a Network

```python
from Network import Network

# Parameters:
# - input: number of input neurons
# - hidden: initial number of hidden neurons
# - output: number of output neurons
# - pop: population size (number of genomes)
network = Network(input=4, hidden=10, output=2, pop=50)
```

### Evaluating and Training

```python
# Evaluate each genome in the population
for genome in network.genomes:
    # Feed forward through the network
    outputs = genome.activate([0.1, 0.2, 0.3, 0.4])
    
    # Calculate fitness (higher is better)
    # This is problem-specific
    genome.score = calculate_fitness(outputs)

# Perform selection and create next generation
best_genome = network.select(show_best=True)
```

### Mutation Operations

The library automatically performs various mutations:

- **Add Neuron**: Creates new neurons in hidden layers (1% probability)
- **Add Layer**: Adds new hidden layers (0.1% probability per existing layer)
- **Connect Neurons**: Creates new connections (20% probability)
- **Remove Connection**: Removes existing connections (5% probability)
- **Bias Mutation**: Adjusts neuron bias values (15-25% probability)
- **Weight Mutation**: Adjusts connection weights (15-25% probability)

### Genetic Crossover

```python
# Crossover happens automatically during selection
# Two parent genomes create offspring with combined traits
parent1 = network.genomes[0]
parent2 = network.genomes[1]
child = parent1.generate_baby(parent2)
```

### Saving and Loading

```python
# Save the network state
network.save("my_network.npz")

# Load a previously saved network
network.load("my_network.npz")
```

## 📚 API Reference

### Network Class

#### `__init__(input, hidden, output, pop)`
Initialize a new network population.

**Parameters:**
- `input` (int): Number of input neurons
- `hidden` (int): Initial number of hidden neurons
- `output` (int): Number of output neurons
- `pop` (int): Population size

#### `select(show_best=True)`
Perform natural selection and create next generation.

**Parameters:**
- `show_best` (bool): Print best genome statistics

**Returns:**
- Best performing genome from current generation

#### `save(name="save.npz")`
Save network state to file.

**Parameters:**
- `name` (str): Filename for saving

#### `load(name="save.npz")`
Load network state from file.

**Parameters:**
- `name` (str): Filename to load from

### Genome Class

#### `__init__(input, hidden, output, baby=False)`
Initialize a genome (neural network).

**Parameters:**
- `input` (int): Number of input neurons
- `hidden` (int): Number of hidden neurons
- `output` (int): Number of output neurons
- `baby` (bool): If True, create empty genome for crossover

#### `activate(inputs)`
Feed forward through the network.

**Parameters:**
- `inputs` (list): List of input values

**Returns:**
- list: Output values from output neurons

#### `mutate(repeat=1, connect=False)`
Apply random mutations to the genome.

**Parameters:**
- `repeat` (int): Number of mutation attempts
- `connect` (bool): Force connection mutations

#### `generate_baby(parent)`
Create offspring through genetic crossover.

**Parameters:**
- `parent` (Genome): Other parent genome

**Returns:**
- Genome: New offspring genome

#### `get_schema()`
Get simplified network architecture.

**Returns:**
- list: Layer sizes [input, hidden1, hidden2, ..., output]

#### `get_detailed_schema()`
Get complete network structure with all weights and biases.

**Returns:**
- list: Detailed schema for serialization

### Neuron Class

#### `activate()`
Compute neuron activation using weighted sum and tanh activation function.

#### `connect(to_neuron)`
Create a connection to another neuron.

**Parameters:**
- `to_neuron` (Neuron): Target neuron for connection

### Selector Functions

#### `calc_probability(scores, genome=True)`
Calculate selection probabilities based on fitness scores.

**Parameters:**
- `scores` (list): Fitness scores or genome objects
- `genome` (bool): If True, extract scores from genome objects

**Returns:**
- list: Cumulative probabilities for selection

#### `select(probabilities)`
Select an individual based on fitness probabilities.

**Parameters:**
- `probabilities` (list): Cumulative probability distribution

**Returns:**
- int: Selected index

## 💡 Examples

### Example 1: XOR Problem

```python
from Network import Network

def evaluate_xor(genome):
    """Evaluate genome on XOR problem"""
    test_cases = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    error = 0
    for inputs, expected in test_cases:
        outputs = genome.activate(inputs)
        error += abs(outputs[0] - expected[0])
    
    # Lower error = higher score
    genome.score = -error

# Create network
network = Network(input=2, hidden=4, output=1, pop=50)

# Evolve for 100 generations
for gen in range(100):
    for genome in network.genomes:
        evaluate_xor(genome)
    
    best = network.select()

network.save("xor_solution.npz")
```

### Example 2: Classification Task

```python
from Network import Network
import random

def evaluate_classifier(genome, dataset):
    """Evaluate genome on classification task"""
    correct = 0
    for data, label in dataset:
        outputs = genome.activate(data)
        predicted = outputs.index(max(outputs))
        if predicted == label:
            correct += 1
    
    genome.score = correct / len(dataset)

# Create network for classification
# 10 inputs, 20 hidden neurons, 3 output classes
network = Network(input=10, hidden=20, output=3, pop=100)

# Generate sample dataset
dataset = [(
    [random.random() for _ in range(10)],
    random.randint(0, 2)
) for _ in range(100)]

# Train
for generation in range(500):
    for genome in network.genomes:
        evaluate_classifier(genome, dataset)
    
    best = network.select()
    
    if generation % 50 == 0:
        print(f"Gen {generation}: Best accuracy = {best.score:.2%}")

network.save("classifier.npz")
```

### Example 3: Continuous Evolution

```python
from Network import Network

# Create network
network = Network(input=5, hidden=15, output=3, pop=150)

# Load previous training if exists
try:
    network.load("evolving_network.npz")
    print(f"Loaded network from generation {network.genereation}")
except:
    print("Starting fresh network")

# Continue evolution
for i in range(100):
    for genome in network.genomes:
        # Your fitness evaluation
        genome.score = your_fitness_function(genome)
    
    best = network.select()
    
    # Save progress every 10 generations
    if i % 10 == 0:
        network.save("evolving_network.npz")
```

## 🔬 How It Works

### Natural Selection Process

1. **Initialization**: Create a population of random neural networks
2. **Evaluation**: Assess each network's performance on the task
3. **Selection**: Choose parents based on fitness scores
4. **Crossover**: Combine two parents to create offspring
5. **Mutation**: Randomly modify offspring networks
6. **Replacement**: Form new generation
7. **Repeat**: Continue for many generations

### Genetic Encoding

Each genome stores:
- **Network topology**: Number and arrangement of layers and neurons
- **Connection weights**: Strength of connections between neurons
- **Biases**: Offset values for each neuron
- **Activation functions**: How neurons process signals

### Fitness-Proportionate Selection

Selection probability is proportional to fitness:
```
P(i) = fitness(i) / Σ fitness(all)
```

Better performing networks have higher chance of becoming parents.

### Mutation Strategies

The library uses multiple mutation strategies to explore the solution space:

- **Structural mutations**: Add/remove neurons and connections
- **Parametric mutations**: Adjust weights and biases
- **Adaptive rates**: Different probabilities for different mutation types

## 🎯 Best Practices

1. **Population Size**: Start with 50-100 genomes, increase for complex problems
2. **Hidden Neurons**: Begin with small networks, let evolution add complexity
3. **Fitness Function**: Design clear, differentiable fitness metrics
4. **Training Time**: Run for hundreds or thousands of generations
5. **Save Regularly**: Checkpoint your best networks periodically
6. **Experiment**: Try different population sizes and initial architectures

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- Add more activation functions (ReLU, sigmoid, etc.)
- Implement speciation to maintain diversity
- Add NEAT (NeuroEvolution of Augmenting Topologies) features
- Parallel fitness evaluation
- Visualization tools for network topology
- More example problems and benchmarks

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by NEAT (NeuroEvolution of Augmenting Topologies)
- Based on genetic algorithm principles
- Uses concepts from evolutionary computation

## 📞 Contact

- GitHub: [@omselkara](https://github.com/omselkara)
- Project Link: [https://github.com/omselkara/Natural-Selection-AI](https://github.com/omselkara/Natural-Selection-AI)

## 🔮 Future Plans

- [ ] Add more activation functions
- [ ] Implement speciation
- [ ] Create visualization tools
- [ ] Add comprehensive test suite
- [ ] Support for recurrent connections
- [ ] Multi-objective optimization
- [ ] Parallel processing support
- [ ] Web-based demo interface

---

Made with ❤️ for evolutionary computation enthusiasts
