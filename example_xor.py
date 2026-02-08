"""
Example: Solving the XOR problem using Natural Selection AI

This example demonstrates how to evolve a neural network to solve
the classic XOR (exclusive or) problem using genetic algorithms.
"""

from Network import Network

def evaluate_xor(genome):
    """
    Evaluate a genome's performance on the XOR problem.
    
    XOR truth table:
    Input (A, B) -> Output
    (0, 0) -> 0
    (0, 1) -> 1
    (1, 0) -> 1
    (1, 1) -> 0
    """
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    total_error = 0
    for inputs, expected in test_cases:
        outputs = genome.activate(inputs)
        # Calculate error for this test case
        error = abs(outputs[0] - expected)
        total_error += error
    
    # Convert error to fitness (lower error = higher fitness)
    # Use negative error so higher fitness is better
    genome.score = -total_error

def main():
    print("=" * 60)
    print("Natural Selection AI - XOR Problem Example")
    print("=" * 60)
    
    # Create network with 2 inputs, 4 hidden neurons, 1 output, population of 50
    print("\nInitializing network...")
    print("  - Input neurons: 2")
    print("  - Hidden neurons: 4")
    print("  - Output neurons: 1")
    print("  - Population size: 50")
    
    network = Network(input=2, hidden=4, output=1, pop=50)
    
    print("\nTraining for 100 generations...")
    print("-" * 60)
    
    best_genome = None
    for generation in range(100):
        # Evaluate all genomes in the population
        for genome in network.genomes:
            evaluate_xor(genome)
        
        # Perform selection and create next generation
        best_genome = network.select(show_best=True)
        
        # Check if we've found a good solution
        if best_genome.score > -0.1:  # Very low error
            print(f"\n✓ Solution found at generation {generation}!")
            break
    
    print("-" * 60)
    print("\nTesting best genome:")
    print("=" * 60)
    
    # Test the best genome
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    for inputs, expected in test_cases:
        outputs = best_genome.activate(inputs)
        predicted = round(outputs[0])
        status = "✓" if predicted == expected else "✗"
        print(f"{status} Input: {inputs} -> Output: {outputs[0]:.4f} (expected: {expected}, predicted: {predicted})")
    
    # Show network architecture
    schema = best_genome.get_schema()
    print(f"\nFinal network architecture: {schema}")
    print(f"Total neurons: {sum(schema)}")
    
    # Save the trained network
    print("\nSaving trained network to 'xor_solution.npz'...")
    network.save("xor_solution.npz")
    print("✓ Network saved successfully!")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
