import random
import torch
import pytorch_lightning as pl
from copy import deepcopy
from stereo_to_3D import stereo_to_3D
from architecture import surrogate_arch_mod
import argparse
import numpy as np

def parseargs():
    p = argparse.ArgumentParser(description="Runs genetic algorithm")
    p.add_argument('micros_3D',help="Path to folder containing 3D micros files")
    p.add_argument('micros_2D',help="Path to .csv file containing microstruture characteristics")
    p.add_argument("-g","-generations", type=int, default=20, help="Generations of networks")
    p.add_argument("-mr","-mut_rate", type=float, default=0.2, help="Mutation rate")
    p.add_argument("-cr","-cross_rate", type=float, default=0.8, help="Crossover rate")
    p.add_argument("-ml","-max_layers", type=int, default=5, help="Maximum layers")
    p.add_argument("-ps","-pop_size", type=int, default=10, help="Population size")
    p.add_argument("-nr","-neuron_range",type=int,nargs=2,default=[32,512],help='Range of neurons to include in the training.')
    p.add_argument("-lf","-loss_fn",type=str,default="MSE")
    p.add_argument("-pin","-params_in",type=int,nargs='+',default=[0,1,2,3,4,5,6,7,8,9,10,11,12],help='Column indexes for characteristics to train on. Should be formatted as a list of integers seperated by a single space ' '.')
    p.add_argument("-pout","-params_out",type=int,nargs='+',default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],help='Column indexes for characteristics to train on. Should be formatted as a list of integers seperated by a single space ' '.')
    p.add_argument("-ga","-gamma",type=float,default=0.9)
    p.add_argument("-e","-epochs",type=int,default=100)

    args = p.parse_args()
    
    return args

args = parseargs()

# Define mutation and crossover probabilities as GLOBAL VARIABLES
MUTATION_RATE = args.mr
CROSSOVER_RATE = args.cr
POPULATION_SIZE = args.ps
GENERATIONS = args.g
MAX_LAYERS = args.ml  # Limit max number of layers
NEURON_RANGE = args.nr  # Min and max neurons per layer
INPARAMS = args.pin
OUTPARAMS = args.pout
TARGET_PATH = args.micros_3D
INPUT_PATH = args.micros_2D
EPOCHS = args.e

def generate_random_architecture():
    """Generate a random architecture string."""
    num_layers = random.randint(1, MAX_LAYERS)
    return ','.join(f'd{random.randint(*NEURON_RANGE)}' for _ in range(num_layers))

def fitness(architecture_str, train_dataloader=None, val_dataloader=None):
    """Evaluate the architecture by training a model and computing validation loss."""

    print("Going to train",architecture_str)

    # Initialize model
    model = stereo_to_3D(architecture_str,
                                TARGET_PATH,INPUT_PATH,
                                INPARAMS,OUTPARAMS,
                                1.0e-4,32,0.2,None)  

    # Train model. add strategy=L.strategies.DDPStrategy(find_unused_parameters=False),
    # if on Joule
    trainer = pl.Trainer(max_epochs=EPOCHS, 
                            accelerator="auto",
                            # strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
                            strategy="ddp",
                            enable_progress_bar=True)
    # trainer.fit(model, train_dataloader, val_dataloader)
    trainer.fit(model)
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float('inf')))
    
    # Complexity penalty (reduce preference for large networks)
    num_neurons = sum(int(layer[1:]) for layer in architecture_str.split(','))
    penalty = 0.00001 * num_neurons
    
    return -val_loss.item() + penalty  # Lower loss is better

def mutate(architecture):
    """Randomly modify an architecture string."""
    layers = architecture.split(',')
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(layers) - 1)
        new_neurons = random.randint(*NEURON_RANGE)
        layers[idx] = f'd{new_neurons}'  # Modify one layer
    return ','.join(layers)

def crossover(parent1, parent2):
    """Perform one-point crossover between two architectures."""
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2  # No crossover, return as is
    
    p1_layers, p2_layers = parent1.split(','), parent2.split(',')
    split = random.randint(1, min(len(p1_layers), len(p2_layers)) - 1)
    child1 = ','.join(p1_layers[:split] + p2_layers[split:])
    child2 = ','.join(p2_layers[:split] + p1_layers[split:])
    return child1, child2

def genetic_algorithm():
    """Run the genetic algorithm to optimize network architecture."""
    population = [generate_random_architecture() for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}")

        savename = './generation_'+str(generation+1)+'.npy'
        np.save(savename,np.array(population))
        
        # Evaluate fitness
        fitness_scores = []
        for arch in population:
            fitness_scores.append(fitness(arch))
        
        # Select the best architectures (elitism)
        sorted_population = [arch for _, arch in sorted(zip(fitness_scores, population), reverse=True)]
        new_population = sorted_population[:2]  # Keep top 2 architectures
        
        # Create next generation
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(sorted_population[:5], 2)  # Select from top 5
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = new_population[:POPULATION_SIZE]  # Maintain population size
        
    
    # Return the best architecture found
    best_architecture = max(population, key=lambda arch: fitness(arch))
    return best_architecture, populations

if __name__ == "__main__":

    best_arch = genetic_algorithm()
    print("Best architecture found:", best_arch)
