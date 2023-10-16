import neat
from experiment_config import *

if __name__ == '__main__':
	print("REP v_02 source code")
	print(f'seed(repetition id): {seed}')
	print(f'param1: {param1}')
	print(f'param2: {param2}')
	print(f'param3: {param3}')

	neat_config = neat.Config(neat.DefaultGenome,
                  neat.DefaultReproduction,
                  neat.DefaultSpeciesSet,
                  neat.DefaultStagnation,
                  'neat_config.cfg'
                  )
	print(f'neat_config: {neat_config.genome_config}, {neat_config.nodes}')
	
 
