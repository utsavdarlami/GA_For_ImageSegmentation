"""Base class for image segmentation."""
import numpy as np
import pygad
from pygad import GA
from typing import Callable, Type
from loguru import logger


class GASegmentation():
    """
    Performs segmentation using Genetic Algorithm by finding k-1 thresholds \
    where k is the number of segments.

    [WIP]
    """

    def __init__(self,
                 fitness_func: Callable[[np.ndarray, np.ndarray], float],
                 n_classes: int = 4,
                 num_generations: int = 100):
        """
        Initialize the image segmentation class.

        Args:
         x_image : The image which is to be segmented
        Return:
         None
        """
        self.fitness_func = fitness_func

        self.n_classes = n_classes
        self.num_genes = self.n_classes - 1
        self.last_fitness = 0.0
        # Number of generations.
        self.num_generations = num_generations
        # Number of solutions to be selected as parents in the mating pool.
        self.num_parents_mating = 7
        self.gene_type = int
        self.init_range_low = 0
        self.init_range_high = 255
        self.sol_per_pop = 50

        # Initialze the Genetic Algorithm using pygad
        self._initialize_ga()

    def _initialize_ga(self):
        self.ga_instance = pygad.GA(num_generations=self.num_generations,
                                    num_parents_mating=self.num_parents_mating,
                                    fitness_func=self.fitness_func,
                                    sol_per_pop=self.sol_per_pop,
                                    num_genes=self.num_genes,
                                    gene_type=self.gene_type,
                                    init_range_low=self.init_range_low,
                                    init_range_high=self.init_range_high,
                                    on_generation=GASegmentation.callback_generation
                                    )

    @staticmethod
    def callback_generation(ga_instance: Type[GA]):
        """Log the results of each generation."""
        solution, solution_fitness, _ = ga_instance.best_solution()
        logger.info(f"Generation = {ga_instance.generations_completed}")
        logger.info(f"Fitness    = {solution_fitness}")
        logger.info(f"Solution   = {solution}")

    def segment(self, x_image: np.ndarray) -> np.ndarray:
        """Perform Segmentation."""
        # Running the GA to optimize the thresholds.
        self.ga_instance.run()
        solution, _, _ = self.ga_instance.best_solution()
        if not isinstance(x_image, np.ndarray):
            raise TypeError(f"x_image should not be Type of {type(x_image)}")
        segmented_image = np.digitize(x_image, bins=solution)
        return segmented_image

    def draw_plot(self):
        """
        After the generations complete, some plots are showed that summarize \
        the how the outputs/fitenss values evolve over generations.

        None
        """
        self.ga_instance.plot_fitness()

    def log_best_thresholds(self):
        """Log the details of the best solution."""
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
