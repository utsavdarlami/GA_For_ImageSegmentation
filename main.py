"""Segment passed image."""
import argparse
import os
import numpy as np
from scipy import stats
from skimage import color, io
import sys
from loguru import logger
from segmentation import GASegmentation

def is_sorted(a: np.ndarray) -> bool:
    """Check whether the array is in sorted or not in ascending order."""
    if not isinstance(a, np.ndarray):
        return False
    return np.all(a[:-1] <= a[1:])


def entropy_of_pixel(pixels: np.ndarray, base=None) -> float:
    """Compute entropy of pixel distribution."""
    _, counts = np.unique(pixels, return_counts=True)
    ent = stats.entropy(counts)
    return ent


def calculate_segment_entropy(image: np.ndarray,
                              thresh: np.ndarray = None) -> float:
    """
    Caculate the average entropy of the regions segmented \
    using the thresholds.

    Args:
      image:
      thresh:
    Results:
       Avg of the entropy for `len(thresh) + 1` segments
    """
    if not isinstance(thresh, np.ndarray):
        return 0.0

    regions = np.digitize(image, bins=thresh)
    sum_of_entropies = 0.0
    for idx in range(len(thresh) + 1):
        segmented_pixels = image[regions == idx]
        region_entropy = entropy_of_pixel(segmented_pixels)
        sum_of_entropies += region_entropy

    return sum_of_entropies/(len(thresh)+1)


def fitness_function(solution: np.ndarray,
                     solution_idx: np.ndarray) -> float:
    """
    Calculate the fitness value of a \
    solution/state in the current population.

    Args:
    Returns:
    """
    global x_image
    if not is_sorted(solution):
        return 0.0
    # Entropy of the segmented pixel could be a one measure for the fitness
    # we can also add variance as the mesasure of fitness
    fitness = calculate_segment_entropy(x_image,
                                        thresh=solution)
    return fitness


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Image path")
    parser.add_argument("n", type=int, help="Number of segments", default=2)
    args = parser.parse_args()

    image_path = args.path
    n_classes = args.n

    logger.info(image_path)

    x_image = io.imread(image_path, as_gray=True)

    image_segmenter = GASegmentation(fitness_func=fitness_function,
                                     n_classes=n_classes,
                                     num_generations=100)

    logger.info("Performing Segmentation of image")
    seg_img = image_segmenter.segment(x_image)

    image_name = os.path.split(image_path)
    output_name = "segmented_" + image_name[1]

    seg_img = color.label2rgb(seg_img, bg_label=0)
    seg_img = (seg_img * 255).clip(0, 255).round().astype(np.uint8)

    logger.info(f"Writing the segmented image to {output_name}")
    io.imsave(output_name, seg_img)
    logger.info("| Done |")
