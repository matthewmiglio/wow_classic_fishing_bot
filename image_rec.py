import numpy as np
import os
from collections import defaultdict
from constants import LOOT_COLOR_DATA
from scipy.spatial import cKDTree

COLOR_PALETTE = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (0, 0, 0),  # Black
]

def get_color_frequencies(image: np.ndarray) -> dict:
    """
    Returns a dictionary of 12 colors and the frequency of pixels that align with that color.

    Args:
        image (np.ndarray): A numpy array representing the image, expected to be of shape (height, width, 3).

    Returns:
        dict: A dictionary where keys are RGB color tuples and values are their pixel frequencies.
    """
    # Initialize the dictionary to count frequencies
    color_count = defaultdict(int)

    # Initialize the color count dictionary with all colors set to zero
    for color in COLOR_PALETTE:
        color_count[color] = 0

    # Reshape the image to a 2D array of pixels (each pixel is a 3-element RGB tuple)
    height, width, _ = image.shape
    pixels = image.reshape(-1, 3)

    # Create a k-d tree from the color palette
    kdtree = cKDTree(COLOR_PALETTE)

    # Find the closest color in the COLOR_PALETTE for each pixel
    distances, indices = kdtree.query(pixels)

    # Count the frequencies of each color
    for index in indices:
        closest_color = tuple(COLOR_PALETTE[index])
        color_count[closest_color] += 1

    # Convert the defaultdict to a regular dictionary before returning
    return dict(color_count)


def classification_scorer(image: np.ndarray) -> list:
    """
    Classifies the image and returns a list of tuples (class_name, score) based on color similarity.

    Args:
        image (np.ndarray): The image to classify.

    Returns:
        list: A list of tuples in the form (class_name, score).
    """
    # Step 1: Get the color frequencies of the image
    image_color_frequencies = get_color_frequencies(image)

    # Step 2: Compare the image color frequencies against each class in loot_dict
    scores = {}
    for class_name, class_color_frequencies in LOOT_COLOR_DATA.items():
        # Calculate the score for this class
        score = calculate_class_score(image_color_frequencies, class_color_frequencies)
        scores[class_name] = score

    return scores


def calculate_class_score(image_frequencies: dict, class_frequencies: dict) -> float:
    """
    Calculate a score for a class based on how well its color frequencies match the image.

    Args:
        image_frequencies (dict): The color frequencies of the image.
        class_frequencies (dict): The color frequencies for the class.

    Returns:
        float: A score between 0 and 1.
    """
    # We can compute the similarity as an inverse of the Euclidean distance
    distance = 0.0
    total_pixels = sum(image_frequencies.values())

    # Iterate through all color keys (we assume the color palette is the same for all classes)
    for color in COLOR_PALETTE:
        image_count = image_frequencies.get(color, 0)
        class_count = class_frequencies.get(color, 0)

        # We calculate the squared difference of the relative frequencies of each color
        image_relative_frequency = image_count / total_pixels if total_pixels > 0 else 0
        class_relative_frequency = class_count / total_pixels if total_pixels > 0 else 0

        distance += (image_relative_frequency - class_relative_frequency) ** 2

    # Step 4: Convert the distance to a score (using the inverse of the distance)
    score = 1 / (1 + distance)  # Higher scores mean more similar

    return score
