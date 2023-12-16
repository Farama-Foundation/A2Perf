# give me a template script with name==main and app.run absl
import collections
import glob
import os
import pickle

import matplotlib.pyplot as plt
from absl import app
from absl import flags

_DESIGN_FILE_PATH = flags.DEFINE_string('design_file_path',
                                        '../../a2perf/domains/web_navigation/web_navigation/environment_generation/data/',
                                        'Path to directory containing design files')
_DIFFICULTY_LEVELS_TO_PLOT = flags.DEFINE_list('difficulty_levels_to_plot',
                                               ['1', '5', '10'],
                                               'Difficulty levels to plot')


def main(_):
    # glob the design file path to get all designs
    design_file_path = _DESIGN_FILE_PATH.value

    # the file should end with pkl and include "websites" in the name
    design_file_paths = glob.glob(os.path.join(design_file_path, '*websites*.pkl'))
    difficulty_levels_to_plot = _DIFFICULTY_LEVELS_TO_PLOT.value
    # Prepare dictionaries to hold data for each difficulty level
    num_primitives = collections.defaultdict(list)
    difficulty_numbers = collections.defaultdict(list)
    num_pages = collections.defaultdict(list)

    # Get all of the different website levels first
    for file_path in design_file_paths:
        difficulty_level = os.path.basename(file_path).split('.')[0].split('_')[
            1]  # extract difficulty level from file name
        if difficulty_level not in difficulty_levels_to_plot:
            continue

        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)

        # Collect data for each difficulty level
        for website in dataset:
            num_primitives[difficulty_level].append(website._total_num_primitives)
            difficulty_numbers[difficulty_level].append(website.difficulty)
            num_pages[difficulty_level].append(website.num_pages)

    # Now plot histograms for the number of primitives and the difficulty number for each difficulty level
    plt.figure(figsize=(18, 5))

    for i, difficulty_level in enumerate(difficulty_levels_to_plot, start=1):
        plt.subplot(1, 3, 1)
        plt.hist(num_primitives[difficulty_level], bins=20, alpha=0.5, label=f'Difficulty Level {difficulty_level}')
        plt.title('Number of Primitives')

        plt.subplot(1, 3, 2)
        plt.hist(difficulty_numbers[difficulty_level], bins=20, alpha=0.5, label=f'Difficulty Level {difficulty_level}')
        plt.title('Difficulty Scores')

        plt.subplot(1, 3, 3)
        plt.hist(num_pages[difficulty_level], bins=20, alpha=0.5, label=f'Difficulty Level {difficulty_level}')
        plt.title('Number of Pages')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    app.run(main)
