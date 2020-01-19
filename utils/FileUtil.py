'''
    Utility class for file IO
'''
# Imports
import numpy as np

# Constants
TRAINING_FOLDER_PATH = "../training_data/"
TRAINING_DATA_NAMES = ["apple.npy", "banana.npy", "broccoli.npy", "carrot.npy", "grapes.npy", "mushroom.npy", "pineapple.npy", "strawberry.npy", "watermelon.npy"]
TRAINING_DATA_INDEX_NAME_DICT = {a: b for a, b in enumerate(TRAINING_DATA_NAMES)}
TRAINING_DATA_NAME_INDEX_DICT = {b: a for a, b in enumerate(TRAINING_DATA_NAMES)}


# Methods
def load_data(data_name, normalize=False):
    data = np.load(TRAINING_FOLDER_PATH + data_name)
    if normalize:
        data = data / 255.0
    return data


def load_all_data(normalize=False):
    return [load_data(name, normalize) for name in TRAINING_DATA_NAMES]


def get_index(name):
    return TRAINING_DATA_NAME_INDEX_DICT[name]


def get_name(index, is_display_name=False):
    return TRAINING_DATA_INDEX_NAME_DICT[index][0].upper() + TRAINING_DATA_INDEX_NAME_DICT[index][1:-4] if is_display_name else TRAINING_DATA_INDEX_NAME_DICT[index]


if __name__ == "__main__":
    print("testing starts")
    d = load_data(TRAINING_DATA_NAMES[0], normalize=True)
    print(type(d))
    for i in range(10):
        for a in range(28):
            for b in range(28):
                print("{:4}".format(d[i][a * 28 + b]), end=" ")
            print()
        print("\n\n")
