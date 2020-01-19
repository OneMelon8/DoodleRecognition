'''
    Reminder:
        This project will attempt to create a neural network that can identify drawings of food
    References:
        https://www.youtube.com/watch?v=pqY_Tn2SIVA&list=PLRqwX-V7Uu6Zs14zKVuTuit6jApJgoYZQ
'''
# Imports
import keras
import numpy as np
import random
import utils.FileUtil as fUtil
import utils.DisplayUtil as dUtil

# Constants / Configurations
TRAIN_TEST_RATIO = 0.8
EPOCHS = 20

# Option: create new / load existing
NEW_MODEL = False
# Option: load training data from cache
LOAD_FROM_CACHE = False
# Option: save training array to cache
SAVE_TO_CACHE = False
# Option: save trained neural network data
SAVE_NEURAL_NETWORK = False

# Code starts
if not LOAD_FROM_CACHE:
    # Load training / testing data
    print("Loading training and testing data...")
    training, testing = [], []  # Array of tuples: (image, answer)
    category_count = len(fUtil.TRAINING_DATA_NAMES)
    for category in fUtil.TRAINING_DATA_NAMES:
        # Print the progress
        print(">> Loading " + fUtil.get_name(fUtil.get_index(category)) + "... (" 
              +str(fUtil.get_index(category) + 1) + "/" + str(category_count) + ")")
        # Normalize the data for more efficiency
        data = fUtil.load_data(category, normalize=True)
        # Split the data into training data and testing data
        train_limit = int(len(data) * TRAIN_TEST_RATIO)
        index = fUtil.get_index(category)
        # Append the current data to master data list
        training += [(image_data, [1 if a == index else 0 for a in range(category_count)]) for image_data in data[:train_limit]]
        testing += [(image_data, [1 if a == index else 0 for a in range(category_count)]) for image_data in data[train_limit:]]
    
    # Shuffle training / testing data
    print("Shuffling training and testing data...")
    random.shuffle(training)
    random.shuffle(testing)
    
    # Unpack to x => y format
    print("Unpacking data -- training... (1/2)")
    # Data structure: [(q1, a1), (q2, a2), ...]
    train_x, train_y = np.array([a[0] for a in training]), np.array([a[1] for a in training])
    print("Unpacking data -- testing... (2/2)")
    test_x, test_y = np.array([a[0] for a in testing]), np.array([a[1] for a in testing])
    
    if SAVE_TO_CACHE:
        # Save numpy array to cache
        print("Saving current numpy arrays to cache...")
        np.save("../cache/train_x.npy", train_x)
        np.save("../cache/train_y.npy", train_y)
        np.save("../cache/test_x.npy", test_x)
        np.save("../cache/test_y.npy", test_y)
else:
    # Load cached data
    print("Loading data from cache -- training... (1/2)")
    train_x, train_y = np.load("../cache/train_x.npy"), np.load("../cache/train_x.npy")
    print("Loading data from cache -- testing... (2/2)")
    test_x, test_y = np.load("../cache/test_x.npy"), np.load("../cache/test_y.npy")
    
    # Shuffle cached data
    print("Shuffling training data... (1/4)")
    np.random.shuffle(train_x)
    print("Shuffling training data... (2/4)")
    np.random.shuffle(train_y)
    print("Shuffling testing data... (3/4)")
    np.random.shuffle(test_x)
    print("Shuffling testing data... (4/4)")
    np.random.shuffle(test_y)

# Neural network model code starts...
if NEW_MODEL:
    # Create neural network
    print("Building neural network model...")
    # Sequential model because this is easy stuff
    model = keras.Sequential([
            keras.layers.Dense(64, input_shape=(784,), activation="relu"),
            keras.layers.Dense(len(fUtil.TRAINING_DATA_NAMES), activation="softmax")
        ])
    # Set optimizer and loss function
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=["accuracy"])
else:
    # Load the neural network
    print("Using saved neural network from models/my_model.h5")
    model = keras.models.load_model("../models/my_model.h5")

# Log the model summary
print(model.summary())

# Train the neural network for x epochs
print("Training the neural network...")
# model.fit(train_x, train_y, epochs=EPOCHS)

if SAVE_NEURAL_NETWORK:
    # Save model
    print("Saving the neural network...")
    model.save("../models/my_model.h5") 

# Test the neural network
print("Testing the neural network...")
result = model.evaluate(test_x, test_y, verbose=2)
print("Testing results: " + str(result))

# Interactive testing
while (True):
    # Get data from user's drawing
    data = np.array([dUtil.show_drawable_canvas()])
    # Get prediction
    prediction = model.predict(data).tolist()[0]
    # Soft max is a function that tells you the percentage of each outcome
    # The maximum of those percentages are selected as the most probable outcome
    prediction_dict = {fUtil.get_name(a, is_display_name=True): b for a, b in enumerate(prediction)}
    # Sort and print the data
    print("Best match: " + fUtil.get_name(prediction.index(max(prediction)), is_display_name=True))
    for name, probability in sorted(prediction_dict.items(), key=lambda a: a[1], reverse=True):
        print(">> " + name + ": {0:.2%}".format(probability))

