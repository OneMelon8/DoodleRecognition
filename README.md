Simple Doodle Recognition
==================
# About
This is a doodle recognition neural network written in python using Google's TensorFlow library. The training data is gathered from the "**Quick, Draw!**" doodle dataset (https://github.com/googlecreativelab/quickdraw-dataset). Currently, the models are only trained to identify doodles as *apples, bananas, broccoli, carrots, grapes, mushrooms, pineapples, strawberries, and watermelons*.


# Experimentation
I experimented with different amounts of hidden layers, the number of hidden nodes, the activation functions, the optimizer, and the loss function. Each of my experimentation networks can be found in the **/model** folder under a name that states the general information about the model.


# Navigation
The runner file is the **/main/Main.py** file. There are also some utility files that I wrote for loading data and displaying them; those will be located in **/utils/\*Util.py**. To run the doodle recognition model and test upon it, simply download the **numpy** data from the Quick, Draw! dataset and import them into the **/training_data** folder with "\*.npy" file extension. As mentioned before, the saved models can be found in **/model** folder under respective names.


To use the custom draw interface, hold **left click** to draw on the canvas; **double left click** to clear the canvas; and **middle click** to submit the doodle for the neural network to classify. The results should be printed into the console after a short delay.


# Demonstration
![Image tkinter drawing interface](https://i.imgur.com/GZCuNOi.png)
![Image classification results](https://i.imgur.com/XI86blz.png)
