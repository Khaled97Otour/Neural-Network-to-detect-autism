# Neural-Network-to-detect-autism
Detecting autism by building a neural network RNN model (LSTM) that read sequence of frames to determine
whether the movement normal or suspicious

# video generators was done by https://github.com/metal3d/keras-video-generators

Requirements are:
- Python >= 3.6 (Python 2 will never been supported)
- OpenCV
- numpy
- Keras >= 2
- TensorFlow >= 1.15 (or other backend, not tested, TensorFlow is needed by Keras)

After build my dataset I have to create my model which was done by two steps.

# First I build two CNN lightweight model (mobile_net and squeezenet):

- The input image was (60,60,3).
- The out put of each model was a Global pool 2D.

# Second I Concatenate this two model and create a RNN model :

- I used a GRU.
- 5 frames were used for Sequential and create the motion. 
- SGD optimizer was used to have the best result.
