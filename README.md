# Neural-Network-to-detect-autism
Detecting autism by building a neural network RNN model (LSTM) that read sequence of frames to determine
whether the movement normal or suspicious

Requirements are:
- Python >= 3.6 (Python 2 will never been supported)
- OpenCV
- numpy
- Keras >= 2
- TensorFlow >= 1.15 (or other backend, not tested, TensorFlow is needed by Keras)

After building my dataset I have to create my model which was done by two steps.

## First I build two CNN lightweight model (mobile_net and squeezenet):

- The input image was (60 , 60 , 3 ).
- The output of each model was a Global pool 2D.
- the reason I use the lightweight model way to save memory and have a fast performer

## Second I Concatenate this two model and create a RNN model :

- I used a LSTM.
- 5 frames were used for Sequential and create the motion. 
- Adamax optimizer was used to have the best result.

## The results were the following :
![model_acc](https://user-images.githubusercontent.com/93203143/182431539-49edee40-5a83-4ba4-8e19-67a6c91d2608.PNG)
![model_loss](https://user-images.githubusercontent.com/93203143/182431542-03578be9-3bf3-4790-ac20-63f544efea6c.PNG)
