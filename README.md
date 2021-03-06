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
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)           [(None, 5, 60, 60, 0                                            
__________________________________________________________________________________________________
time_distributed_1 (TimeDistri (None, 5, 1024)      3239808     input_27[0][0]                   
__________________________________________________________________________________________________
time_distributed_2 (TimeDistri (None, 5, 512)       735424      input_27[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)    (None, 5, 1536)      0           time_distributed_16[0][0]        
                                                                 time_distributed_17[0][0]        
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 5, 256)       1836032     concatenate_55[0][0]             
__________________________________________________________________________________________________
lstm_2 (LSTM)                  (None, 64)           82176       lstm_9[0][0]                     
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 64)           0           lstm_10[0][0]                    
__________________________________________________________________________________________________
dense_32 (Dense)                (None, 2)            130         dropout_4[0][0]                  

. Total params: 5,893,570
. Trainable params: 5,871,682
. Non-trainable params: 21,888

## The results were the following :

![model_acc](https://user-images.githubusercontent.com/93203143/182432733-e89f64f3-2f4d-484e-800e-6e38f564693d.PNG)
![model_loss](https://user-images.githubusercontent.com/93203143/182432738-8befe49f-4eed-41e1-bdb3-bab9ba2232cb.PNG)
