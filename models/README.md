# Generate Labels and train 

## Training Data

Framework: **OpenCV**.

How to get the training dataset? 
Sampling method: BlinkDB.

-sample from the whole video. 

## Label 

Framework: **keras + tensorflow**
Model: **YoloV3** 
File format:
|Object|probability|x-left|x-right|y-top|y-bottom|

## Train Cheap Model

Framework: **Keras + tensorflow** 
Model: Light weight model
- Copying from BlazeIt is OK?
- Tiny ResNet (predict the counts.) 
We have to make clear of the output of this tiny ResNet.


## Other tricks

- Skip frames (number of frames) 
- Difference detector.

