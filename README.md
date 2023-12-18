# Neural Network for Heart Disease Prediction

## Overview

The project investigates different neural network configurations for classifying heart disease based on a Python script described in [Geeks for Geeks article](https://www.geeksforgeeks.org/heart-disease-prediction-using-ann/). 
TensorFlow is used for designing and implementing the neural network, and TensorBoard is used for result comparison.

## Experiments and Findings

Experiments 1 to 4 are executed directly in the Python script "Project.py" by modifying the neural network settings within the code. 
Experiment 5, which uses early stopping, is executed in a separate script named "Experiment5".

- **Experiment 1:**
  - Hidden Layers: 1
  - Units in Hidden Layer: 14

- **Experiment 2:**
  - Hidden Layers: 3
  - Units in Hidden Layer: 14

- **Experiment 3:**
  - Hidden Layers: 1
  - Units in Hidden Layer: 7

- **Experiment 4:**
  - Hidden Layers: 1
  - Units in Hidden Layer: 28

- **Experiment 5:**
  - Hidden Layers: 1
  - Units in Hidden Layer: 14
  - Early Stopping

## Code Execution

To analyse the neural networks in Tensorboard:

1. Run the provided code with the specified settings. 
2. For TensorBoard visualization, run the following command in the Python shell:
   python -m tensorboard.main --logdir=./logs
3. Click the provided link
