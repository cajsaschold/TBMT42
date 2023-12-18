import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

import datetime

###########################
# To recreate the figures #
###########################
#Run this code 5 times, then change the settings to the experiment2 settings presented in the abstract and change the folder path on
#line 94 to 'experment2'. Run the code 5 times for each experment1-experment4. Experiment 5 is in a seperate python script.
#After running each experiment 5 times each, run this command line in the python shell to get the link to tensorboard:

#python -m tensorboard.main --logdir=./logs

#click on the link you recive and go to the tab named 'scalars'. Select all of the plots for experiment1 with the regexp function
#Do this with all of the experiments --> now you have all of the figures.
#(the values of the accuracy could vary a bit from the results in the abstract, this is due to that NN uses random initialazation)


def main():

    #Load the data
    data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv') 

    sick = (data['target'] == 1).sum()
    not_sick = (data['target'] == 0).sum()
    others = len(data) - sick - not_sick

    print('Number of samples that is sick:', sick)
    print('Number of samples that is not sick:',not_sick)
    print('If there are any NULL values:',others)

    #Devide the variables
    X = data.iloc[:,:11].values #independent variables
    y = data["target"].values #dependent variable

    #Split data into Train and Test dataset 
    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0 )

    #Scale the data
    from sklearn.preprocessing import StandardScaler 
    sc = StandardScaler() 
    #sc learns the mean and standard deviation from the training dataset X_train (using fit_transform)
    X_train = sc.fit_transform(X_train) 
    #and then scales X_train using these parameters.
    X_test = sc.transform(X_test) 

    #Building the Model
    ###############################
    classifier = Sequential() 
    classifier.add(Dense(activation = "relu", input_dim = 11,  
                        units = 8, kernel_initializer = "uniform")) 
    classifier.add(Dense(activation = "relu", units = 14,  
                        kernel_initializer = "uniform")) 
    #The neural network is designed for binary classification, and the last layer uses a sigmoid activation function that outputs a probability value between 0 and 1
    classifier.add(Dense(activation = "sigmoid", units = 1,  
                        kernel_initializer = "uniform")) 
    
    #Configures the model for training
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy',  
                    metrics = ['accuracy'] ) 
    
    #create a folder named 'logs'
    log_dir = "logs/experiment1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Fitting the Model
    #batch_size=8: This defines the number of samples that will be used to update the model's weights in each iteration. 
    #epochs is the number of laps (forward and backward) the entire dataset will pass through the NN, i.e. how many times the learning algorithm will work through the entire training dataset.
    classifier.fit(X_train , y_train , batch_size = 8 , epochs = 100, callbacks=[tensorboard_callback])

    #Performing prediction and rescaling
    y_pred = classifier.predict(X_test) 
    #If the probability is over 0.5 it will belong in the positive class. i.e. they have heart disease
    y_pred = (y_pred > 0.5) 

    #Confusion Matrix
    cm = confusion_matrix(y_test,y_pred) 

    #plots the confusion matrix
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = [False, True]) 
    cm_display.plot()
    plt.savefig('confusion_matrix_plot.png', format='png')

    #Accuracy
    #the accuracy is calculated by the true values (true pos and true neg) devided by all of the values (i.e. number of individuals)
    accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1]) 
    print(accuracy*100) 







if __name__ == '__main__':
  main()