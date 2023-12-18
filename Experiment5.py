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

#to get the link to tensorboard
#python -m tensorboard.main --logdir=./logs

def main():

    #Load the data
    data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv') 

    X = data.iloc[:,:11].values #independent variables
    y = data["target"].values #dependent variable

    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0 )

    from sklearn.preprocessing import StandardScaler 
    sc = StandardScaler() 
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test) 

    classifier = Sequential() 
    classifier.add(Dense(activation = "relu", input_dim = 11,  
                        units = 8, kernel_initializer = "uniform")) 
    classifier.add(Dense(activation = "relu", units = 28,  
                        kernel_initializer = "uniform")) 
    classifier.add(Dense(activation = "sigmoid", units = 1,  
                        kernel_initializer = "uniform")) 
    
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy',  
                    metrics = ['accuracy'] ) 
    
    #early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # Log messages
    restore_best_weights=True) # Restore model weights from the epoch with the best value of the monitored quantity

    log_dir = "logs/experiment5/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    classifier.fit(X_train , y_train , batch_size = 8 , epochs = 100, validation_data=(X_test, y_test), callbacks=[tensorboard_callback, early_stopping])

    y_pred = classifier.predict(X_test) 

    y_pred = (y_pred > 0.5) 

    cm = confusion_matrix(y_test,y_pred) 

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = [False, True]) 
    cm_display.plot()
    plt.savefig('confusion_matrix_plot.png', format='png')

    accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1]) 
    print(accuracy*100) 







if __name__ == '__main__':
  main()