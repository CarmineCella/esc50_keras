# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:11:14 2016

ESC-50 dataset classification with Keras 

@author: Carmine E. Cella, 2016, ENS Paris
"""

import os
import fnmatch
import librosa
import numpy as np
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
    
SAMPLELEN=110272
compute = False

params = {'ncoeff': 20, 'fft': 4096, 'hop': 2048, \
          'nclasses': 50, 'nsamples':2000, \
          'nfolds': 3, 'split':.25, \
          'bsize': 32, 'nepoch': 2}

def compute_features (root_path, params):
    nframes = int(SAMPLELEN / params['hop']);
        
    X_data = np.zeros ((params['nsamples'], 1, params['ncoeff'], nframes))
    y_data = np.zeros ((params['nsamples']))
    
    classes = 0
    samples = 0
    for root, dir, files in os.walk(root_path):
        print ("class: " + root.split("/")[-1])
        waves = fnmatch.filter(files, "*.wav")

        if len(waves) != 0:
            for items in waves:
                print ("\tsample: " + items)
                y, sr = librosa.load(root + '/' + items)
                C = librosa.feature.mfcc(y=y, sr=sr, hop_length=params['hop'],n_mfcc = params['ncoeff'])
                C = C[:params['ncoeff'],:nframes]
                X_data[samples, 0, :C.shape[0], :C.shape[1]] = C
                y_data[samples] = classes
                samples = samples + 1
            classes = classes + 1            
        print ("")
            
    print ("classes = " + str (classes))
    print ("samples = " + str (samples))

    return X_data, y_data
    
def create_folds (X_data, y_data, params):    
    cv = StratifiedShuffleSplit(y_data, n_iter=params['nfolds'], test_size=params['split'])
    
    for train_index, test_index in cv:
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
    return X_train, X_test, y_train, y_test, cv
    
    
def svm_classify(X_data, y_data, cv, params):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    svm = SVC(C=1., kernel="linear")
    
    pipeline = make_pipeline(StandardScaler(), svm)

    X_flatten = X_data.view()
    X_flatten.shape = X_flatten.shape[0], -1

    scores = cross_val_score(pipeline, X_flatten, y_data, cv=cv, scoring="accuracy")
    return scores, cv
    
def vgg_classify(X_train, X_test, y_train, y_test, params):
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, params["nclasses"])
    Y_test = np_utils.to_categorical(y_test, params["nclasses"])
    
    model = Sequential()
    
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(1, X_train.shape[2], X_data.shape[3])))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params["nclasses"]))
    model.add(Activation('softmax'))
    
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

    model.fit(X_train, Y_train,
            batch_size=params["bsize"],
            nb_epoch=params["nepoch"],
            validation_data=(X_test, Y_test),
            shuffle=True)
                
if __name__ == "__main__":
    print ("ESC-50 classification with Keras");
    print ("")    

    if compute == True:
        print ("computing features...")
        X_data, y_data = compute_features ("../../datasets/ESC-50-master", params)
        print ("saving features...")
        np.save ('X_data', X_data)
        np.save ('y_data', y_data)
    else:
        print ("reloading features...")
        X_data = np.load ("X_data.npy")
        y_data = np.load ("y_data.npy")    

    print ("making folds...")
    X_train, X_test, y_train, y_test, cv = create_folds(X_data, y_data, params)
        
    print ("computing linear SVM basline...")
    scores, cv = svm_classify(X_data, y_data, cv, params)
    print ("scores: " + str(scores))
    
    print ("computing vgg CNN classification...")
    vgg_classify(X_train, X_test, y_train, y_test, params)
    
    