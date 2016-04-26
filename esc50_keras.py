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
from keras.callbacks import History
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from sklearn.svm import SVC
from keras.preprocessing.image import ImageDataGenerator
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class ShapeWrapper(BaseEstimator):
    
    def __init__(self, estimator, to_shape=(-1,)):
        self.estimator = estimator
        self.to_shape = to_shape
        
    def _transform(self, X):
        Xreshaped = X.view()
        new_shape = (X.shape[0],) + self.to_shape
        Xreshaped.shape = new_shape
        return Xreshaped
        
    def fit(self, X, y):
        Xreshaped = self._transform(X)
        self.estimator.fit(Xreshaped, y)
        return self
    
    def predict(self, X):
        Xreshaped = self._transform(X)
        return self.estimator.predict(Xreshaped)
    
    def transform(self, X, y=None):
        Xreshaped = self._transform(X)
        return self.estimator.transform(Xreshaped)
        
SAMPLELEN = 110272

params = {'compute_features':False, 'compute_baseline': False, 'compute_cnn': True, \
          'standardize_data':True, 'augment_data': True, \
          'ncoeff': 20, 'fft': 2048, 'hop': 1024, \
          'nclasses': 50, 'nsamples':2000, \
          'nfolds': 1, 'split':.25, \
          'bsize': 128, 'nepoch': 300}

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
    return cv
        
def standardize (X_train, X_test):
    mu = np.mean (X_train, axis=0)
    de = np.std (X_train, axis=0)
    
    X_train = (X_train - mu) / de
    X_test = (X_test - mu) / de
    return X_train, X_test

def svm_classify(X_train, X_test, y_train, y_test, params):
    svm = SVC(C=1., kernel="linear")
    rsvm = ShapeWrapper(svm)
    rsvm.fit(X_train, y_train)
    y_pred = rsvm.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)

    return score, cm
    
def rate_scheduler (epoch):
    if epoch < 75:
        return 0.01
    else:
        print ("lower learning rate")
        return 0.01 / 10
        
def cnn_classify(X_train, X_test, y_train, y_test, params):
    nb_classes = params["nclasses"]
    batch_size = params["bsize"]
    nb_epoch =params["nepoch"]
    img_channels = 1
    img_rows = X_train.shape[2]
    img_cols = X_train.shape[3]
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    model = Sequential()
    
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                             input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
#    model.add(Convolution2D(32, 3, 3))
#    model.add(Activation('relu'))
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
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
                 
    bl = History()
    lr = LearningRateScheduler(rate_scheduler)
    
    if params["augment_data"]:
        print ("augment data...")
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
        
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            callbacks=[bl, lr],
                            validation_data=(X_test, Y_test))       
    else:
        model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              callbacks=[bl, lr],
              validation_data=(X_test, Y_test),
              shuffle=True)
    return bl
            
if __name__ == "__main__":
    print ("ESC-50 classification with Keras");
    print ("")    

    if params["compute_features"] == True:
        print ("computing features...")
        X_data, y_data = compute_features ("../../datasets/ESC-50-master", params)
        print ("saving features...")
        np.save ('params', params)
        np.save ('X_data', X_data)
        np.save ('y_data', y_data)
    else:
        print ("reloading features...")
        X_data = np.load ("X_data.npy")
        y_data = np.load ("y_data.npy")    

    X_data.astype('float32')
    y_data.astype('uint8')
        
    print ("making folds...")
    cv = create_folds(X_data, y_data, params)
        
    cnt = 1
    for train_index, test_index in cv:
        print ("----- fold: " + str (cnt) + " -----")
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        if params["standardize_data"] == True:
            print ("standardizing data...")
            X_train, X_test = standardize(X_train, X_test)
        
        
        if params["compute_baseline"] == True:
            print ("computing linear SVM baseline...")
            score, cm = svm_classify(X_train, X_test, y_train, y_test, params)
            print ("score " + str(score))
            plt.matshow(cm)
            plt.title ('Confusion matrix')
            plt.show ()

        if params["compute_cnn"] == True:        
            print ("computing CNN classification...")
            bl = cnn_classify(X_train, X_test, y_train, y_test, params)
            plt.plot (bl.history['acc'])
            plt.plot(bl.history['val_acc'])
            plt.title ('Accuracy (train vs test)')
            plt.show ()
                
            plt.plot (bl.history['loss'])
            plt.plot(bl.history['val_loss'])
            plt.title ('Loss (train vs test)')
            plt.show ()
            
            np.save ('fold_acc'+str(cnt), bl.history['acc'])
            np.save ('fold_val_acc'+str(cnt), bl.history['val_acc'])
            np.save ('fold_loss'+str(cnt), bl.history['loss'])
            np.save ('fold_val_loss'+str(cnt), bl.history['val_loss'])
        
        cnt = cnt + 1
#eof
    