# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:11:14 2016

ESC-50 dataset classification with Keras 

@author: Carmine E. Cella, 2016, ENS Paris
"""

import os
import fnmatch
import joblib
import librosa
import numpy as np
import os.path
from sklearn.cross_validation import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import History
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from sklearn.svm import SVC
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class ShapeWrapper(BaseEstimator):
    def __init__(self, estimator, to_shape=(-1,)):
        self.to_shape = to_shape
        self.estimator = estimator
        
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

SAMPLELEN = 110250 # in samples is 5 sec @ 22050

params = {'plot':True, 'features':'cqt', \
          'compute_baseline': True, 'compute_cnn': True, \
          'standardize_data':True, 'augment_data': True, \
          'ncoeff': 100, 'hop': 2048, \
          'nclasses': 50, 'nsamples':2000, \
          'nfolds': 1, 'split':.2, \
          'bsize': 128, 'nepoch': 400}

def get_features (file, features, hop, bins):
    y = np.zeros(SAMPLELEN);   
    yt, sr = librosa.core.load  (file, mono=True)
    
    if len(yt) == 0: 
        print ('*** warning: empty file -> ' + file + '! ***')
        return 0

    min_length = min(len(y), len(yt))
    y[:min_length] = yt[:min_length]
    
    if features == 'mfcc':
        C = librosa.feature.mfcc(y=y, sr=sr, hop_length=params['hop'],n_mfcc = params['ncoeff'])    
    elif features == 'cqt':
        C = np.log1p(1000 * np.abs (librosa.core.cqt( y=y, sr=sr, hop_length=hop, n_bins=bins, real=False)))    
    else:
        print('unsupported features')
        return 0
    return C

cachedir = os.path.expanduser('~/esc50_joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=1)
cached_get_features = memory.cache(get_features)

def compute_features (root_path, params):
    hop = params['hop']
    bins = params['ncoeff']
    features = params['features']
        
    y_data = np.zeros ((params['nsamples']))
    
    classes = 0
    samples = 0
    
    X_list = []
    for root, dir, files in os.walk(root_path):
        waves = fnmatch.filter(files, "*.wav")
        if len(waves) != 0:
            print ("class: " + root.split("/")[-1])
            X_list.append([
                cached_get_features(os.path.join(root, item), features, hop, bins)
                for item in waves]) 
            for item in waves:
                y_data[samples] = classes
                samples = samples + 1
            classes = classes + 1
    
    X_flat_list = [X_list[class_id][file_id]
                for class_id in range(len(X_list))
                for file_id in range(len(X_list[class_id]))]
                    
    X_data = np.stack(X_flat_list, axis=2)
    X_data = np.transpose(X_data, (2,0,1))
    d1 = X_data.shape[0]
    d2 = X_data.shape[1]
    d3 = X_data.shape[2]
    X_data = np.reshape(X_data, (d1,1,d2,d3))
    
    print ("classes = " + str (classes))
    print ("samples = " + str (samples))

    return X_data, y_data

def create_folds (X_data, y_data, params):    
    cv = StratifiedShuffleSplit(y_data, n_iter=params['nfolds'], test_size=params['split'])
    return cv
        
def standardize (X_train, X_test):
    mu = np.mean (X_train, axis=0)
    de = np.std (X_train, axis=0)
    
    eps = np.finfo('float32').eps
    X_train = (X_train - mu) / (eps + de)
    X_test = (X_test - mu) / (eps + de)
    return X_train, X_test

def svm_classify(X_train, X_test, y_train, y_test, params):
    svm = SVC(C=1.)
    #rsvm = ShapeWrapper(svm)
    X_svm_train = X_train.view ()
    X_svm_train = np.reshape (X_svm_train, (X_svm_train.shape[0], -1))    
    X_svm_test = X_test.view ()
    X_svm_test = np.reshape (X_svm_test, (X_svm_test.shape[0], -1))  
    svm.fit(X_svm_train, y_train)
    y_pred = svm.predict(X_svm_test)
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
    nb_epoch = params["nepoch"]
    img_channels = 1
    img_rows = X_train.shape[2]
    img_cols = X_train.shape[3]
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    model = Sequential()
    
    model.add(Convolution2D(32, 5, 5, border_mode='same',
                             input_shape=(img_channels, img_rows, img_cols)))
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
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    optim = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
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
            width_shift_range=0.4,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.4,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
#    
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            callbacks=[bl],
                            validation_data=(X_test, Y_test))       
    else:
        model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              callbacks=[bl],
              validation_data=(X_test, Y_test),
              shuffle=True)
              
    return bl, model
            
def plot_cnn_results(bl):
    plt.plot (bl.history['acc'])
    plt.plot(bl.history['val_acc'])
    plt.title ('Accuracy (train vs test)')
    plt.show ()
    
    plt.plot (bl.history['loss'])
    plt.plot(bl.history['val_loss'])
    plt.title ('Loss (train vs test)')
    plt.show ()
    
if __name__ == "__main__":
    print ("ESC-50 classification with Keras");
    print ("")    

    print ("computing features...")
    X_data, y_data = compute_features ("../../datasets/ESC-50-master", params)

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
            
        if params['plot'] == True:
            X_flat = np.transpose(X_train, (0, 3, 1, 2))
            X_flat = np.reshape(X_flat, (X_flat.shape[0]*X_flat.shape[1], X_flat.shape[3]))
            X_corr = np.dot (X_flat.T, X_flat)
        
            plt.imshow (X_corr)
            plt.title ('Correlation matrix')
            plt.show ()

        if params["compute_baseline"] == True:
            print ("computing SVM baseline...")
            score, cm = svm_classify(X_train, X_test, y_train, y_test, params)
            print ("score " + str(score))
            
            if params["plot"] == True:
                plt.matshow(cm)
                plt.title ('Confusion matrix')
                plt.show ()

        if params["compute_cnn"] == True:        
            print ("computing CNN classification...")
            bl, model = cnn_classify(X_train, X_test, y_train, y_test, params)
            if params["plot"] == True:
                plot_cnn_results(bl)
            
            print ('max train accuracy ' + str(np.max(bl.history['val_acc'])))

            np.save ('fold_acc'+str(cnt), bl.history['acc'])
            np.save ('fold_val_acc'+str(cnt), bl.history['val_acc'])
            np.save ('fold_loss'+str(cnt), bl.history['loss'])
            np.save ('fold_val_loss'+str(cnt), bl.history['val_loss'])
        
        cnt = cnt + 1
#eof
    