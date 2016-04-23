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
from sklearn.cross_validation import KFold

SAMPLELEN=110272
compute = True

params = {'ncoeff': 90, 'fft': 2048, 'hop': 2048, \
          'nclasses': 50, 'nsamples':2000, \
          'nfolds': 3, 'split':.25}

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
    
def create_folds (kf):    
    for train_index, test_index in kf:
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

    return X_train, y_train, X_test, y_test
    
def svm_classify_data (X_data, y_data, params):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    svm = SVC(C=1., kernel="linear")
    
    pipeline = make_pipeline(StandardScaler(), svm)
    
    from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
    
    y = y_data
    X = X_data.view()
    X.shape = X.shape[0], -1

    cv = StratifiedShuffleSplit(y, n_iter=params['nfolds'], test_size=params['split'])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return scores
    
def compute_measures (confmat):
    acc = 0
    aoc = 0
    fmeasure = 0 
    
    return acc, aoc, fmeasure
    
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
    
    print ("computing SVM basline...")
    scores = svm_classify_data(X_data, y_data, params)
    print ("scores: " + str(scores))
    
    X_train, y_train, X_test, y_test = create_folds(cv)
    


    