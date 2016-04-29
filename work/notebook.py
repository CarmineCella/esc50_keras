import numpy as np

X_data = np.load ("X_data.npy")
y_data = np.load ("y_data.npy")

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator

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


Xtrain, Xtest, ytrain, ytest = train_test_split(X_data, y_data)

#Xtrain.shape = Xtrain.shape[0], -1
#Xtrain = Xtrain.reshape(X_train.shape[0], -1)


svm = SVC(kernel="linear", C=1)

rsvm = ShapeWrapper(svm)

rsvm.fit(Xtrain, ytrain)

ypred = rsvm.predict(Xtest)

cm = confusion_matrix(ytest, ypred)


