from SupervisedPCA import SupervisedPCARegressor
from SupervisedPCA import SupervisedPCAClassifier
from sklearn import datasets
import numpy as np

diabetes = datasets.load_iris()
X = diabetes.data
Y = diabetes.target

spca = SupervisedPCAClassifier(threshold=1.7)
spca.fit(X, Y)
print(spca._model.coef_)
