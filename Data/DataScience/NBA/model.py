# %%
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import seaborn as sns
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    precision = 0
    fbeta = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        precision += precision_score(test_labels, predicted_labels)
        fbeta += fbeta_score(test_labels, predicted_labels, average='binary', beta=0.5)
    recall/=3
    precision/=3
    fbeta/=3
    # print("Conf matrix=\n",confusion_mat)
    # print("Recall=",recall)
    # print("Precision=",precision)
    # print("fbeta_score=",fbeta)
    return classifier,confusion_mat


def norm_standard(X, test=False):
    """
    Normalize and standardize the data. 
    If test = False, we fit and transform the train data
    if test = True, we fit the with the train data but we transform the test data
    return: the transformed data
    """
    
    minmax = MinMaxScaler()
    standard = StandardScaler()
    if test:
        df_vals = np.load('df_vals.npy')
        minmax.fit(df_vals)
        df_vals = minmax.transform(df_vals)
        standard.fit(df_vals)
        X = minmax.transform(X)
        X = standard.transform(X)
    else:
        X=minmax.fit_transform(X)
        X=standard.fit_transform(X)
    return X

def train():
    """
    Train the model after preprocessing the data and then save it
    return: the model
    """

    # Load dataset
    df = pd.read_csv("./nba_logreg.csv")

    # replacing Nan values (only present when no 3 points attempts have been performed by a player)
    df["3P%"]=df["3P%"].fillna(0.0)

    # Those are the conflicted names because they appear more than one time
    conflict_names=df.groupby("Name").count()["GP"][df.groupby("Name").count()["GP"]>1].index

    # Now we see if they have the same features and the same target
    supp_lines = 0
    for name in conflict_names:
        group = df[df["Name"]==name].groupby(df.columns[:-1].tolist()).mean()
        if(any([target not in [1.0, 0.0] for target in group["TARGET_5Yrs"].tolist()])):
            idx = df[df["Name"]==name].index.tolist()
            df.drop(idx,axis=0,inplace=True)
            supp_lines += len(idx)

    # Using the Inter Quantile Range method (IQR) to retrieve the outliers
    Q1 = df.loc[:,"GP":"TOV"].quantile(0.25)
    Q3 = df.loc[:,"GP":"TOV"].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR

    for col in df.columns[1:-1].tolist():
        lower_points= (df[col]<lower_bound[col]).sum()
        upper_points= (df[col]>upper_bound[col]).sum()

    for col in df.columns[1:-1].tolist():
        df[col][df[col]<lower_bound[col]]=lower_bound[col]
        df[col][df[col]>upper_bound[col]]=upper_bound[col]

    # extract names, labels, features names and values
    names = df['Name'].values.tolist() # players names
    labels = df['TARGET_5Yrs'].values # labels
    paramset = df.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
    df_vals = df.drop(['TARGET_5Yrs','Name'],axis=1).values
    np.save('df_vals.npy', df_vals) 

    # normalize and standardize dataset
    X = norm_standard(df_vals, test=False)
    
    # Oversample the minority class
    ros = RandomOverSampler(random_state=42)
    X, labels = ros.fit_resample(X, labels)

    # # TODO build a training set and choose a classifier which maximize recall score returned by the score_classifier function
    clf,cm=score_classifier(X,RandomForestClassifier(),labels)

    # Save the model
    joblib.dump(clf, Path(BASE_DIR).joinpath("classifier.joblib"))

    return clf

def prediction(X):
    """
    Prediction, load the model if it exists, otherwise train it and then
    make a single player prediction.
    X : [GP MIN PTS FGM FGA FG%  3P Made 3PA 3P%  FTM FTA FT%  OREB DREB REB AST STL BLK TOV] values
    return: label prediction
    """

    model_file = Path(BASE_DIR).joinpath("classifier.joblib")
    if not model_file.exists():
        clf = train()
    else:
        # Load model
        clf = joblib.load(model_file)

    X = norm_standard([X], test=True)
    
    return clf.predict(X)

