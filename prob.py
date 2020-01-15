import numpy as np
import sys
import random

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB



def get_test_data():
    test_file=pd.read_csv("data/test.csv")
    return test_file

def get_train_data():
    train_file=pd.read_csv("data/train.csv")
    return train_file


def main():
    train_data_set=get_train_data()
    test_data_set=get_test_data()
    data=[train_data_set,test_data_set]

    # map the male value to 1 and female value to 0 in train and test data set
    for dataset in data:
        dataset['Sex']=dataset['Sex'].map({'female':0,'male':1})
    
    # fill missing values of age column to random values between mean-std and mean+std
    for dataset in data:
        mean=dataset['Age'].mean()
        std=dataset['Age'].std()
        rand_age=random.randint(int(mean-std),int(mean+std))
        dataset['Age'].fillna(rand_age,inplace=True)
    
    for dataset in data:
        dataset['Embarked'].fillna('S',inplace=True)
        dataset['Fare'].fillna(dataset['Fare'].mean(),inplace=True)
    
    for dataset in data:
        dataset.loc[(dataset.Embarked == 'S'),'Embarked']=ord('S')
        dataset.loc[(dataset.Embarked == 'C'),'Embarked']=ord('C')
        dataset.loc[(dataset.Embarked == 'Q'),'Embarked']=ord('Q')

    
    for dataset in data:
        del dataset['Cabin']
        del dataset['Ticket']
        del dataset['Name']
    

    Y_train=train_data_set['Survived']
    del train_data_set['Survived']
    X_train=train_data_set
    print(test_data_set.info())
    random_forest=RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train,Y_train)
    Y_prediction=random_forest.predict(test_data_set)
    abc=test_data_set['PassengerId']
    print(Y_prediction.shape)
    dict={'x':abc,'y':Y_prediction}
    df=pd.DataFrame(dict)
    df.to_csv('file.csv')
    

    



    




if __name__ == "__main__":
    main()