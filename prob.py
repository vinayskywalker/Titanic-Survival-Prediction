import numpy as np
import sys
import random
import string

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier





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
        dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)
    
    for dataset in data:
        dataset.loc[(dataset.Embarked == 'S'),'Embarked']=0
        dataset.loc[(dataset.Embarked == 'C'),'Embarked']=1
        dataset.loc[(dataset.Embarked == 'Q'),'Embarked']=2
    
    # print(train_data_set[['Parch','Survived']].groupby(['Parch'],as_index=False).mean())


    for dataset in data:
        dataset['FamilySize']=dataset['Parch']+dataset['SibSp']+1
    
    for dataset in data:
        dataset['Alone']=0
        dataset.loc[(dataset['FamilySize']==1,'Alone')]=1
    
    
    divide_agegroup=pd.cut(train_data_set['Age'],5)
    for dataset in data:
        dataset.loc[dataset['Age']<=16,'Age']=0
        dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age']=1
        dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age']=2
        dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64),'Age']=3
        dataset.loc[(dataset['Age']>64),'Age']=4
    
    title_list=['Mrs','Mr','Master','Miss','Major','Rev','Dr','Ms','Mlle','Col','Capt','Mme','Countess','Don','Jonkheer']
    name_train_list=train_data_set['Name'].to_list()
    name_test_list=test_data_set['Name'].to_list()
    i=0
    for i in range(len(name_train_list)):
        for title in title_list:
            if name_train_list[i].find(title)!=-1:
                name_train_list[i]=title
                break
    
    i=0
    for i in range(len(name_test_list)):
        for title in title_list:
            if name_test_list[i].find(title)!=-1:
                name_test_list[i]=title
                break
    
    i=0
    for i in range(len(name_train_list)):
        if name_train_list[i]=='Mme' or name_train_list[i]=='Countess':
            name_train_list[i]='Mrs'
        elif name_train_list[i]=='Ms' or name_train_list[i]=='Mlle':
            name_train_list[i]='Miss'
        elif name_train_list[i]=='Major' or name_train_list[i]=='Rev' or name_train_list[i]=='Col' or name_train_list[i]=='Dr':
            name_train_list[i]='Mr'
        elif name_train_list[i]=='Capt' or name_train_list[i]=='Don' or name_train_list[i]=='Jonkheer':
            name_train_list[i]='Mr'
    # print(train_data_set[['Alone','Survived']].groupby(['Alone'],as_index=False).mean())
    i=0
    for i in range(len(name_test_list)):
        if name_test_list[i]=='Mme' or name_test_list[i]=='Countess':
            name_test_list[i]='Mrs'
        elif name_test_list[i]=='Ms' or name_test_list[i]=='Mlle':
            name_test_list[i]='Miss'
        elif name_test_list[i]=='Major' or name_test_list[i]=='Rev' or name_test_list[i]=='Col' or name_test_list[i]=='Dr':
            name_test_list[i]='Mr'
        elif name_test_list[i]=='Capt' or name_test_list[i]=='Don' or name_test_list[i]=='Jonkheer':
            name_test_list[i]='Mr'
    
    train_data_set['Name']=name_train_list
    test_data_set['Name']=name_test_list

    for dataset in data:
        dataset['Name']=dataset['Name'].map({'Master':0,'Mr':1,'Mrs':2,'Miss':3})
    
    # print(train_data_set['Name'])
    
    for dataset in data:
        del dataset['Cabin']
        del dataset['Ticket']
        del dataset['Parch']
        del dataset['SibSp']
        del dataset['FamilySize']

    
    # print(train_data_set.info())
    

    
    

    Y_train=train_data_set['Survived']
    del train_data_set['Survived']
    X_train=train_data_set
    print(test_data_set.info())
    random_forest=RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train,Y_train)
    Y_prediction=random_forest.predict(test_data_set)
    abc=test_data_set['PassengerId']
    print(Y_prediction.shape)
    dict={'PassengerId':abc,'Survived':Y_prediction}
    df=pd.DataFrame(dict)
    df.to_csv('file.csv')



    

    



    




if __name__ == "__main__":
    main()