# Titanic-Survival-Prediction

Training dataset contains 11 features and one target features "Survival"

## feature Extraction

There are 12 column in train dataset which is
PassengerId,Survived,Pclass,Name,Sex,Age,Sibsp,Parch,Ticket
Fare,Cabin,Embarked

There are lot of missing value in train as well as test file
cabin column contains lot of missing values so i deleted that column.

Embarked column contains only two missing values so fill them with the most frequent values.
To fill the missing value in Age column, just take the mean and standard deviation of train as well as test data set
and replace null values with random number between mean-std and mean+std

To make all the values either to float or int, map male to 1 and female to 0 in sex column.
replace all values in Embarked column with their ASCII values and replace missing values in fare with mean of fare column values.

Delete Cabin,ticket,Name column from test data set from dataset.
Using Random Forest predict the Survival for the test data set

Got 74.162% Accuracy

After Creating New features Alone and mapping 'Embarked' Features to 0,1,2 gives accuracy of 72.21.It may be because i deleted three features 'Cabin','Name','Ticket'.
from the data,age group between 0-16 have higher survival rate so divide the age group in five different parts.
Now dividing group in five parts and mapping with 0-4,delete the some features like Parch,SibSp and FamilySize gives the Accuracy of 74.162%.
