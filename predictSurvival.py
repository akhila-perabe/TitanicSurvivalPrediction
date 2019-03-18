import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

testSetup = True

# Read the training set from the csv file
dataset = pd.read_csv("train.csv")

dataset.Sex.replace(to_replace={'male':0, 'female':1}, inplace=True)
dataset.Embarked.replace(to_replace={'C':0, 'Q':1, 'S':2, None:3 }, inplace=True)
dataset = dataset.fillna(value={'Age':0})

# Get the X and Y for the datset
X = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked']]
Y = dataset.loc[:, ['Survived']]

if testSetup == True:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50, random_state=1234)
else:
    X_train = X
    Y_train = Y    

# Create the model and train
#model = AdaBoostClassifier(algorithm='SAMME', n_estimators=50) #82.37
model = MLPClassifier(hidden_layer_sizes=(100,30,50), activation='relu', alpha=0.00001, random_state=1234, batch_size=100) #80-82
model = model.fit(X_train, Y_train.values.ravel())
print(model.get_params())

if testSetup == False:
    # Load the test data
    testData = pd.read_csv("test.csv")
    testData.Sex.replace(to_replace={'male':0, 'female':1}, inplace=True)
    testData.Embarked.replace(to_replace={'C':0, 'Q':1, 'S':2, None:3 }, inplace=True)
    testData = testData.fillna(value={'Age':0})
    X_test = testData.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked']]

#Predict for test data using the model
predictions = model.predict(X_test)

if testSetup == False:
    file = open("gender_submission.csv", "w")
    file.write("PassengerId,Survived\n")

    # Output the predictions
    idList = testData.loc[:, 'PassengerId']
    i=0
    for y in predictions:
        file.write(str(idList[i]) + "," + str(y) + "\n")
        i += 1
    file.close()
else:
    print ()
    print ("Accuracy:", accuracy_score(Y_test, predictions))
    print ()

