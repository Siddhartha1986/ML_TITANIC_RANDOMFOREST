import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib


titanic_df = pd.read_csv("data/titanic_data_ML.csv")   #read the dataset as dataframe

X= titanic_df[["Pclass","gender"]]    #select the features 
y = titanic_df["Survived"]            ##select the target 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,random_state = 42, stratify = y)  ##split the data in trainning and test data


#Lets figure out what best vale for hyperparamters using cross validation

rfc = RandomForestClassifier()    #create an instance for the k neighbour classifier
param_grid = {"n_estimators": [10,50,100,200], "max_depth": [None,5,10,20]}    #define a parameter grid
grid_search = GridSearchCV(rfc, param_grid = param_grid, cv = 5 )    #perfrom the gird search for corss validation
grid_search.fit(X,y)

print("Best values for hyperparameters:",grid_search.best_params_)  # output: Best values for hyperparameters: {'max_depth': , 'n_estimators': }


new_rfc = RandomForestClassifier(n_estimators = grid_search.best_params_['n_estimators'], max_depth = grid_search.best_params_['max_depth'])   #creating a new instance for the model using the predicted hyperparamter
new_rfc.fit(X_train, y_train)        #train the model with the trainning data
y_predict = new_rfc.predict(X_test)  #predict using the x_test data

#calculate the accuracy
accuracy = accuracy_score(y_test, y_predict)
print("The accuracy for the model is:", accuracy)

c_m= confusion_matrix(y_test, y_predict)
print("Confusion _ matrix: \n ",c_m)

joblib.dump(new_rfc,"output_models/new_rfc_model.sav")


