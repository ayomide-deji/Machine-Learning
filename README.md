# Machine-Learning

#This is my first machine learning project that i would commit

#TOPIC: CREDIT CARD ELIGIBILITY PREDICTION

URL for the dataset: https://www.kaggle.com/datasets/rohitudageri/credit-card-details
Introduction:

Problem Statement:

The influx of credit card applications in the banking sector comes with a significant operational challenge. The manual assessment of credit card applications is not only prone to errors, but requires a lot of 

time. to streamline this process and enhance efficiency, it is important to integrate machine learning techniques in the banking sector in order to automate credit card approval process. 

Aims/Objectives:

1 Develop an automatic credit card approval system using machine learning algorithms

Methodology:

1 Load the dataset to Jupyter Notebook

2 Split the data into test data and train data

3 Data Preprocessing: Identify a blend of numerical and non-numerical features, diverse value ranges, and missing entries for accurate model predictions,

4 Exploratory data analysis to extract insights,

5 Feature Engineering,

6 Modelling and Hyperparameter Tuning

7 Model Evaluation

#Importing the required libraries for this project

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.compose import make_column_selector as selector

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report

from sklearn.pipeline import Pipeline

#load the data and see what the data looks like

df = pd.read_csv(r"C:\Users\AYOMIDE\Desktop\M505 project\Credit_card.csv")

df.head()

From the initial dataframe, we can see that there is no target variable. In this supervised learning, a target variable is required for our machine model. Hence, we load in our target variable

#Next, load the target data

label = pd.read_csv(r"C:\Users\AYOMIDE\Desktop\M505 project\Credit_card_label.csv")

label

#merge both feature and target data using the LEFT Join

df = df.merge(label, on = "Ind_ID", how = "left")

df

Looking at the credit card dataset, we can see key attributes essential for evaluating creditcard worthiness and making approval decisions.

Typical characteristics in the dataset include: Ind_ID GENDER, Car_Owner, Propert_Owner, CHILDREN, Annual_income, Type_Income, EDUCATION, Marital_status, Housing_type, Birthday_count, Employed_days, 

Mobile_phone, Work_Phone EMAIL_ID, Type_Occupation. Family_Members, and the target column known as label.

The dataset consist of both categorical and Numerical features.

Next, we need to know more about the dataset.Let us explore the dataset.

Before exploration, some columns that would not be required for our training needs to be dropped

Afterwards, we then we label the training variables as X, and the target variable will be labelled y

X = df.drop(['Ind_ID', 'label', 'Birthday_count', 'Work_Phone', 'Phone', 'Family_Members'], axis=1)

X

y = df['label']

Before proceeding with building our machine learning pipeline, There are some essential preprocessing steps also known as feature engineering. These remaining tasks can be categorized into:

1 Splitting the data into training and testing sets.

2 Data Preprocessing on the train data

3 Converting categorical data to numeric data using LabelEncoding.

4 Standardizing the feature values to ensure a consistent scale.

#Splitting data to test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("The X_train consist of", X_train.shape[0], "rows","and", X_train.shape[1], "columns")

print("The y_train consist of", y_train.shape[0], "rows")

DATA PREPROCESSING

#using the .info() command to know the data types of each column 

X_train.info()

#Next, check null values

X_train.isnull().sum()

The train data consist of 422 null values. dropping these null values implies less train data.

Solution:

To resolve this, the categorical null values are replaced with the mode of the categorical column and the numerical null values are replaced with the mean of the numerical column

#replace the Nan values of categorical features by calculating the mode and filling the missing 

#columns with the mode respectively

X_train['Type_Occupation'].fillna(X_train['Type_Occupation'].mode()[0], inplace=True)

X_train['GENDER'].fillna(X_train['GENDER'].mode()[0], inplace=True)

#Afterwards, replace the missing numerical features with the mean of each numerical 

#column respectively

X_train['Annual_income'].fillna(X_train['Annual_income'].mean(), inplace=True)

X_train.isnull().sum()

#Describe the train Dataset. 

X_train.describe()

FEATURE ENGINEERING

 Feature Engineering is an important procedure in developing a machine learning pipeline because by normalizing and scaling, it ensures that all features in the dataset are of similar scale. For this pipeline, 
 
 the ColumnTransformer would be used to perform scaling and Normalization. 

#Scaling and encoding the train data

CT = ColumnTransformer(transformers=[('num', StandardScaler(), selector(dtype_include=['float64', 'int64'])), 
                                    
                                     ('cat', OneHotEncoder(), selector(dtype_include='category'))])

# Now, you can fit and transform your data

x_train_scaled = CT.fit_transform(X_train)

x_test_scaled = CT.fit_transform(X_test)

x_train_scaled

Another crucial aspect is to check for imbalancing

Imba =y_train.value_counts()

sns.barplot(x = Imba.index, y = Imba.values)

plt.show()

The data appears to be highly imbalanced

Solution: Apply SMOTE to balance data

# Apply SMOTE to the training set

from imblearn.over_sampling import SMOTE

X_train_resampled, y_train_resampled = SMOTE().fit_resample(x_train_scaled, y_train)

Imba =y_train_resampled.value_counts()

sns.barplot(x = Imba.index, y = Imba.values)

plt.show()

MODEL BUILDING AND HYPERPARAMETER TUNING

The problem is a classification problem, Hence the DecisionTreeClassifier, RandomForestClassifier and GradientBoosting is used to train the model. For training:

1 We Created a dictionary named models, defining the 3 classification algorithms with their respective parameter grids

2 We performed gridsearchCV by iterating over each model in the dictionary

3 After fitting each gridsearchCV object, we store the best validation score in the best_scores dictionary

4 Finally, we find the model with the hidghest validation score using the max function

# Definining models and their respective parameter grids

models = {

    'DecisionTree': (DecisionTreeClassifier(), {"criterion" : ["gini", "entropy"],
    
                  "splitter": ["best", "random"],
                  
                  "max_depth" : [None, 80, 90, 100, 120, 130, 150],
                  
                  "min_samples_split" : [2, 5, 10],
                  
                  "min_samples_leaf": [1, 2, 4]}),
   
    'RandomForest': (RandomForestClassifier(), {"n_estimators" : [8, 16, 32, 64, 128, 256],
    
                    "max_features" : [1, 2, 3],
                    
                    "bootstrap" : [True, False]}),
   
    'GradientBoosting': (GradientBoostingClassifier(), {"n_estimators": [30, 60, 90, 120, 150],
    
                 "learning_rate": [0.01, 0.1, 0.2, 0.4, 0.6],
                 
                 "max_depth": [1, 3, 5, 7]})
}

# Perform grid search for each model

best_scores = {}

for model_name, (model, param_grid) in models.items():

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_scores[model_name] = grid_search.best_score_

# next, find the best model based on the best_scores

best_model = max(best_scores, key=best_scores.get)

best_score = best_scores[best_model]

print(" The Best Model is :", best_model)

print(" The Validation Score of this model is:", best_score)

EVALUATION OF THE MODEL WITH THE BEST SCORE

#First we get the best model instance based on the selected model name

best_model_instance = models[best_model][0]

#next we Fit the best model on the entire training data

best_model_instance.fit(X_train_resampled, y_train_resampled)


# then we make predictions on the test data

y_pred = best_model_instance.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

con_mat = confusion_matrix(y_test, y_pred)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print("The Accuracy of The Best Model=", accuracy)

print("Precision of The Best Model=", precision)

print("Recall of The Best Model=", recall)

print("F1-Score of The Best Model=", f1)

print("Confusion Matrix of The Best Model:\n", con_mat)

print("Classification Report of The Best Model:\n", cr)

p = sns.heatmap(pd.DataFrame(con_mat), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

CONCLUSION

The pipeline proposed solving a business problem of predicting whether a customer is eligible for a credit card or not

Strength of the proposed solution:

The Strength of the proposed solution lies in its ability to accurately predict credit card eligibility.

Limitations:

The Limitations to this project is in the content of the dataset.

Implication of Result on Business Problem:

It has a positive impact for financial institutions,as it helps them understand factors that contribute to credit card approval and identify customers eligibility for a credit card

Data-Driven Recommendations:

Segment credit-card applicants based on model features

Model Explainabilty:

In this business problem, we not only require a highly explainable model, it is required that the model has a high level of accuracy.The RandomForestClassifier model creates a balance between interpretabilty and 

accuracy. Hence, this model has been accurately been able to predict customers' eligibility for a credit-card

 

