# Machine-Learning

#This is my first machine learning project that i would commit

#TOPIC: CREDIT CARD ELIGIBILITY PREDICTION

URL for the dataset: https://www.kaggle.com/datasets/rohitudageri/credit-card-details

#In commercial banking sector, the influx of credit card applications presents a significant operational challenge. The manual assessment of these applications is

#not only laborious and prone to errors but also consumes valuable time.

#Factors such as Type of income, income levels, and multiple inquiries on credit reports often lead to the rejection of many applications. To streamline this 

#process and enhance efficiency, the integration of machine learning techniques has become increasingly relevant in the banking sector. By automating the credit

#card approval process, banks can expedite decision-making while ensuring accuracy and consistency.

#This project aims to develop an automatic credit card approval system using machine learning algorithms.

#The project utilizes the Credit Card Approval dataset sourced from Kaggle, which is a repository for data. The workflow comprises:

#Loading the dataset to Jupyter Notebook,

#Splitting the data into test data and train data,

#Identifying a blend of numerical and non-numerical features, diverse value ranges, and missing entries requiring preprocessing for accurate model predictions,

#Performing exploratory data analysis to extract insights,

#Feature Engineering,

#Modelling and Hyperparameter Tuning

#Creating a machine learning pipeline to predict credit card approval status based on individual applications.

#Importing the required libraries for this project

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report

#load the data and see what the data looks like

df = pd.read_csv(r"C:\Users\AYOMIDE\Desktop\M505 project\Credit_card.csv")

df.head()

from sklearn.pipeline import Pipeline

