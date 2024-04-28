import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load the datasets
path = '~/Spaceship titanic/ds200-hw7/'
df_labeled = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')

# Split into train and validation sets
df_train, df_val = sklearn.model_selection.train_test_split(df_labeled, test_size=0.2, random_state=42)

# Specify columns
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']

# Impute missing values
impute_vals = {
    'Age': df_train['Age'].median(),
    'RoomService': df_train['RoomService'].median(),
    'FoodCourt': df_train['FoodCourt'].median(),
    'ShoppingMall': df_train['ShoppingMall'].median(),
    'Spa': df_train['Spa'].median(),
    'VRDeck': df_train['VRDeck'].median()
}

df_train.fillna(impute_vals, inplace=True)
df_val.fillna(impute_vals, inplace=True)
