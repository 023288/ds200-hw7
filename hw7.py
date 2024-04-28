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

# Encoding of categorical variables
for col in ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']:
    unique_vals = df_train[col].unique()
    mapping = {k: v for v, k in enumerate(unique_vals)}
    df_train[col] = df_train[col].map(mapping)
    df_val[col] = df_val[col].map(mapping).fillna(-1)  # Use -1 for unseen values in validation set

# Prepare feature matrices and target vectors
features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
X_train = df_train[features]
X_val = df_val[features]
y_train = df_train['Transported']
y_val = df_val['Transported']

# Model 1: Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_val)
acc_val_tree = np.mean(y_pred_tree == y_val)
print(f'Decision Tree Validation Accuracy: {acc_val_tree}')

# Model 2: Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_val)
acc_val_rf = np.mean(y_pred_rf == y_val)
print(f'Random Forest Validation Accuracy: {acc_val_rf}')

# Model 3: XGBoost
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_val)
acc_val_xgb = np.mean(y_pred_xgb == y_val)
print(f'XGBoost Validation Accuracy: {acc_val_xgb}')

# Impute missing values and also Apply the manual encoding 
# for the test set using the same values from the training set
df_test.fillna(impute_vals, inplace=True)
for col in ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']:
    df_test[col] = df_test[col].map(mapping).fillna(-1)  # Use -1 for unseen values in the test set

X_test = df_test[features]

# Making predictions with the best model
test_predictions = xgb_clf.predict(X_test)

# Create submission
submission_df = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Transported': test_predictions
})

submission_df['Transported'] = submission_df['Transported'].astype(bool).map({True: 'True', False: 'False'}) # Convert predictions from boolean to string as required
submission_path = path + 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f'Submission file saved to: {submission_path}')
