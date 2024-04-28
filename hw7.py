import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection

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
