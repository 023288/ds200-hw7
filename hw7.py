import pandas as pd

# Load the datasets
path = '~/Spaceship titanic/ds200-hw7/'
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')

# Display the first few rows of the training data and summarize missing values
train_data.head(), train_data.isnull().sum(), train_data.info(), test_data.head(), test_data.isnull().sum(), test_data.info()
