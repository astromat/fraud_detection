import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

# read files
df1 = pd.read_csv('test_identity.csv')
df2 = pd.read_csv('test_transaction.csv')
df3 = pd.read_csv('train_identity.csv')
df4 = pd.read_csv('train_transaction.csv')
df5 = pd.read_csv('sample_submission.csv')

# we want to detect fraud; yes or no => binary target Use Logostic Regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# read data df4
#print(df4.describe())
features = ['TransactionID']+['TransactionDT']+['TransactionAmt']+['card%d' % number for number in range(1,340)]

# Create X (containing the features), and y (containing target variable)
target = 'isFraud'
y = df4[target]
X = df4.drop(columns=[target])
# drop NAN
#X = X.dropna()
#y = y.dropna()

def normalize(X):
    """
    Normalize by subtracting the mean and by dividing by the standard deviation.
    """
    #for feature in X.columns:
    X -= X.mean()
    X /= X.std()
    return X

X_norm = normalize(X[X.columns[2]])
#y = normalize(y)

#print(X[X.columns[0]])
plt.scatter(y,X[X.columns[2]])
plt.show()


#define model
model = LogisticRegression()
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

# Loop through the splits (only one)
for train_indices, test_indices in splitter.split(X, y):
    # Select the train and test data
    #print(train_indices, test_indices)
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    
    # Normalize the data
    #X_train = normalize(X_train)
    #X_test = normalize(X_test)
    
    # Fit and predict!
    model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    
    # And finally: show the results
    print(classification_report(y_test, y_pred))






