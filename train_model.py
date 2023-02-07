# Step 1 - import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 2 - load dataset
print("Loading data...")
df = pd.read_csv('cancer_classification.csv')
target_encoding = {0: 'Benign', 1: 'Malignant'}


# Step 3 - split data, training and test sets
print("Spliting dataset now into train and test sets...")
X = df.drop('benign_0__mal_1', 1)
y = df['benign_0__mal_1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4 - training
print('Training Model...')
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5 - feature selection
print("Selecting best features now...")
feats = pd.DataFrame({"Features" : X.columns, "Importance" : model.feature_importances_ * 100})

final_cols = []
imp_level = 6
for i in range(len(X.columns)):
    if feats['Importance'][i] >= imp_level:
        final_cols.append(feats['Features'][i])

print('Done!')
# Step 6 - fine-tune model
print('Now fine-tuning the model...')
model.fit(X_train[final_cols], y_train)

# Step 7 - save model
print('All done, saving the model now.')
__version__ = '0.1.0'
with open(f'breast_cancer_model-{__version__}.pkl', 'wb') as file:
    pickle.dump(model, file)