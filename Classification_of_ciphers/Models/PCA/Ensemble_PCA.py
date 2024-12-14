
# Standard modules
import pandas as pd
import numpy as np
import sys

# Preprocessing modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, \
    OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer  # Import for handling NaN values

# Train-test split module
from sklearn.model_selection import train_test_split

# Dimension reduction module
from sklearn.decomposition import PCA

# Classifier module
from sklearn.ensemble import GradientBoostingClassifier

# Pipeline modules
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Performance metric modules
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Plotting modules
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('dark_background')

plt.rcParams['figure.figsize'] = (4.0, 4.0)  # Set default size of plots
pd.options.display.max_columns = None

# Load CIPHER Data
file = 'DATASET/DATASET_CIPHER.csv'
df_CIPHER = pd.read_csv(file)

# Drop the 'Plaintext' column
df_CIPHER.drop(['Plaintext'], axis=1, inplace=True)

# Create lists of categorical and continuous features
categorical_features = ['Encryption Type', 'Ciphertext']
continuous_features = df_CIPHER.columns[~df_CIPHER.columns.isin(categorical_features)].to_list()

df_CIPHER[categorical_features] = df_CIPHER[categorical_features].astype('category')

df_CIPHER['Encryption Type'].value_counts().plot(kind='barh')

# Remove the target variable 'Encryption Type' from the list of categorical features
categorical_features.remove('Encryption Type')

# Stratified train and test split of the data
X = df_CIPHER.drop('Encryption Type', axis=1)
y = df_CIPHER['Encryption Type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=1
)
print(f'{X_train.shape[0]} training samples and {X_test.shape[0]} test samples')

# Build pipeline for continuous and categorical features

# Pipeline object for continuous features with NaN imputation
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Replace NaN values with mean
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=44))
])

# Pipeline object for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehotenc', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor object for all features
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_transformer, continuous_features),
        ('categorical', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Define a classifier object
classifier = GradientBoostingClassifier()

# Define the entire classification model
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

# Fit the model on the train data and test on the test data
model.fit(X_train, y_train)

# Predict the output labels for the test set
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))


# Load validation data
validation_file = 'DATASET/VALIDATION_CIPHER.csv'  # Update with your file path
df_validation = pd.read_csv(validation_file)

# Drop the 'Plaintext' column if it exists in validation data
if 'Plaintext' in df_validation.columns:
    df_validation.drop(['Plaintext'], axis=1, inplace=True)

# Separate features and target variable in validation data
X_val = df_validation.drop('Encryption Type', axis=1)
y_val = df_validation['Encryption Type']

# Ensure the data types match
X_val[categorical_features] = X_val[categorical_features].astype('category')

# Use the trained model pipeline to predict the validation set
y_val_pred = model.predict(X_val)

# Generate confusion matrix
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

from sklearn.metrics import accuracy_score

# Generate confusion matrix
print(confusion_matrix(y_val, y_val_pred))

# Print classification report
print(classification_report(y_val, y_val_pred))

# Calculate and print validation accuracy
validation_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {validation_accuracy:.4f}')
