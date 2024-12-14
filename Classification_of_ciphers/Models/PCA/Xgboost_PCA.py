# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Standard modules
import pandas as pd
import numpy as np
import sys

# Preprocessing modules
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer  # Import for handling NaN values

# Train-test split module
from sklearn.model_selection import train_test_split

# Dimension reduction module
from sklearn.decomposition import PCA

# Classifier modules
from xgboost import XGBClassifier  # Import XGBoost Classifier

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


file = 'DATASET/DATASET_CIPHER.csv'
dfICU = pd.read_csv(file)

# Drop the 'Plaintext' column
dfICU.drop(['Plaintext','Encryption'], axis=1, inplace=True)

# Create lists of categorical and continuous features
categorical_features = ['Encryption Type', 'Ciphertext']
continuous_features = dfICU.columns[~dfICU.columns.isin(categorical_features)].to_list()

dfICU[categorical_features] = dfICU[categorical_features].astype('category')

dfICU['Encryption Type'].value_counts().plot(kind='barh')

# Remove the target variable 'Encryption Type' from the list of categorical features
categorical_features.remove('Encryption Type')

# Stratified train and test split of the data
X = dfICU.drop('Encryption Type', axis=1)
y = dfICU['Encryption Type']

# Label encoding the target variable y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=1
)
print(f'{X_train.shape[0]} training samples and {X_test.shape[0]} test samples')

# Build pipeline for continuous and categorical features

# Pipeline object for continuous features with NaN imputation
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Replace NaN values with mean
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=43))
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

# Define XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='mlogloss')

# Define the entire classification model
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_classifier)])

# Fit the model on the train data and test on the test data
model.fit(X_train, y_train)

# Predict the output labels for the test set
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

# Load validation data
validation_file = 'DATASET/VALIDATIION_CIPHER.csv'  # Update with your file path
df_validation = pd.read_csv(validation_file)

# Drop the 'Plaintext' column if it exists in validation data
if 'Plaintext' in df_validation.columns:
    df_validation.drop(['Plaintext'], axis=1, inplace=True)

# Separate features and target variable in validation data
X_val = df_validation.drop('Encryption Type', axis=1)
y_val = df_validation['Encryption Type']

# Ensure the data types match
X_val[categorical_features] = X_val[categorical_features].astype('category')

# Label encode the validation target variable
y_val_encoded = label_encoder.transform(y_val)

# Use the trained model pipeline to predict the validation set
y_val_pred = model.predict(X_val)

# Generate confusion matrix
print(confusion_matrix(y_val_encoded, y_val_pred))
print(classification_report(y_val_encoded, y_val_pred))

# Calculate and print validation accuracy
validation_accuracy = accuracy_score(y_val_encoded, y_val_pred)
print(f'Validation Accuracy: {validation_accuracy:.4f}')
