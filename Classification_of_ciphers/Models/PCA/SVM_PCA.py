# Standard modules
import pandas as pd
import numpy as np
import sys

# Preprocessing modules
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (4.0, 4.0)
pd.options.display.max_columns = None


file = 'DATASET/DATASET_CIPHER.csv'
dfICU = pd.read_csv(file)

# Drop unnecessary columns
dfICU.drop(['Plaintext','Encryption'], axis=1, inplace=True)

# Define feature lists
categorical_features = ['Encryption Type', 'Ciphertext']
continuous_features = dfICU.columns[~dfICU.columns.isin(categorical_features)].to_list()
dfICU[categorical_features] = dfICU[categorical_features].astype('category')

# Remove target variable from categorical features
categorical_features.remove('Encryption Type')

# Train-test split
X = dfICU.drop('Encryption Type', axis=1)
y = dfICU['Encryption Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

# Compute class weights for Columnar Transposition and Double Columnar Transposition
unique_classes = y_train.unique()
class_weight_dict = {cls: 1.0 for cls in unique_classes}
class_weight_dict['Double Columnar Transposition'] = 10.0
class_weight_dict['Columnar Transposition'] = 10.0


# Print class weights to confirm
print("Class weights:", class_weight_dict)

# Create pipelines for continuous and categorical features
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=43))
])
categorical_transformer = Pipeline(steps=[
    ('onehotenc', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_transformer, continuous_features),
        ('categorical', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Define SVM classifier with custom class weights
classifier = SVC(kernel='linear', class_weight=class_weight_dict, random_state=1)

# Create model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

# Train the model
model.fit(X_train, y_train)

# Test predictions
y_pred = model.predict(X_test)
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred))
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred))

# Load validation data
validation_file = 'DATASET/VALIDATION_CIPHER.csv'
df_validation = pd.read_csv(validation_file)
if 'Plaintext' in df_validation.columns:
    df_validation.drop(['Plaintext'], axis=1, inplace=True)

# Separate validation features and target
X_val = df_validation.drop('Encryption Type', axis=1)
y_val = df_validation['Encryption Type']
X_val[categorical_features] = X_val[categorical_features].astype('category')

# Predict on validation data
y_val_pred = model.predict(X_val)
print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))
print("Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred))

# Validation accuracy
validation_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {validation_accuracy:.4f}')
