import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('dark_background')
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from openTSNE import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

# Load data
FILE = 'DATASET/DATASET_CIPHER.csv'
csv = pd.read_csv(FILE)
dfCIPHER =csv

# Drop unnecessary columns
dfCIPHER.drop(columns=['Plaintext', 'Ciphertext', 'Encryption'], errors='ignore', inplace=True)

# Handle NaN values by replacing them with the mean of each column for continuous features
continuous_features = dfCIPHER.columns[~dfCIPHER.columns.isin(['Encryption Type'])].to_list()

# Replacing NaNs with mean values for continuous features
dfCIPHER[continuous_features] = dfCIPHER[continuous_features].fillna(dfCIPHER[continuous_features].mean())
print(f"Total NaN values replaced with mean in training data.")

# Define feature lists
categorical_features = ['Encryption Type']
continuous_features = dfCIPHER.columns[~dfCIPHER.columns.isin(categorical_features)].to_list()
dfCIPHER[categorical_features] = dfCIPHER[categorical_features].astype('category')
categorical_features.remove('Encryption Type')

# Stratified train-test split
X = dfCIPHER.drop('Encryption Type', axis=1)
y = dfCIPHER['Encryption Type'].astype('category')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
le = LabelEncoder()
y_train = le.fit_transform(y_train)

print(f'{X_train.shape[0]} training samples and {X_test.shape[0]} test samples')

# Custom transformer for TSNE
class TSNETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter=500, perplexity=40, initialization="random", metric="cosine", n_jobs=8, random_state=42, verbose=True):
        self.n_iter = n_iter
        self.perplexity = perplexity
        self.initialization = initialization
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        self.tsne = TSNE(n_iter=self.n_iter, perplexity=self.perplexity, initialization=self.initialization,
                         metric=self.metric, n_jobs=self.n_jobs, random_state=self.random_state, verbose=self.verbose)
        self.tsne_model = self.tsne.fit(X)
        return self

    def transform(self, X, y=None):
        return self.tsne_model.transform(X)

# Continuous features pipeline with random initialization
continuous_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('tsne', TSNETransformer(n_iter=500, perplexity=40, initialization="random", metric="cosine", n_jobs=8, random_state=42, verbose=True))
])

# Categorical features pipeline
categorical_transformer = Pipeline(steps=[
    ('onehotenc', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_transformer, continuous_features),
        ('categorical', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Classifier (SVM with RBF kernel and class weights for imbalanced classes)
classifier = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Fit model
model.fit(X_train, y_train)

# Predict test set
y_pred = model.predict(X_test)
y_pred = le.inverse_transform(y_pred)
print(confusion_matrix(y_test, y_pred))

# t-SNE transform test data for plotting
X_test_transformed = preprocessor.transform(X_test)

# Save model
joblib.dump(model, '../svm_tsne_model.joblib')

# Check presence of each category in `y_test`
count_no = sum(y_test == 'No')
count_yes = sum(y_test == 'Yes')
if count_yes > 0:
    ratio = count_no / count_yes
    print(f"Ratio (No/Yes): {ratio}")
else:
    print("No instances of 'Yes' in y_test")

# Plot transformed test samples
fig, ax = plt.subplots(1, 1)
categories = np.unique(y_test)
colors = ['red', 'blue', 'yellow', 'green', 'orange', 'purple']
for i, category in enumerate(categories):
    idx = y_test == category
    color = colors[i % len(colors)]
    ax.scatter(X_test_transformed[idx, 0], X_test_transformed[idx, 1], label=f'Category: {category}', c=color, s=3)
ax.set_xlabel('T-SNE Component-1')
ax.set_ylabel('T-SNE Component-2')
ax.set_title('T-SNE Transformed Test Data')
ax.legend()
plt.show()

# Load validation data
validation_file = 'DATASET/VALIDATION_CIPHER.csv'
df_val = pd.read_csv(validation_file)
df_val = df_val[df_val['Encryption Type'] != 'Rail Fence']  # Exclude 'Rail Fence' from validation data as well
df_val.drop(columns=['Plaintext', 'Ciphertext', 'Encryption'], errors='ignore', inplace=True)

# Replacing NaNs with mean values for validation set continuous features
df_val[continuous_features] = df_val[continuous_features].fillna(df_val[continuous_features].mean())
print(f"Total NaN values replaced with mean in validation data.")

X_val = df_val.drop('Encryption Type', axis=1)
y_val = df_val['Encryption Type'].astype('category')

# Transform and predict on validation set
X_val_transformed = model.named_steps['preprocessor'].transform(X_val)
y_val_encoded = le.transform(y_val)
y_val_pred = model.named_steps['classifier'].predict(X_val_transformed)
y_val_pred_labels = le.inverse_transform(y_val_pred)

# Validation results
print("Validation Set Results:")
print(confusion_matrix(y_val, y_val_pred_labels))
print(classification_report(y_val, y_val_pred_labels))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_labels))

# Display label encoding
labels = le.classes_
print("Label Encoding:")
for index, label in enumerate(labels):
    print(f"{index}: {label}")

# Confusion matrix with labels
conf_matrix_val = confusion_matrix(y_val, y_val_pred_labels)
conf_matrix_df = pd.DataFrame(conf_matrix_val, index=labels, columns=labels)
print("\nValidation Confusion Matrix with Label Names:")
print(conf_matrix_df)
