import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib


data = pd.read_csv('DATASET/DATASET_CIPHER.csv')


X = data.drop(columns=['Encryption Type', 'Plaintext', 'Ciphertext', 'Encryption'])
y = data['Encryption Type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train)


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


random_forest = RandomForestClassifier(random_state=42)
svm = SVC(kernel='linear', probability=True, random_state=42)


ensemble = VotingClassifier(
    estimators=[('random_forest', random_forest), ('svm', svm)],
    voting='soft'
)


ensemble.fit(X_train_scaled, y_train)


y_test_pred = ensemble.predict(X_test_scaled)


print("\nTest Set Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred))


validation_data = pd.read_csv('DATASET/VALIDATION_CIPHER.csv')

X_val = validation_data.drop(columns=['Encryption Type', 'Plaintext', 'Ciphertext'])
y_val = validation_data['Encryption Type']


X_val_scaled = scaler.transform(X_val)


y_val_pred = ensemble.predict(X_val_scaled)


print("\nValidation Set Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))
