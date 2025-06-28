import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


df = pd.read_csv("sample_creditcard_dataset.csv")


scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
X = df.drop(columns=['Time', 'Class'])
y = df['Class']


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


fraud_sample = df[df['Class'] == 1].sample(1, random_state=42).copy()


print("\n🧾 Enter transaction details:")

amount = float(input("Transaction Amount (₹): "))
is_night = input("Was it done at night? (yes/no): ").strip().lower()
rapid_burst = input("Multiple quick transactions in short time? (yes/no): ").strip().lower()
international = input("Is it an international transaction? (yes/no): ").strip().lower()


if is_night == 'yes':
    fraud_sample.iloc[0, 0:5] = -abs(fraud_sample.iloc[0, 0:5]) * 1.5  

if rapid_burst == 'yes':
    fraud_sample.iloc[0, 5:10] = -abs(fraud_sample.iloc[0, 5:10]) * 1.5  

if international == 'yes':
    fraud_sample.iloc[0, 10:15] = -abs(fraud_sample.iloc[0, 10:15]) * 1.5  


scaled_amount = scaler.transform(pd.DataFrame([[amount]], columns=["Amount"]))[0][0]
fraud_sample['Amount'] = scaled_amount


input_data = fraud_sample.drop(columns=['Time', 'Class'])
prediction = model.predict(input_data)[0]
confidence = model.predict_proba(input_data)[0][1]


print("\n Prediction Result:")
if prediction == 1:
    print(f" FRAUD DETECTED! Model confidence: {confidence*100:.2f}%")
else:
    print(f" Transaction is SAFE. Model confidence: {(1 - confidence)*100:.2f}%")
