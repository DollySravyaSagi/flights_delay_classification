import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/processed.csv")

from sklearn.model_selection import train_test_split
X = df.drop('ARRIVAL_DELAY', axis=1)
y = (df['ARRIVAL_DELAY'] > 0).astype(int)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#model
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob > 0.42).astype(int)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("output/confusion_matrix.png")
plt.show()

import joblib

joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")