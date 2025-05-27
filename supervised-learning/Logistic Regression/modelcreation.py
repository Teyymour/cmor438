import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection     import train_test_split
from sklearn.compose             import ColumnTransformer
from sklearn.preprocessing       import OneHotEncoder, StandardScaler
from imblearn.pipeline           import Pipeline as ImbPipeline
from imblearn.over_sampling      import SMOTE
from sklearn.linear_model        import LogisticRegression
from sklearn.metrics             import classification_report, confusion_matrix

# 1) Load the cleaned dataset
df = pd.read_csv(
    r"C:\Users\jbats\Projects\cmor438\supervised-learning\Logistic Regression\Cleaned_dataset.csv"
)

# 2) Features & target
X = df.drop(columns=["Flight_code", "Class"])
y = df["Class"]

# 3) Column definitions
numeric_cols = ["Duration_in_hours", "Days_left", "Fare"]
categorical_cols = [
    "Journey_day", "Airline", "Source",
    "Departure", "Total_stops", "Arrival", "Destination"
]

# 4) Preprocessor: scale numerics, one‐hot encode cats
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
])

# 5) Build an imbalanced‐learn pipeline:
#    1) preprocess → 2) SMOTE → 3) logistic regression
clf = ImbPipeline([
    ("pre",    preprocessor),
    ("smote",  SMOTE(random_state=42)),
    ("lr",     LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        class_weight="balanced",
        max_iter=1000,
        verbose=1        # you can remove or lower verbosity later
    ))
])

# 6) Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7) Fit & predict (this will preprocess → resample → train internally)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 8) Evaluation
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=clf.classes_, yticklabels=clf.classes_,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
