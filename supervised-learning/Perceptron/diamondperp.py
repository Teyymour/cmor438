import pandas as pd
import numpy as np

from sklearn.model_selection    import train_test_split, GridSearchCV
from sklearn.preprocessing     import StandardScaler, PolynomialFeatures, OrdinalEncoder
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import Perceptron
from sklearn.metrics           import classification_report, confusion_matrix

# 1) Load the data
df = pd.read_csv("/Users/pbat/Projects/cmor438/data/diamond-prices.csv")

# 2) Create a binary target: is carat > 0.5?
df['target'] = (df['carat'] > 0.5).astype(int)

# 3) Split off features vs target
X = df.drop(columns=['carat','price','target'])
y = df['target']

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5) Define exact ordering for your ordinals:
cut_order     = ["Fair","Good","Very Good","Premium","Ideal"]
color_order   = ["J","I","H","G","F","E","D"]        # from worst → best
clarity_order = ["I3","I2","I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

# 6) Build preprocessors
numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols     = ["cut","color","clarity"]

numeric_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("poly",  PolynomialFeatures(degree=2, include_bias=False))
])

ordinal_enc = OrdinalEncoder(
    categories=[cut_order, color_order, clarity_order],
    dtype=float
)

preprocessor = ColumnTransformer([
    ("nums", numeric_pipe, numeric_cols),
    ("cats", ordinal_enc, cat_cols),
])

# 7) Wrap into a single pipeline with a Perceptron placeholder
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", Perceptron(random_state=42))
])

# 8) Grid‐search only the Perceptron’s hyper‐parameters
param_grid = {
    "clf__penalty":   [None, "l2", "l1", "elasticnet"],
    "clf__alpha":     [1e-4, 1e-3, 1e-2],
    "clf__eta0":      [1.0, 0.1, 0.01],
    "clf__max_iter":  [500, 1000],
    "clf__tol":       [1e-3, 1e-4]
}

gs = GridSearchCV(
    pipe,
    param_grid   = param_grid,
    cv           = 5,
    scoring      = "accuracy",
    n_jobs       = -1,
    verbose      = 1
)

# 9) Fit the grid search on your training set
gs.fit(X_train, y_train)

print(f"\n→ Best CV accuracy: {gs.best_score_:.4f}")
print("→ Best hyper‐parameters:", gs.best_params_)

# 10) Final evaluation on the held‐out test set
y_pred = gs.predict(X_test)

print("\nTest‐set classification report:")
print(classification_report(y_test, y_pred, target_names=["≤0.5 carat",">0.5 carat"]))

print("Test‐set confusion matrix:")
print(confusion_matrix(y_test, y_pred))
