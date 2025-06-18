#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# 1) Load your data
df = pd.read_csv("/Users/pbat/Projects/cmor438/data/diamond-prices.csv")

# 2) Feature engineering
df["volume"] = df["x"] * df["y"] * df["z"]
df["carat_depth_ratio"] = df["carat"] / df["depth"]

# 3) Define features & target
NUM_FEATS = ["carat", "depth", "table", "x", "y", "z", "volume", "carat_depth_ratio"]
ORD_FEATS = ["cut", "color", "clarity"]

X = df[NUM_FEATS + ORD_FEATS]
y = df["price"]

# 4) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5) Specify the ordinals in their true order
ordinal_categories = [
    ["Fair", "Good", "Very Good", "Premium", "Ideal"],        # cut
    ["J", "I", "H", "G", "F", "E", "D"],                       # color
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]   # clarity
]
ord_enc = OrdinalEncoder(
    categories=ordinal_categories,
    dtype=int,
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

# 6) Build preprocessing pipeline with ordinal encoding
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("ord", ord_enc, ORD_FEATS),
])

base_pipe = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(random_state=42))
])

# 7) Wrap in a log‐transforming regressor
ttr = TransformedTargetRegressor(
    regressor=base_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

# 8) Define a small grid and use successive halving to tune
param_grid = {
    "regressor__rf__n_estimators": [50, 100],
    "regressor__rf__max_depth": [10, 20, None],
    "regressor__rf__min_samples_split": [2, 5, 10],
}

search = HalvingGridSearchCV(
    ttr,
    param_grid,
    factor=3,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)

print("🔍 Running HalvingGridSearchCV…")
search.fit(X_train, y_train)
print("✅ Best hyperparameters:", search.best_params_)

# 9) Evaluate on the hold‐out set
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\nFinal performance on test set:")
print(f"  MAE   : ${mae:,.2f}")
print(f"  RMSE  : ${rmse:,.2f}")
print(f"  R²    : {r2:.4f}")
print(f"  MAPE  : {100*mape:.2f}%")
