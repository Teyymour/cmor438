import pandas as pd
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

#1 Reading in dataset
df = pd.read_csv(
    r"C:\Users\jbats\Projects\cmor438\data\uber.csv"
)
df.head()
print(df.head())

#2 Feature Engineering 
#2a Parsing datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"]    = df["pickup_datetime"].dt.hour
df["weekday"] = df["pickup_datetime"].dt.weekday
df["month"]   = df["pickup_datetime"].dt.month

#2b Distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi     = np.radians(lat2 - lat1)
    dlambda     = np.radians(lon2 - lon1)
    a      = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(
    df.pickup_latitude, df.pickup_longitude,
    df.dropoff_latitude, df.dropoff_longitude
)

# drop any bad rows
df = df[df["distance_km"]>0]
df = df[df["fare_amount"]>0].reset_index(drop=True)
print("After cleaning:", df.shape)

# Features & Target
features = [
    "passenger_count",
    "distance_km",
    "hour",
    "weekday",
    "month" ]

X = df[features]
y = df["fare_amount"]

#4 Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train/Test shapes:", X_train.shape, X_test.shape)

# 5) Preprocessing + KNN pipeline
preprocessor = ColumnTransformer([
    ("num",   StandardScaler(), ["passenger_count","distance_km"]),
    ("time",  OneHotEncoder(sparse_output=False), ["hour","weekday","month"])
])

pipe = Pipeline([
    ("pre", preprocessor),
    ("knn", KNeighborsRegressor())
])

# 6) Grid-search k & weighting
param_grid = {
    "knn__n_neighbors": [5, 10, 15, 20, 25],
    "knn__weights":     ["uniform", "distance"]
}
search = GridSearchCV(
    pipe, param_grid,
    cv=3, n_jobs=-1, scoring="neg_mean_absolute_error"
)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best CV MAE:", -search.best_score_)

# 7) Evaluate on test set
best_knn = search.best_estimator_
y_pred   = best_knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
within_2 = np.mean(np.abs(y_pred - y_test) <= 2)

print(f"\nTest MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R² : {r2:.3f}")
print(f"±$2 buffer accuracy: {within_2:.3%}")


