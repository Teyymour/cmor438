import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

# 1) Load data
DATA_PATH = r"C:\Users\jbats\Projects\cmor438\data\uber.csv"
print("Files in data folder:", os.listdir(os.path.dirname(DATA_PATH)))

df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# 2) Feature engineering
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"]    = df["pickup_datetime"].dt.hour
df["weekday"] = df["pickup_datetime"].dt.weekday
df["month"]   = df["pickup_datetime"].dt.month

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points (lat1, lon1) and (lat2, lon2)
    using the Haversine formula. Returns kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    # convert degrees to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    # differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # apply haversine formula
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1_rad) * np.cos(lat2_rad) * (np.sin(dlon / 2) ** 2))
    distance = 2 * R * np.arcsin(np.sqrt(a))

    return distance

df["distance_km"] = haversine(
    df.pickup_latitude, df.pickup_longitude,
    df.dropoff_latitude, df.dropoff_longitude
)

# 2b) Drop invalid rows
df = df[(df["distance_km"]>0) & (df["fare_amount"]>0)]
print("After basic clean:", df.shape)

# 2c) Cap outliers at 99th percentile
cap_dist = df["distance_km"].quantile(0.99)
cap_fare = df["fare_amount"].quantile(0.99)
df["distance_km"] = df["distance_km"].clip(upper=cap_dist)
df["fare_amount"] = df["fare_amount"].clip(upper=cap_fare)

# 2d) Additional features
df["is_weekend"]     = (df["weekday"] >= 5).astype(int)
df["morning_peak"]   = df["hour"].between(7,9).astype(int)
df["dist_x_pass"]    = df["distance_km"] * df["passenger_count"]

# 2e) Log–transform target
df["fare_log"] = np.log1p(df["fare_amount"])

# 3) Features & targets
features = [
    "passenger_count",
    "distance_km",
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "morning_peak",
    "dist_x_pass"
]
X = df[features]
y_log  = df["fare_log"]
y_fare = df["fare_amount"]

# 4) Train/test split (keep both log & original fares)
X_train, X_test, ylog_train, ylog_test, yfare_train, yfare_test = train_test_split(
    X, y_log, y_fare,
    test_size=0.2,
    random_state=42
)
print("Train/Test shapes:", X_train.shape, X_test.shape)

# 5) Pipeline & grid
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), features)
])
pipe = Pipeline([
    ("pre", preprocessor),
    ("knn", KNeighborsRegressor())
])

param_grid = {
    "knn__n_neighbors": [5,10,15,20,25],
    "knn__weights":     ["uniform","distance"],
    "knn__p":           [1,2]
}
search = GridSearchCV(
    pipe, param_grid,
    cv=3, n_jobs=-1, scoring="neg_mean_absolute_error"
)
search.fit(X_train, ylog_train)

print("Best params:", search.best_params_)
print("Best CV MAE (log-scale):", -search.best_score_)

# 6) Evaluate on TEST
best = search.best_estimator_
ylog_pred = best.predict(X_test)
y_pred    = np.expm1(ylog_pred)  # back to dollar-scale

mse = mean_squared_error(yfare_test, y_pred)
mae = mean_absolute_error(yfare_test, y_pred)
r2  = r2_score(yfare_test, y_pred)
within_4 = np.mean(np.abs(y_pred - yfare_test) <= 4)

print(f"\nTest MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R² : {r2:.3f}")
print(f"±$2 buffer accuracy: {within_4:.1%}")
