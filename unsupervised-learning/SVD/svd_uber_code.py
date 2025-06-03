#!/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load the CSV data
df = pd.read_csv('/Users/pbat/cmor438/data/uber.csv')

print(df.head())

# 3) Convert pickup_datetime to a pandas datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# 4) Create an 'hour' column from that datetime
df['hour'] = df['pickup_datetime'].dt.hour

# 5) Choose numeric features and drop any rows with missing values
features = ['fare_amount',
            'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'hour']

X = df[features].dropna()
print("\nNumber of rows after dropna():", X.shape[0])
print("Number of features (columns) used:", X.shape[1])


# 6) Center the data by subtracting each column’s mean
X_centered = X - X.mean()

# 7) Convert the centered DataFrame to a NumPy array
X_mat = X_centered.values

# 8) Run full SVD (U, S, Vt)
U, S, Vt = np.linalg.svd(X_mat, full_matrices=False)

# 9) Print shapes of U, S, Vt and the top singular values
print('Shape of U: ', U.shape)
print('Shape of S: ', S.shape)
print('Shape of Vt:', Vt.shape)
print('Top 10 singular values:', S[:10])

X_proj = X_mat.dot(Vt.T[:, :2])

# 10) Project centered data onto the first two right singular vectors
X_proj = X_mat.dot(Vt.T[:, :2])

print('\nFirst 5 rows of the 2D projection:')
print(X_proj[:5])

print('Top 10 singular values:', S[:10])

# 10) Project centered data onto the first two right singular vectors
X_proj = X_mat.dot(Vt.T[:, :2])

print('\nFirst 5 rows of the 2D projection:')
print(X_proj[:5])

# 11) Plot the first two principal component scores
plt.figure(figsize=(6,6))
plt.scatter(X_proj[:,0], X_proj[:,1], s=1, alpha=0.5)
plt.title('Uber data: 2D projection onto top 2 PCs')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# 12) Find the top 3 extreme values along PC1
#    (these are the row‐indices of the original X after dropna())
idx_sorted_by_pc1 = np.argsort(np.abs(X_proj[:, 0]))[::-1]
top3_idx = idx_sorted_by_pc1[:3]

print("\nTop 3 trips by |PC1| (original row indices in X):", top3_idx)

# Look up their feature values in X (before centering):
print("\nRaw feature values for those trips:")
print(X.iloc[top3_idx])




