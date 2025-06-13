# Neural Network Regression

![MLP Illustration](./image.png)

A Neural Network Regression model approximates the relationship between inputs and the target by composing multiple layers of linear transformations and nonlinear activations. It captures complex nonlinear patterns and interactions.

### Mathematical Explanation

h₀ = x  
h_l = activation(W_l · h_{l−1} + b_l) for l = 1…L  
y_hat = W_{L+1} · h_L + b_{L+1}  

Parameters W_l and b_l are learned by minimizing the sum of squared errors over the training data using gradient‐based optimization.

---

## Task

We build a Neural Network Regression model to predict diamond prices (`price`) by:  
- Loading the Kaggle “Diamonds Price Dataset”  
- Cleaning data  
- Encoding categorical features (`cut`, `color`, `clarity`)  
- Splitting into train/test sets  
- Scaling numeric features and training `MLPRegressor(hidden_layer_sizes=(100,50), max_iter=300)` on `price`  
- Computing MSE, MAE, R²; visualizing residuals vs predicted; actual vs predicted scatter; buffer accuracy (±\$500)  

---

## Dataset & Features

Diamonds Price Dataset  
- Source: Kaggle (amirhosseinmirzaie/diamonds-price-dataset)  
- File: `diamonds.csv` (~54 000 rows)  

Features used:  
- `carat` (weight in carats)  
- `cut` (quality of cut: Fair → Ideal)  
- `color` (color rating: D → J)  
- `clarity` (clarity rating: I1 → IF)  
- `depth` (total depth percentage)  
- `table` (width of top surface percentage)  
- `x` (length in mm)  
- `y` (width in mm)  
- `z` (height in mm)  

Target:  
- `price` (price in US dollars)  

---

## Libraries  
- pandas — data loading & manipulation  
- numpy — numerical operations  
- scikit-learn — `MLPRegressor`, `StandardScaler`, `pd.get_dummies`, `train_test_split`, evaluation metrics  
- matplotlib — plotting residuals and actual vs predicted scatter  
