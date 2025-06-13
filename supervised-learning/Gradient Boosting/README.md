# Gradient Boosting Regression

![Gradient Boosting Illustration](./image.png)

A Gradient Boosting Regression model builds an ensemble of shallow trees sequentially, each one correcting errors of the previous ensemble. It captures complex nonlinear relationships and interactions and tends to outperform single trees by reducing bias and variance.

### Mathematical Explanation

Predicted value:  
y_hat = Σₘ (η · hₘ(x))  

Here hₘ(x) is the m-th decision tree, η is the learning rate, and M is the number of trees. Each hₘ is fit to the residuals (negative gradients) of the loss from the current ensemble, minimizing squared error.

---

## Task

We build a Gradient Boosting model to predict diamond prices (`price`) by:  
1. Loading the Kaggle “Diamonds Price Dataset”  
2. Cleaning data  
3. Engineering `volume = x * y * z`  
4. Encoding categorical features (`cut`, `color`, `clarity`)  
5. Splitting into train/test sets  
6. Training `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)`  
7. Computing MSE, MAE, R²; visualizing residuals vs predicted; actual vs predicted scatter; feature importances  

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
- `depth` (total depth %)  
- `table` (width of top surface %)  
- `x` (length in mm)  
- `y` (width in mm)  
- `z` (height in mm)  
- `volume` (x * y * z)  

Target:  
- `price` (price in US dollars)  

---

## Libraries  
- pandas — data loading & manipulation  
- numpy — numerical operations  
- scikit-learn — `GradientBoostingRegressor`, `OneHotEncoder`/`pd.get_dummies`, `train_test_split`, evaluation metrics  
- matplotlib — plotting residuals, actual vs predicted, feature importances  
