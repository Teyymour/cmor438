# Linear Regression

![Linear Regression Illustration](./image.png)

A Linear Regression model assumes a linear relationship between input features and the target. It is fast, interpretable, and provides a strong baseline for diamond price prediction.

Predicted value:  
y_hat = β0 + β1*x1 + β2*x2 + … + βp*xp

Coefficients β0…βp are chosen to minimize the sum of squared errors over all n samples:  
minimize ∑(yi – (β0 + β1*xi1 + … + βp*xip))²

---

## Task

We build a Linear Regression model to predict diamond prices (price) by:  
1. Loading the Kaggle “Diamonds Price Dataset”  
2. Cleaning data  
3. Encoding categorical features (cut, color, clarity)  
4. Splitting into train/test sets  
5. Scaling numeric features and fitting LinearRegression on price  
6. Computing MSE, MAE, R²; visualizing residuals vs predicted and actual vs predicted  

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
- scikit-learn — `LinearRegression`, `StandardScaler`, `OneHotEncoder`/`pd.get_dummies`, `train_test_split`, evaluation metrics  
- matplotlib — plotting residuals and actual vs predicted scatter  
