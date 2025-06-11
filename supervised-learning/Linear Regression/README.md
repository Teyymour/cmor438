# Linear Regression

A Linear Regression model assumes a linear relationship between input features and the target. It is fast, interpretable, and provides a strong baseline for fare prediction.

Linear regression models the target variable $y$ as a linear combination of input features $x_j$:

\[
\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j x_j.
\]

Here, $\beta_0$ is the intercept and $\beta_j$ are the feature coefficients. The parameters are estimated by minimizing the residual sum of squares across $n$ observations:

\[
\min_{\beta_0,\beta} \sum_{i=1}^{n} \bigl(y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}\bigr)^2.
\]

---

## Task

We build a Linear Regression model to predict airline ticket fare (`Average_Fare`) by:
1. Loading the Kaggle “Airline Market Fare Prediction” dataset  
2. Cleaning data
3. Engineering relevant features (`fare_per_mile`, `distance_log`, `pax_ratio`, binary flags)  
4. Splitting into train/test sets  
5. Scaling numeric features, encoding categorical variables, and training `LinearRegression` on log-transformed fares  
6. Computing MSE, MAE, R², and visualizing residuals & true vs predicted scatter  

---

## Dataset & Features

Airline Market Fare Prediction  
- Source: Kaggle (orvile/airline-market-fare-prediction-data)  
- File: `MarketFarePredictionData.csv` (~316 k rows)  

Features used:  
- `NonStopMiles` (miles flown nonstop)  
- `MktMilesFlown` (market-miles flown)  
- `Pax` (total passengers)  
- `CarrierPax` (carrier’s passengers)  
- `Market_share` (market share of this route)  
- `Market_HHI` (market concentration index)  
- `LCC_Comp` (low-cost carrier competition level)  
- `fare_per_mile` (Average_Fare / NonStopMiles)  
- `distance_log` (log1p of NonStopMiles)  
- `pax_ratio` (CarrierPax / Pax)  
- `is_roundtrip` (1 = round trip, 0 = one way)  
- `has_multiple_airports` (binary flag for multiple airports)  
- `nonstop_flag` (1 = nonstop, 0 = connecting)  

Target:  
- `fare_log` (log1p of Average_Fare. We log transform for stability and to reduce right-skew)  

---

## Libraries  
- pandas — data loading & manipulation  
- numpy — numerical operations  
- scikit-learn — `LinearRegression`, preprocessing, evaluation  
- matplotlib — plotting residuals & true vs predicted scatter
