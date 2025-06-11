# Neural Network Regression

A Neural Network Regression model approximates the relationship between inputs and target by composing multiple layers of linear transformations and nonlinear activations. It captures complex nonlinear patterns and interactions.

### Mathematical Explanation

The model maps input vector x through L layers. At each layer l, the output h_l is:

h_0 = x
h_l = activation( W_l · h_{l-1} + b_l ) for l = 1 to L

The final output y-hat = W_{L+1} · h_L + b_{L+1}.

Parameters W_l and b_l are learned by minimizing the sum of squared errors over training data using gradient-based optimization.

---

## Task

We build a Neural Network Regression model to predict airline ticket fare (`Average_Fare`) by:
1. Loading the Kaggle “Airline Market Fare Prediction” dataset  
2. Cleaning data  
3. Engineering relevant features (`fare_per_mile`, `distance_log`, `pax_ratio`)  
4. Splitting into train/test sets  
5. Scaling numeric features and training `MLPRegressor` on log-transformed fares  
6. Computing MSE, MAE, R², visualizing residuals & true vs predicted scatter, and buffer accuracy (±$20)

---

## Dataset & Features

Airline Market Fare Prediction  
- Source: Kaggle (orvile/airline-market-fare-prediction-data)  
- File: `MarketFarePredictionData.csv` (~316 k rows)  

Features used:  
- `NonStopMiles`  
- `MktMilesFlown`  
- `Pax`  
- `CarrierPax`  
- `Market_share`  
- `Market_HHI`  
- `LCC_Comp`  
- `fare_per_mile`  
- `distance_log`  
- `pax_ratio`  
- `RoundTrip`  
- `Multi_Airport`  
- `Non_Stop`  

Target:  
- `fare_log` (log1p of Average_Fare)

---

## Libraries  
- pandas — data loading & manipulation  
- numpy — numerical operations  
- scikit-learn — `MLPRegressor`, `StandardScaler`, `train_test_split`, evaluation metrics  
- matplotlib — plotting residuals & true vs predicted scatter

