# Gradient Boosting Regression

A Gradient Boosting Regression model builds an ensemble of decision trees in a stage-wise fashion. Each new tree fits the residuals (errors) of the previous ensemble to reduce the overall error.

### Mathematical Explanation

Start with a constant model F0(x) equal to the average of targets. For m from 1 to M:
- Compute residuals rn,i = yi - F_{m-1}(xi) for all training samples.
- Fit a regression tree to predict these residuals.
- Update the model: Fm(x) = F_{m-1}(x) + learning_rate * tree_m(x).

The objective is to minimize the sum of squared errors. Learning rate shrinks contributions of each tree for better generalization.

---

## Task

We build a Gradient Boosting Regression model to predict airline ticket fare (`Average_Fare`) by:
1. Loading the Kaggle “Airline Market Fare Prediction” dataset  
2. Cleaning data  
3. Engineering relevant features (`fare_per_mile`, `distance_log`, `pax_ratio`)  
4. Splitting into train/test sets  
5. Training `GradientBoostingRegressor` on log-transformed fares  
6. Computing MSE, MAE, R², buffer accuracy (±$20), residual plots, true vs predicted scatter, and feature importances

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
- scikit-learn — `GradientBoostingRegressor`, `train_test_split`, evaluation metrics  
- matplotlib — plotting residuals & true vs predicted, feature importances

