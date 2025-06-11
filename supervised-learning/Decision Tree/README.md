# Decision Tree Regression

A Decision Tree Regression model approximates the target by recursively partitioning the feature space and predicting the mean value in each partition. It captures nonlinear relationships and interactions and is interpretable via its tree structure.

### Mathematical Explanation

The model partitions the input space into M disjoint regions R1, R2, ..., RM and predicts a constant value cm for any feature vector x that falls in region Rm.

Predicted value = sum for m = 1 to M of [cm * indicator(x in Rm)].

Each region is defined by binary splits on feature values. cm is the average target of training samples within region Rm. Splits are chosen to minimize the sum of squared errors in the resulting child nodes.

---

## Task

We build a Decision Tree Regression model to predict airline ticket fare (`Average_Fare`) by:
1. Loading the Kaggle “Airline Market Fare Prediction” dataset  
2. Cleaning data  
3. Engineering relevant features (`fare_per_mile`, `distance_log`, `pax_ratio`, binary flags)  
4. Splitting into train/test sets  
5. Encoding categorical variables and training `DecisionTreeRegressor` on log-transformed fares  
6. Computing MSE, MAE, R², visualizing residuals, true vs predicted scatter, feature importances, and optionally tree structure  

---

## Dataset & Features

Airline Market Fare Prediction  
- Source: Kaggle (orvile/airline-market-fare-prediction-data)  
- File: `MarketFarePredictionData.csv` (~316 k rows)  

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
- `fare_log` (log1p of Average_Fare. We log-transform for stability and to reduce right-skew)  

---

## Libraries  
- pandas — data loading & manipulation  
- numpy — numerical operations  
- scikit-learn — `DecisionTreeRegressor`, `OneHotEncoder`, `train_test_split`, evaluation metrics, `plot_tree`  
- matplotlib — plotting residuals, true vs predicted, feature importances

