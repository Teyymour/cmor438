# Random Forest Regression

A Random Forest regressor is an ensemble of decision trees that averages the predictions of multiple trees trained on random subsets of the data and features. This approach reduces overfitting and often yields high accuracy on regression tasks like fare prediction.

---

## Task

In this project, I build a Random Forest model to predict the exact airline ticket fare (`Average_Fare`) and measure **buffer‐accuracy** (percentage of predictions within ±\$100 of the true price). The pipeline includes:

1. Loading the Kaggle “Airline Market Fare Prediction” dataset  
2. Selecting numeric features (`MktMilesFlown`, `NonStopMiles`, `RoundTrip`, `Carrier_freq`)  
3. Splitting into train/test sets  
4. Scaling features and training a `RandomForestRegressor`  
5. Computing MSE, MAE, R², and buffer‐accuracy  
6. Visualizing residuals and performance



---

## Dataset & Features

**Airline Market Fare Prediction**  
- Source: Kaggle (`orvile/airline-market-fare-prediction-data`)  
- File: `MarketFarePredictionData.csv` (~316 k rows)

**Features used:**  
- `MktMilesFlown` (market‐miles flown)  
- `NonStopMiles` (miles flown nonstop)  
- `RoundTrip` (1 = round trip, 0 = one way)  
- `Carrier_freq` (frequency of this carrier‐route combo)

**Target:**  
- `Average_Fare` (the ticket price we predict)
---

## Libraries

- **pandas** — data loading & manipulation  
- **numpy** — numerical operations  
- **scikit-learn** — `RandomForestRegressor`, preprocessing, evaluation, `GridSearchCV`  
- **matplotlib** — plotting residuals & true vs. predicted scatter  

---
