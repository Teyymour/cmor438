Linear Regression

A Linear Regression model assumes a linear combination of input features to predict a continuous target. It is interpretable and provides a baseline for fare prediction.

Task

We build a Linear Regression model to predict airline ticket fare (`Average_Fare`) by:
- loading the Kaggle “Airline Market Fare Prediction” dataset  
- engineering relevant features (`fare_per_mile`, `distance_log`, `pax_ratio`, binary flags)  
- splitting into train/test sets  
- scaling numeric features and encoding categorical variables  
- training `LinearRegression` on log-transformed fares  
- computing MSE, MAE, R², and buffer-accuracy (±\$100)  
- visualizing residuals and true vs predicted scatter  

Dataset & Features

Airline Market Fare Prediction  
Source: Kaggle (orvile/airline-market-fare-prediction-data)  
File: `MarketFarePredictionData.csv` (~316 k rows)

Features used:  
- `NonStopMiles` (miles flown nonstop)  
- `MktMilesFlown` (market miles flown)  
- `Pax`, `CarrierPax` (total and carrier passengers)  
- `Market_share`, `Market_HHI`, `LCC_Comp` (market metrics)  
- Engineered:  
  - `fare_per_mile` = Average_Fare / NonStopMiles  
  - `distance_log` = log1p(NonStopMiles)  
  - `pax_ratio` = CarrierPax / Pax  
  - `is_roundtrip`, `has_multiple_airports`, `nonstop_flag` (binary flags)  
Target:  
- `fare_log` = log1p(Average_Fare) (back-transform via `expm1`)

Libraries

- pandas — data loading & manipulation  
- numpy — numerical operations  
- scikit-learn — `LinearRegression`, preprocessing, evaluation  
- matplotlib — plotting residuals & true vs predicted scatter

