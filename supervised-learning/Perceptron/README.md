# Perceptron

The perceptron is one of the simplest types of artificial neural networks, modeling a single biological neuron. It takes in \(n\) input features, each multiplied by a corresponding weight, sums them up (plus a bias), and passes the result through a hard-limit activation function. If the weighted sum is above zero, the perceptron outputs 1; otherwise it outputs 0. Despite its simplicity, it forms the building block for more complex neural networks and serves as a linear classifier for binary problems.

---

## Task

In this project, I build a perceptron model to classify whether an airline ticket fare is **above \$1 000** or **below \$1 000**. The pipeline includes:

1. Loading the Kaggle “Airline Market Fare Prediction” dataset  
2. Engineering a binary label \(`fare > 1000`\)  
3. Selecting flight-related features (`MktMilesFlown`, `NonStopMiles`, `RoundTrip`, `Carrier_freq`)  
4. Splitting into train/test sets  
5. Standardizing features and training a balanced perceptron  
6. Evaluating accuracy and precision/recall via a classification report  

---

## Dataset

**Airline Market Fare Prediction**  
- Source: Kaggle (`orvile/airline-market-fare-prediction-data`)  
- File used: `MarketFarePredictionData.csv`  
- Size: ~370 MB, ~316 k rows  
- Key columns:  
  - `Average_Fare` (target)  
  - `MktMilesFlown`, `NonStopMiles`, `RoundTrip`, `Carrier_freq` (features)  

---

## Libraries

- **pandas** — data loading & manipulation  
- **numpy** — numerical operations  
- **scikit-learn** — model (`Perceptron`), preprocessing, evaluation  
- **matplotlib** / **seaborn** (optional) — plotting results (confusion matrix, decision boundary)  

---
