## Flight Class Classification with Logistic Regression & SMOTE

This repository contains a Jupyter notebook demonstrating how to classify flight booking cabin classes (Economy, Business, Premium Economy, First) using a multinomial logistic regression model combined with SMOTE upsampling to address class imbalance. The pipeline covers data loading, feature selection, preprocessing, resampling, model training, evaluation, and visualization—all in one interactive notebook.

---

## Project Structure

```
flight-class-classification/
├── data/
│   └── Cleaned_dataset.csv             # Preprocessed flight-booking dataset
├── notebooks/
│   └── flight_class_classification.ipynb  # Jupyter notebook with full classification workflow
└── README.md                           # Documentation (this file)
```

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/flight-class-classification.git
   cd flight-class-classification
   ```
2. **Install dependencies**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```
3. **Prepare the data**

   * Place `Cleaned_dataset.csv` in the `data/` directory. It should include columns:

     * `Flight_code` (identifier)
     * Numeric: `Duration_in_hours`, `Days_left`, `Fare`
     * Categorical: `Journey_day`, `Airline`, `Source`, `Departure`, `Total_stops`, `Arrival`, `Destination`
     * Target: `Class` (cabin class)

---

## Notebook Workflow

The notebook (`flight_class_classification.ipynb`) is organized into clearly labeled sections:

1. **Imports & Settings**

   * Load libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, Scikit-Learn, Imbalanced‑Learn) and configure plotting style.

2. **Data Loading & Preview**

   * Read `Cleaned_dataset.csv` into a DataFrame.
   * Display first few rows and check the class distribution.

3. **Features & Target**

   * Drop identifier column `Flight_code`.
   * Define the target variable: `Class`.
   * Specify numeric features: `Duration_in_hours`, `Days_left`, `Fare`.
   * Specify categorical features: `Journey_day`, `Airline`, `Source`, `Departure`, `Total_stops`, `Arrival`, `Destination`.

4. **Preprocessing Pipeline**

   * Create a `ColumnTransformer` to:

     * Standard-scale numeric features.
     * One‑hot encode categorical features (dropping the first level to avoid collinearity).

5. **Train/Test Split**

   * Perform an 80/20 stratified split to preserve class proportions.

6. **Model Training & Prediction**

   * Build an `ImbPipeline` combining:

     1. the preprocessing transformer,
     2. SMOTE upsampling,
     3. multinomial logistic regression (`saga` solver, balanced class weights).
   * Fit the pipeline on the training data and generate predictions on the test set.

7. **Evaluation**

   * Print a detailed classification report (precision, recall, F1‑score for each class).
   * Plot a confusion matrix heatmap.

8. **Feature Coefficients**

   * Extract the learned coefficients for each class.
   * Plot the top 10 most positive feature coefficients per cabin class to interpret which attributes drive each prediction.

---

## Next Steps

* Compare against other classifiers (Random Forest, XGBoost) within the same pipeline.
* Package the model into a REST API for real-time class predictions.

