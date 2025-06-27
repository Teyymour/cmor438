# CMOR 438 Final Project @ Rice University Spring 2025

---

**By [Teyymour Davoudi](https://github.com/Teyymour), [Patrick Batsell](https://github.com/PatrickBats)**

## Course Description

This final project for **CMOR 438 – Data Science & Machine Learning** applies a variety of machine learning techniques to real-world datasets. Through hands‑on Jupyter notebooks, we demonstrate data exploration, preprocessing, modeling, and evaluation workflows.

---

## Course Instructor

**Dr. Randy R. Davila**
Adjunct Faculty, Department of Computational and Applied Mathematics, Rice University
Email: [rrd6@rice.edu](mailto:rrd6@rice.edu)

---

## Repository Description

The repository is organized into directories for data storage and modeling workflows:

```bash
.
├── data/                        
│   ├── consumer-spending.csv     # Consumer spending metrics over time
│   ├── diamond-prices.csv        # Diamond features and sale prices
│   ├── meteorite-landings.csv    # Meteorite mass, location, and fall year
│   └── uber.csv                  # NYC Uber ride details and fare amounts
│   └── Cleaned_dataset.csv       # Dataset with airplane flight information
├── supervised-learning         
│   
├── unsupervised-learning       
│  
├── README.md
│  
└── Requirements.txt

```

---

## Covered Topics

* **Supervised Learning**: Linear regression, Random Forest regression, Logistic Regression, Perceptron, Neural Networks, Decision Trees, K Means Clustering, Gradient Boosting, K Nearest Neighbors
* **Unsupervised Learning**: K‑Means clustering, PCA, DBSCAN, SVD
* **Exploratory Data Analysis**: Feature engineering, correlation analysis, distribution visualizations
* **Data Handling**: CSV ingestion, preprocessing pipelines

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/Teyymour/cmor438.git
   cd cmor438
   ```
2. **Install dependencies**

   ```bash
   # Using conda
   conda env create -f environments/environment.yml
   conda activate cmor438

   # Or using pip
   pip install -r requirements.txt
   ```

---

## Usage

Launch Jupyter Lab or Notebook and run notebooks in each directory:

```bash
jupyter lab
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

* Professor Randy R. Davila for guidance
* Public datasets from Kaggle and Open Data portals
* CMOR 438 course materials and peer collaboration
