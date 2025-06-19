![PCA Illustration](./PCA.png)

## Principal Component Analysis (PCA)

PCA finds new axes in the data that capture the most variance, and then projects the data onto the top few of these axes to reduce dimensionality while preserving as much information as possible.
---

## Mathematical Explanation

PCA is an unsupervised method that finds orthogonal directions (principal components) capturing maximum variance in the data.

1. **Centering**  
   Given data matrix X ∈ ℝⁿ×ᵈ (rows are samples), subtract the feature means:  
   $$
     \widetilde X = X - \frac{1}{n}\mathbf{1}\mathbf{1}^T X
   $$

2. **Covariance matrix**  
   $$
     \Sigma = \frac{1}{n-1}\,\widetilde X^T \widetilde X
   $$

3. **Eigen-decomposition**  
   Find eigenvalues λᵢ and orthonormal eigenvectors vᵢ of Σ: 
   $$
     \Sigma\,v_i = \lambda_i\,v_i,\quad 
     \lambda_1 \ge \lambda_2 \ge \cdots \ge 0
   $$

4. **Projection**  
   Project the data onto the top k components to get a lower-dimensional representation Y ∈ ℝⁿ×ᵏ:
   $$
     Y = \widetilde X\,[\,v_1,\dots,v_k\,]
   $$

5. **Explained variance ratio**  
   The fraction of total variance captured by component i is  
   $$
     \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}\,.
   $$  
   Choose k so that (∑ᵢ₌₁ᵏ λᵢ) / (∑ⱼ₌₁ᵈ λⱼ) exceeds a desired threshold.

# PCA Analysis of Meteorite Landings

This repository provides tools for performing Principal Component Analysis (PCA) on a meteorite landings dataset. PCA reduces the dimensionality of the data while preserving variance, allowing for both scree plots and three-dimensional visualizations of the top principal components.

## Project Structure

```
PCA/
├── PCAcode.ipynb     # Interactive Jupyter notebook performing the full analysis
├── PCA.png           # Example 3D scatter plot of the first three principal components
└── README.md         # Documentation for setup and usage
```

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/meteorite-pca.git
   cd meteorite-pca/PCA
   ```

2. **Install dependencies**

   ```bash
   pip install -r ../requirements.txt
   ```

   Required packages include `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `openpyxl`.

3. **Prepare the data**

   * Create a `data/` directory at the root of the project.
   * Place your meteorite landings CSV file (e.g., `meteorite_landings.csv`) into `data/`.
   * Ensure the file contains numeric features such as mass, fall date, and geographic coordinates.

## Usage

### Run analysis in Jupyter Notebook

```bash
jupyter notebook PCAcode.ipynb
```

Open the notebook and execute cells interactively to:

1. Load and inspect the input data.
2. Select and clean numeric features.
3. Standardize each feature to zero mean and unit variance.
4. Fit PCA on the standardized data.
5. Generate a scree plot showing explained variance by component.
6. Create a 3D scatter plot of the first three principal components, colored by meteorite class.
7. Save the visualizations (e.g., `PCA.png`).

## Outputs

* **Scree Plot**: A bar chart showing the percentage of total variance explained by each principal component.
* **3D Scatter Plot**: A visualization of samples projected onto the first three principal components, with arrows indicating feature loadings.

## Interpretation

* A sharp decline in explained variance after the first few components indicates that most of the information is captured by those components.
* Clusters or separations in the 3D scatter may reveal grouping by meteorite classification or underlying structure in the data.

---

**Principal Component Analysis**: PCA identifies orthogonal directions in the feature space that maximize variance, facilitating dimensionality reduction and visualization with minimal loss of information.

For more details, refer to the [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
