# Assignment 3 - Principal Component Analysis (PCA)
### CS6140 Machine Learning | Northeastern University
**Name:** Sruthilaya

---

## What is this?

This is my third assignment for CS6140. The goal was to apply PCA on the Wine dataset to understand dimensionality reduction — how you can take 13 chemical features of wine and compress them into fewer components while retaining most of the information. The assignment also had us train classifiers on the PCA-transformed data and interpret what the principal components actually represent in terms of wine chemistry.

---

## Dataset

Wine.csv from the CS6140 course repo. 178 wine samples with 13 chemical measurements as features and Customer_Segment (1, 2, 3) as the target class.

- Features include: Alcohol, Malic_Acid, Ash, Flavanoids, Color_Intensity, Proline etc.
- Target: Customer_Segment — 3 wine types
- One thing to note — Proline had a mean of ~742 and std of ~304, way larger scale than other features. Without scaling it would dominate PCA purely due to magnitude, not actual importance.

---

## Tasks Completed

**Q1 - Data Pipeline & Variance Analysis**

Built `load_and_prepare` which loads the data, separates features and target without hard-coding column names (identified target as last column based on dtype and unique values), splits 80/20 and applies StandardScaler fit on training only to avoid data leakage.

`variance_analysis` fits PCA with all 13 components and plots individual and cumulative explained variance. PC1 alone explains 36.88%, PC2 adds 19.32% — so just 2 components capture over 56% of total variance. The curve has an elbow around PC3-PC4, flattening after PC6.

`find_optimal_components` programmatically finds the minimum components needed for a given threshold:
- 90% variance → 8 components
- 95% variance → 10 components
- 99% variance → 12 components

**Q2 - Classification with PCA**

`train_and_evaluate` applies PCA then trains Logistic Regression and reports confusion matrix, accuracy, precision, recall, F1 (macro averaging). With 10 components got 100% accuracy — perfect diagonal confusion matrix.

`components_experiment` loops through 1-13 components and plots accuracy vs cumulative variance on twin y-axes. Most interesting finding — accuracy jumps to 100% at just 3 components even though 95% variance needs 10. Variance explained ≠ classification accuracy.

`plot_decision_boundaries` visualises decision regions using meshgrid and contourf on 2D PCA data. Logistic Regression draws straight line boundaries since it's a linear classifier. Class 1 is clearly separated on the right (high PC1), Classes 2 and 3 overlap more on the left.

**Q3 - PCA Interpretation & Classifier Comparison**

`plot_pca_loadings` creates a heatmap of loadings and a biplot with arrows. PC1 is driven by Flavanoids (0.43), Total_Phenols (0.39), OD280 (0.38) — phenolic compounds related to wine quality. PC2 is driven by Color_Intensity (0.53) and Alcohol (0.50) — wine body and strength. So PC1 separates wines by quality, PC2 by body.

`compare_classifiers` trains Logistic Regression, SVC (RBF), and KNN with 10-fold cross validation:

| Classifier | Test Accuracy | CV Mean | CV Std |
|---|---|---|---|
| Logistic Regression | 100% | 96.5% | 0.035 |
| SVC (RBF) | 100% | 97.9% | 0.032 |
| KNN | 97.2% | 95.8% | 0.072 |

SVC is the best overall — highest CV mean and lowest std. KNN's jagged decision boundaries and high CV std show it's sensitive to local data and inconsistent across folds.

---

## Tools Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (PCA, LogisticRegression, SVC, KNeighborsClassifier, StandardScaler)
- Jupyter Notebook

---

## How to Run

1. Place `Wine.csv` in a `data/` folder inside this directory (or load directly from the course URL)
2. Run all cells top to bottom

---

## Some things I noticed along the way

- The biggest surprise was that 3 PCA components give 100% classification accuracy even though 95% variance needs 10 components. The first 3 PCs already capture the most discriminative information between wine classes — the rest adds variance but not class separation.
- Proline and Magnesium are int64 while most features are float64 — but they're still continuous measurements, not categories, so they're treated as features just like the rest.
- SVC with RBF kernel gave curved boundaries that follow the natural shape of the data much better than Logistic Regression's straight lines, which explains its higher and more consistent CV score.
- KNN's decision boundaries look jagged and irregular because it doesn't learn a boundary — it just votes based on the 5 nearest neighbors locally, which makes it sensitive to noise.
- Wine is a well-separated dataset — even 2 PCs give 97.2% accuracy. Not all datasets will behave this nicely with PCA.

---

## Files Submitted
- `pca.ipynb` - main notebook
- `pca_submission.pdf` - PDF export of notebook with all outputs
