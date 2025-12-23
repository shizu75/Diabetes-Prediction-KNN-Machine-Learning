# Diabetes-Prediction-KNN-Machine-Learning

This project implements a complete machine learning pipeline for diabetes prediction using the K-Nearest Neighbors (KNN) algorithm. It focuses on biomedical data preprocessing, feature scaling, model training, evaluation, and visualization using clinical health data.

The project uses two datasets: Training.csv for model training and Testing.csv for evaluation. Clinical features include Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, and Age, with Outcome as the target variable (0 = Non-diabetic, 1 = Diabetic).

Zero values in medical features are treated as missing data and replaced using mean imputation. Feature scaling is applied using StandardScaler, which is essential for distance-based models like KNN.

The KNN classifier is trained with 17 neighbors using the Euclidean distance metric. Model performance is evaluated using Accuracy and F1 Score to assess both correctness and class balance.

Multiple visualizations are included:
- Histogram showing outcome distribution
- Scatter plot of predictions using BMI and Age
- Pairwise feature comparison using pair plots
- Decision boundary visualization using BMI and Age

This project demonstrates practical biomedical machine learning concepts including data cleaning, normalization, classification, evaluation, and interpretability.

Technologies used:
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

How to run:
1. Clone the repository
2. Install dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn
3. Update dataset paths in the script
4. Run the Python file

Applications:
Clinical decision support systems, biomedical data analysis, preventive healthcare analytics, and machine learning education.

Future improvements include hyperparameter tuning, comparison with other classifiers, ROC-AUC analysis, cross-validation, and application to real-world hospital datasets.

  Author: Soban Saeed
GitHub: https://github.com/shizu75  
License: MIT
