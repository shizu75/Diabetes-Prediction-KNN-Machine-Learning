import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
import warnings
warnings.filterwarnings('ignore')

# ---------- Training Data Preprocessing ----------
train_data = pd.read_csv(r"D:\Internship\Training.csv")
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

for col in cols_to_fix:
    train_data[col] = train_data[col].replace(0, np.nan)
    mean_val = train_data[col].mean(skipna=True)
    train_data[col] = train_data[col].replace(np.nan, mean_val)

X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1].values

# ---------- Testing Data Preprocessing ----------
test_data = pd.read_csv(r"D:\Internship\Testing.csv")

for col in cols_to_fix:
    test_data[col] = test_data[col].replace(0, np.nan)
    mean_val = test_data[col].mean(skipna=True)
    test_data[col] = test_data[col].replace(np.nan, mean_val)

X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1].values

# ---------- Feature Scaling ----------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------- Model Training ----------
knn = KNeighborsClassifier(n_neighbors=17, p=2, metric='euclidean')
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# ---------- Evaluation ----------
print("F1 Score:", f1_score(Y_test, Y_pred))
print("Accuracy:", accuracy_score(Y_test, Y_pred))

# ---------- Histogram Plot ----------
plt.figure()
test_data["Outcome"].hist(bins=15, color='mediumseagreen')
plt.title("Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ---------- Scatter Plot with Prediction Colors ----------
cmap = sns.cubehelix_palette(as_cmap=True)
plt.figure()
plt.scatter(X_test[:, 4], X_test[:, 6], c=Y_pred, s=50, cmap=cmap)
plt.colorbar(label="Predicted Outcome")
plt.xlabel("BMI")
plt.ylabel("Age")
plt.title("Predictions on Test Data (KNN)")
plt.grid(True)
plt.show()

# ---------- Pair Plot ----------
sns.pairplot(data=test_data, hue='Outcome', palette=['red', 'blue', 'limegreen'])
plt.suptitle("Pairwise Feature Comparison", y=1.02)
plt.show()

# ---------- Decision Boundary Plot ----------
# Use only 2 features for visualization
X_vis = test_data[['BMI', 'Age']].values
Y_vis = test_data['Outcome'].values

knn_vis = KNeighborsClassifier(n_neighbors=17, p=2)
knn_vis.fit(X_vis, Y_vis)

h = 0.2
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

pl.figure(figsize=(8, 6))
pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired, shading='auto')
pl.scatter(X_vis[:, 0], X_vis[:, 1], c=Y_vis, edgecolor='k', s=50)
pl.xlabel('BMI')
pl.ylabel('Age')
pl.title('KNN Decision Boundary')
pl.grid(True)
pl.show()
