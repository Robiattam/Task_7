import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("C:\\project_py\\breast-cancer.csv")
X = df.drop(columns="diagnosis")
Y = df["diagnosis"]

# Encode labels ('M' and 'B') to numeric (1 and 0)
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y_encoded, test_size=0.2, random_state=42)

# Train SVM models
svm_linear = SVC(kernel='linear', C=1)
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()

# Plot decision boundaries
plot_decision_boundary(svm_linear, X_test, y_test, "SVM with Linear Kernel")
plot_decision_boundary(svm_rbf, X_test, y_test, "SVM with RBF Kernel")

# GridSearchCV for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1]
}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters from GridSearchCV:", grid.best_params_)

# Cross-validation accuracy
best_model = grid.best_estimator_
scores = cross_val_score(best_model, X_pca, Y_encoded, cv=5)
print("Cross-validation Accuracy: {:.2f}%".format(np.mean(scores) * 100))

# Evaluation on test data
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Best SVM Model")
plt.tight_layout()
plt.show()
