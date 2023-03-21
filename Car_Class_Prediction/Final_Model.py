import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# 1. ------------------------------------- Loading DataSet -------------------------------------------#
car_df = pd.read_csv("cars_class.csv")  # load dataset as DataFrame
print("Columns :\n", car_df.columns)  # Column Names
print("Dimensions of DataSet: ", car_df.shape)

# 2. -----Exploratory Data Analysis using countplot, boxplot, histplot, heatmap ------------------------#

# 3. ---------------------------------------Pre-Processing -------------------------------------------#
# Target contain no missing values

# Separate Feature and Target Matrix
X = car_df.iloc[:, 1:-1]
y = car_df.iloc[:, -1]

# Exploring Feature Columns
Xcols = X.columns
print(X.info())

# ------------------------------------------ Split Train and Test Data---------------------------------#

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# ----------------------------Handling Missing Values : NO Missing Values-------------------------------#
# -----------------------Handling Categorical Columns : No Categorical Columns--------------------------#
# Index are reset after split
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# -----------------------------------Start: Feature Scaling ----------------------------------#
# Box Plot Without Scaling
plt.title("Box Plot of Features Without Scaling")
plt.xlabel("Feature Columns")
plt.ylabel("Feature Values")
sns.boxplot(data=X_train)
plt.xticks(rotation=90)
plt.show()

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
# Train
X_train = pd.DataFrame(SS.fit_transform(X_train), columns=Xcols)
# Test ------------------------------------------------------
X_test = pd.DataFrame(SS.transform(X_test), columns=Xcols)
# -----------------------------------------------------------

# Box Plot After Scaling:
plt.title("Box Plot of Features After Scaling")
plt.xlabel("Feature Columns")
plt.ylabel("Feature Values")
sns.boxplot(data=X_train)
plt.xticks(rotation=90)
plt.show()
# ----------------------------------- End: Feature Scaling ----------------------------------#

# ------------------------------------- Modelling -------------------------------------------#

# Classification Model
svc = SVC(C=3, kernel='linear')
svc.fit(X_train, y_train)

# Evaluation Metrics Score:
print(" Evaluation Metric Scores: ")
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
# Accuracy
print("Accuracy Train Score: ", metrics.accuracy_score(y_train, y_train_pred))
print("Accuracy Test Score: ", metrics.accuracy_score(y_test, y_test_pred))
# Confusion Matrix
print("Train Confusion Matrix: ", metrics.confusion_matrix(y_train, y_train_pred))
plt.title("Confusion Matrix of Train Data")
sns.heatmap(metrics.confusion_matrix(y_train, y_train_pred), annot=True)
plt.show()
print("Test Confusion Matrix: ", metrics.confusion_matrix(y_test, y_test_pred))
plt.title("Confusion Matrix of Test Data")
sns.heatmap(metrics.confusion_matrix(y_test, y_test_pred), annot=True)
plt.show()
# F1 Score
print("F1 Train Score: ", metrics.f1_score(y_train, y_train_pred, average='macro'))
print("F1 Test Score: ", metrics.f1_score(y_test, y_test_pred, average='macro'))
# Cross Validation Score
print("Train Dataset CV Score: ", cross_val_score(svc, X_train, y_train, cv=10).mean())
print("Test Dataset CV Score: ", cross_val_score(svc, X_test, y_test, cv=10).mean())

# Visualizing Feature Importance:
# calculate the feature importance using the magnitude of the coefficients
coef = svc.coef_
feature_importance = np.abs(coef).sum(axis=0)

# Sorting feature importance in descending order
feature_names = X_train.columns
sort_index = np.flipud(np.argsort(feature_importance))
feature_names_sort = []
feature_importance_sort = []
for i in sort_index:
    feature_names_sort.append(feature_names[i])
    feature_importance_sort.append(feature_importance[i])

# Plotting the feature importance in descending order
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Weights')
plt.xticks(rotation=90)
plt.bar(feature_names_sort, feature_importance_sort)
plt.show()
