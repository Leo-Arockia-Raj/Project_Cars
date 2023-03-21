import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# 1. ------------------------------------- Loading DataSet -------------------------------------------#
car_df = pd.read_csv("cars_price.csv")  # load dataset as DataFrame
print("Columns :\n", car_df.columns)  # Column Names
print("Dimensions of DataSet: ", car_df.shape)

# 2. -----Exploratory Data Analysis using countplot, Boxplot, histplot, lmplot------------------------#

# 3. ---------------------------------------Pre-Processing -------------------------------------------#

# Exploring Target Values and Removing if Null Values Present:
print(car_df['price'].unique())  # Identify Target Column Null values
car_df['price'].replace('?', np.nan, inplace=True)
print("Null Values in Target Columns:", car_df['price'].isnull().sum())
null_price_index = car_df[car_df['price'].isnull()].index  # Null Value Index
car_df.drop(null_price_index, inplace=True)  # drop null indexes from DataSet
print("Dimensions After dropping Targe Null Values: ", car_df.shape)

# Separate Feature and Target Matrix
X = car_df.drop(columns=['price'])
y = car_df[['price']]

# Exploring Feature Columns
Xcols = X.columns
for col in X.columns:
    print(col + ": ")
    print(X[col].value_counts())

# ------------------------------------------ Split Train and Test Data---------------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=X[["engine-type"]])


# ---------------------------------------Start: Handling Missing Values -------------------------------#
# # Checking for Null values or Null replaced Symbols
print("Result for null or null substitutes column-wise")
for col in X_train.columns:
    print(col + ": ")
    print(X_train[col].value_counts())

# Found Null value substituted with '?' --> Replace '?' with numpy.nan
# Train Data
X_train.replace('?', np.nan, inplace=True)
# Test Data ---------------------------
X_test.replace('?', np.nan, inplace=True)
# --------------------------------------

print(" Total Null Values in Each Train Feature Column:")
print(X_train.isnull().sum())

# Train
print("Drop Train Null Value indexes")
X_train_null_index = X_train[X_train.isnull().any(axis=1)].index
X_train.drop(X_train_null_index, inplace=True, axis='index')
y_train.drop(X_train_null_index, inplace=True, axis='index')

# Test -------------------------------------------------
print("Drop Test Null Value indexes")
X_test_null_index = X_test[X_test.isnull().any(axis=1)].index
X_test.drop(X_test_null_index, inplace=True, axis='index')
y_test.drop(X_test_null_index, inplace=True, axis='index')
# ---------------------------------------------------------

# Index are reset after split
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# ----------------------------End : Handling Missing Values ----------------------------------------#

# ------------------------ Start: Handling Categorical Columns: ------------------------------------#

# Target -- Needs to be typecast to integer
#  Train
y_train['price'] = y_train['price'].astype('int')
# Test ------------------------------------------
y_test['price'] = y_test['price'].astype('int')
# -----------------------------------------------

# 'normalized-losses'
#  Train
X_train['normalized-losses'] = X_train['normalized-losses'].astype('int')
# Test ------------------------------------------
X_test['normalized-losses'] = X_test['normalized-losses'].astype('int')
# -----------------------------------------------

# 'make'
print(X_train['make'].value_counts())
print(X_train['make'].nunique())
OE = OrdinalEncoder(categories=[['mercury', 'renault', 'isuzu', 'alfa-romero', 'chevrolet', 'jaguar',
                                 'porsche', 'saab', 'audi', 'plymouth', 'bmw', 'mercedes-benz',
                                 'dodge', 'volvo', 'peugot', 'subaru', 'volkswagen', 'honda',
                                 'mitsubishi', 'mazda', 'nissan', 'toyota']])
# Train
X_train['make'] = OE.fit_transform(X_train[['make']])
# Test ------------------------------------------
X_test['make'] = OE.transform(X_test[['make']])
# -----------------------------------------------

# 'fuel-type'
print(X_train['fuel-type'].value_counts())
print(X_train['fuel-type'].nunique())
# Train
X_train = pd.get_dummies(X_train, columns=['fuel-type'])
# Test ------------------------------------------
X_test = pd.get_dummies(X_test, columns=['fuel-type'])
# -----------------------------------------------
print(X_train.info())

# aspiration
print(X_train['aspiration'].value_counts())
print(X_train['aspiration'].nunique())
# Train
X_train = pd.get_dummies(X_train, columns=['aspiration'])
# Test ------------------------------------------
X_test = pd.get_dummies(X_test, columns=['aspiration'])
# -----------------------------------------------
print(X_train.info())

# num-of-doors
print(X_train['num-of-doors'].value_counts())
print(X_train['num-of-doors'].nunique())
# Train
X_train = pd.get_dummies(X_train, columns=['num-of-doors'])
# Test ------------------------------------------
X_test = pd.get_dummies(X_test, columns=['num-of-doors'])
# -----------------------------------------------
print(X_train.info())

# body-style
print(X_train['body-style'].value_counts())
print(X_train['body-style'].nunique())
# Train
X_train.drop(['body-style'], inplace=True, axis=1)
# Test ------------------------------------------
X_test.drop(['body-style'], inplace=True, axis=1)
# -----------------------------------------------
print(X_train.info())

# drive-wheels
print(X_train['drive-wheels'].value_counts())
print(X_train['drive-wheels'].nunique())
# Train
X_train = pd.get_dummies(X_train, columns=['drive-wheels'])
# Test ------------------------------------------
X_test = pd.get_dummies(X_test, columns=['drive-wheels'])
# -----------------------------------------------
print(X_train.info())

# engine-location
print(X_train['engine-location'].value_counts())
print(X_train['engine-location'].nunique())
# Train
X_train.drop(['engine-location'], inplace=True, axis=1)
# Test ------------------------------------------
X_test.drop(['engine-location'], inplace=True, axis=1)
# -----------------------------------------------
print(X_train.info())

# engine-type
print(X_train['engine-type'].value_counts())
print(X_train['engine-type'].nunique())
# Train
X_train = pd.get_dummies(X_train, columns=['engine-type'])
# Test ------------------------------------------
X_test = pd.get_dummies(X_test, columns=['engine-type'])
# -----------------------------------------------
print(X_train.info())

# num-of-cylinders
print(X_train['num-of-cylinders'].value_counts())
print(X_train['num-of-cylinders'].nunique())
OE = OrdinalEncoder(categories=[['twelve', 'three', 'eight', 'two', 'five', 'six', 'four']])
# Train
X_train['num-of-cylinders'] = OE.fit_transform(X_train[['num-of-cylinders']])
# Test ------------------------------------------
X_test['num-of-cylinders'] = OE.transform(X_test[['num-of-cylinders']])
# -----------------------------------------------
print(X_train.info())

# fuel-system
print(X_train['fuel-system'].value_counts())
print(X_train['fuel-system'].nunique())
# Train
X_train.drop(['fuel-system'], inplace=True, axis=1)
# Test ------------------------------------------
X_test.drop(['fuel-system'], inplace=True, axis=1)
# -----------------------------------------------
print(X_train.info())

# bore
print(X_train['bore'].value_counts())
print(X_train['bore'].nunique())
# Train
X_train['bore'] = X_train['bore'].astype('float')
# Test ------------------------------------------
X_test['bore'] = X_test['bore'].astype('float')
# -----------------------------------------------
print(X_train.info())

# stroke
print(X_train['stroke'].value_counts())
print(X_train['stroke'].nunique())
# Train
X_train['stroke'] = X_train['stroke'].astype('float')
# Test ------------------------------------------
X_test['stroke'] = X_test['stroke'].astype('float')
# -----------------------------------------------
print(X_train.info())

# horsepower
print(X_train['horsepower'].value_counts())
print(X_train['horsepower'].nunique())
# Train
X_train['horsepower'] = X_train['horsepower'].astype('int')
# Test ------------------------------------------
X_test['horsepower'] = X_test['horsepower'].astype('int')
# -----------------------------------------------
print(X_train.info())

# peak-rpm
print(X_train['peak-rpm'].value_counts())
print(X_train['peak-rpm'].nunique())
# Train
X_train['peak-rpm'] = X_train['peak-rpm'].astype('int')
# Test ------------------------------------------
X_test['peak-rpm'] = X_test['peak-rpm'].astype('int')
print(X_train.info())
# -----------------------------------------------
# ------------------------------ End: Handling Categorical Columns -------------------------- #


# -----------------------------------Start: Feature Scaling ----------------------------------#
# Box Plot Without Scaling
plt.title("Box Plot of Cars Dataset Feature Column Without Scaling")
plt.xlabel("Feature Columns")
plt.ylabel("Feature Values")
sns.boxplot(data=X_train)
plt.xticks(rotation=90)
plt.show()

# Saving and Verifying Train-Test Column Names :
# Train Column Names:
X_train_Cols = X_train.columns
print("Train Column Names:", X_train_Cols)
# Train Column Names:
X_test_Cols = X_test.columns
print("Test Column Names:", X_test_Cols)

from sklearn.preprocessing import StandardScaler

SS = StandardScaler()
# Train
X_train = pd.DataFrame(SS.fit_transform(X_train), columns=X_train_Cols)
# Test ----------------------------------------
X_test = pd.DataFrame(SS.transform(X_test), columns=X_test_Cols)
# ----------------------------------------------

# Box Plot After Scaling:
plt.title("Box Plot of Cars Dataset Feature Column After Scaling")
plt.xlabel("Feature Columns")
plt.ylabel("Feature Values")
sns.boxplot(data=X_train)
plt.xticks(rotation=90)
plt.show()
# ----------------------------------- End: Feature Scaling ----------------------------------#


# ------------------------------------- Modelling -------------------------------------------#

# Initialization for Train-Test Score Comparison Table
Model = []
Train_Score = []
Test_Score = []

# Hyperparameter Tuning Using RandomizedSearchCV:
Gradient_Boosting_Tune = False
if Gradient_Boosting_Tune:
    from sklearn.model_selection import RandomizedSearchCV

    param = {"n_estimators": [10, 100, 500, 1000],
             "learning_rate": [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7],
             "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             }

    rscv = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=param, cv=5, n_iter=50)
    rscv.fit(X_train, y_train)
    print("GradientBoostingRegressor")
    print("Best Score: ", rscv.best_score_)
    print("Best Parameter Values: ", rscv.best_params_)
    print("Best Estimator Selected by Grid", rscv.best_estimator_)

# Handle OverFitting by Manual Tuning of max_features:
GradientBoostingRegressor_Manual_Tune = False
if GradientBoostingRegressor_Manual_Tune:
    max_features = np.arange(1, 31)
    train_scores = []
    test_scores = []
    for mf in max_features:
        gbr_max_features = GradientBoostingRegressor(n_estimators=10, learning_rate=0.3, max_depth=2, random_state=0,
                                                     max_features=mf)
        gbr_max_features.fit(X_train, y_train)
        train_scores.append(gbr_max_features.score(X_train, y_train))
        test_scores.append(gbr_max_features.score(X_test, y_test))

    max_features_summary_dic = {"Max_Features": max_features,
                                "Train Score List": train_scores,
                                "Test Score List": test_scores}
    max_features_summary_df = pd.DataFrame(max_features_summary_dic)

    plt.title("Max_features Gradient Boosting Regressor Hyper Parameter Vs Train/Test Scores ")
    plt.xlabel("Max features Hyper_Paramenter")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.plot(max_features_summary_df["Max_Features"], max_features_summary_df["Train Score List"], 'r*-',
             label='Train Score')
    plt.plot(max_features_summary_df["Max_Features"], max_features_summary_df["Test Score List"], 'bD-',
             label='Test Score')
    plt.legend()
    plt.show()

# Regression Model:
gbc = GradientBoostingRegressor(n_estimators=10, learning_rate=0.3, max_depth=2, random_state=0, max_features=13)
gbc.fit(X_train, y_train)
Model.append("GradientBoostingRegressor")
Train_Score.append(gbc.score(X_train, y_train))
Test_Score.append(gbc.score(X_test, y_test))

# R2_Score:
print("R2_Score of Train and Test:")
from tabulate import tabulate
print(tabulate(np.transpose([Model, Train_Score, Test_Score]), headers=["Model", "Train_Score", "Test_Score"],
               tablefmt="github"))

# Evaluation Metrics Score:
print(" Evaluation Metric Scores: ")
y_train_pred = gbc.predict(X_train)
y_test_pred = gbc.predict(X_test)
print("Train R2_Score: ", metrics.r2_score(y_train, y_train_pred))
print("Test R2_Score: ", metrics.r2_score(y_test, y_test_pred))
print("Train MAE: ", metrics.mean_absolute_error(y_train, y_train_pred))
print("Test MAE: ", metrics.mean_absolute_error(y_test, y_test_pred))
print("Train MSE: ", metrics.mean_squared_error(y_train, y_train_pred))
print("Test MSE: ", metrics.mean_squared_error(y_test, y_test_pred))

# Cross Validation Score:
print("CV Score of Train Dataset:", cross_val_score(gbc, X_train, y_train, cv=6).mean())
print("CV Score of Test Dataset:", cross_val_score(gbc, X_test, y_test, cv=6).mean())

# Combining OHE Feature importances for mapping to original Feature:
feature_original = car_df.columns
features_original_importances = []
cat_vars_list = []
for var in feature_original:
    indices = [i for i, name in enumerate(gbc.feature_names_in_) if name.startswith(var)]
    importance = np.sum(gbc.feature_importances_[indices])
    features_original_importances.append(importance)

# Index of Sorted features_original_importances list:
sort_index = list(np.argsort(features_original_importances))
sort_index.reverse()

# Sorted bar chart of feature importances for categorical variables:
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=90)
# Sorting using sort_index
features = []
feature_importance = []
for i in sort_index:
    features.append(feature_original[i])
    feature_importance.append(features_original_importances[i])

plt.bar(features, feature_importance)
plt.show()

# Displaying the Feature Importance:
print("Feature Importances: ")
print(tabulate(np.transpose([features, feature_importance]), headers=['Feature', 'Importance'], tablefmt='grid'))
