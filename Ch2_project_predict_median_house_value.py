#Chapter 2 – End-to-end Machine Learning project
#Task here is to predict median house values in Californian districts, 
#given a number of features from these districts.

#Import libraries
import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
 
 
#Declare Constants and paths
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/" 
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#Util functions
def fetch_housing_data(housing_url, housing_path): 
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz") 
    urllib.request.urlretrieve(housing_url, tgz_path) 
    housing_tgz = tarfile.open(tgz_path) 
    housing_tgz.extractall(path=housing_path) 
    housing_tgz.close()

def load_housing_data(housing_path): 
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path)

def display_scores(scores, modelType):
    print("Model Scores: ", modelType)
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:", scores.std())

######STEP 1: DATA GATHERING
#Download the California housing data
fetch_housing_data(HOUSING_URL,HOUSING_PATH)
#Load downloaded data
housing = load_housing_data(HOUSING_PATH)

######STEP 2: PRELIMINARY Data Exploration
housing.head() # check top 5 rows of housing
housing.info() # check the non-null count of each column
housing["ocean_proximity"].value_counts() # check catagories and count 
housing.hist(bins=50, figsize=(20,15)) 
plt.show()

######STEP 3: TRAIN/TEST SPLIT 
# This is done early to keep the training set aside and to avoid "Data Snooping Bias"
# Stratified split on income_cat. Test data (20%)
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#Dropping dummy columns only create for stratification
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

######STEP 4: TRAINING Data Exploration
# Make a for exploration, to avoid harming training data
housing = strat_train_set.copy()

# 4.1: Visualization
#Visualize geographical information
housing.plot(kind="scatter", x="longitude", y="latitude")

# using alpha parameter reduce the opacity, 
# so that more dense areas can be clearly identified.
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# look at population and median house value
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
            s=housing["population"]/100, label="population", 
            figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), 
            colorbar=True,) 
plt.legend()

# 4.2: Correlations
# Correlation of every feature with each other
corr_matrix = housing.corr()

# Another way of checking correlation. By Plotting.
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"] # selecting only a few parameters
scatter_matrix(housing[attributes], figsize=(12, 8))

######STEP 5: FEATURE ENGINEERING
# Try to create some features that will make more sense.
#housing["rooms_per_household"] = housing["total_rooms"]/housing["households"] 
#housing["bedrooms_per_household"] = housing["total_bedrooms"]/housing["households"] 
#housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"] 
#housing["population_per_household"]=housing["population"]/housing["households"]

# Feature Engineering as above, using Custom transformer
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


######STEP 6: ML Data preparation
#Seperate X's from Y, as we don't necessarily want to apply same transformation to both
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

#Construct ML pipeline (input, stadardize, onhot encode, add features)
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), #data imputation
        ('attribs_adder', CombinedAttributesAdder()), #Adding extra features
        ('std_scaler', StandardScaler()), #data standardisation
    ])

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

######STEP 6: TRAINING & EVALUATING Model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#linear Model
lin_reg = LinearRegression()
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

#Decision Tree
tree_reg = DecisionTreeRegressor()
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                scoring="neg_mean_squared_error", cv=10) 
tree_rmse_scores = np.sqrt(-tree_scores)

#Random Forest
rf_reg = RandomForestRegressor()
rf_scores = cross_val_score(rf_reg, housing_prepared, housing_labels,
                scoring="neg_mean_squared_error", cv=10) 
rf_rmse_scores = np.sqrt(-rf_scores)

#SVR-linear - Support Vector Regressor with Linear Kernel
svr_lin_reg = SVR(kernel="linear")
svr_lin_scores = cross_val_score(svr_lin_reg, housing_prepared, housing_labels,
                scoring="neg_mean_squared_error", cv=10) 
svr_lin_rmse_scores = np.sqrt(-svr_lin_scores)

#SVR-rbf - Support Vector Regressor with rbf Kernel
svr_rbf_reg = SVR(kernel="rbf")
svr_rbf_scores = cross_val_score(svr_rbf_reg, housing_prepared, housing_labels,
                scoring="neg_mean_squared_error", cv=10) 
svr_rbf_rmse_scores = np.sqrt(-svr_rbf_scores)


display_scores(lin_rmse_scores,"LinearRegression")
display_scores(tree_rmse_scores,"DecisionTree")
display_scores(rf_rmse_scores,"RandomForest")
display_scores(svr_lin_rmse_scores,"SVR-linear")
display_scores(svr_rbf_rmse_scores,"SVR-rbf")

#As Random Forest is performing best so far. 
# We can try improving the results by using GridSearch
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},# try 12 (3×4) combinations of hyperparameters
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}, # then try 6 (2×3) combinations with bootstrap set as False
]
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, 
                    cv=5, scoring='neg_mean_squared_error',
                    return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


#Trying RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

rnd_feature_importances = rnd_search.best_estimator_.feature_importances_
sorted(zip(rnd_feature_importances, attributes), reverse=True)

######STEP 7: Model results on test data 
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions) 
final_rmse = np.sqrt(final_mse)
print("Final Model TEST RMSE:", final_rmse) #--47730.22690385927

#Confidence Interval
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))