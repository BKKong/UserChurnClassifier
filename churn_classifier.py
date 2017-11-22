import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# Read in the data
df_ex_outlier = pd.read_csv("churn_features.csv")


### Data Transformation (log, truncate)

df_tranf = df_ex_outlier.copy()

def Truncate(x):
    return min(x, 475000.0)    # roughly 97.5% percentile


def LogTranf(x):
    if x == 0:
        return -1
    else:
        return np.log(x)

cols2 = ['play_sum_1st', 'play_sum_2nd', 'play_sum_3rd']
for col in cols2:
    df_tranf[col] = df_tranf[col].apply(Truncate)

cols = ['play_freq_1st','play_freq_2nd','play_freq_3rd','play_songs_3rd','play_singers_3rd',
        'down_singers_1st','down_singers_2nd','down_singers_3rd']
for col in cols:
    df_tranf[col] = df_tranf[col].apply(LogTranf)


### Building Classification Models

# Helper Functions

def print_results(y_true, y_pred):
    print("Accuracy of the model is: {}".format(accuracy_score(y_true, y_pred)))
    print("Precision of the model is: {}".format(precision_score(y_true, y_pred)))
    print("Recall of the model is: {}".format(recall_score(y_true, y_pred)))
    print("F1-score of the model is: {}".format(f1_score(y_true, y_pred)))


def print_top_coefs(lst, labels):
    out_list = []
    for i in np.argsort(lst)[::-1]:
        out_list.append((labels[i], lst[i]))
    return out_list


# Train Test Split

selected_features = ['play_freq_1st', 'play_perc_1st', 'play_songs_1st', 'play_singers_1st', 'play_sum_1st',
                     'play_freq_2nd', 'play_perc_2nd', 'play_songs_2nd', 'play_singers_2nd', 'play_sum_2nd',
                     'play_freq_3rd', 'play_perc_3rd', 'play_songs_3rd', 'play_singers_3rd', 'play_sum_3rd',
                     'down_freq_1st', 'down_singers_1st',
                     'down_freq_2nd', 'down_singers_2nd',
                     'down_freq_3rd', 'down_singers_3rd',
                     'days_from_lastplay', 'days_from_lastdown']

X = df_tranf[selected_features].values
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)
scaled_X.shape

y = df_tranf['label'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3, random_state=16)


## Model 1: Logistic Regression

# Fit a logistic regression model

try_logRegress = LogisticRegression(penalty='l2', C=10, fit_intercept=True) 
try_logRegress.fit(X_train, y_train)
y_train_pred = try_logRegress.predict(X_train)
y_test_pred = try_logRegress.predict(X_test)

# Tune Parameters with Grid Search

tuned_parameters = {'C': [0.001,0.01,0.1,1,10,100],
                    'penalty': ['l1','l2']}

clf = GridSearchCV(estimator=LogisticRegression(fit_intercept=True), param_grid=tuned_parameters, cv=5,
                       scoring='roc_auc')
clf.fit(X_train, y_train)

print(clf.best_params_, clf.best_score_)

# Retrain Model with best parameters

model_logRegress = LogisticRegression(penalty='l1', C=0.01, fit_intercept=True) 
model_logRegress.fit(X_train, y_train)
y_train_predict = model_logRegress.predict(X_train)
y_test_predict = model_logRegress.predict(X_test)

print("Training set scores:")
print_results(y_train, y_train_predict)

print("Test set scores:")
print_results(y_test, y_test_predict)

# AUC Score 

print("Area Under Curve (AUC) of the Logistic Regression is: {}".format(roc_auc_score(y_test, y_test_predict)))


## Model 2: Multinomial Naive Bayes 

model_naiveBayes = MultinomialNB(alpha=0.5, fit_prior=True, class_prior=None)  
model_naiveBayes.fit(X_train, y_train)
y_train_predict = model_naiveBayes.predict(X_train)
y_test_predict = model_naiveBayes.predict(X_test)

print("Training set scores:")
print_results(y_train, y_train_predict)

print("Test set scores:")
print_results(y_test, y_test_predict)


## Model 3: Random Forest

# Fit a random forest model
try_randForest = RandomForestClassifier(max_depth = 25, n_estimators = 200, min_samples_leaf = 10)    
try_randForest.fit(X_train, y_train)
y_train_pred = try_randForest.predict(X_train)
y_test_pred = try_randForest.predict(X_test)

# AUC Score
print(roc_auc_score(y_test, y_test_predict))

print("Training set scores:")
print_results(y_train, y_train_predict)

print("Testing set scores:")
print_results(y_test, y_test_predict)


# Tune random forest model parameters

parameters_rf = {"n_estimators": np.arange(100, 301, 50),
                  "min_samples_leaf": np.arange(10,20,5)}

rf_grid = GridSearchCV(estimator=RandomForestClassifier(max_depth=15), param_grid=parameters_rf, cv=5,
                       scoring='roc_auc')
rf_grid.fit(X_train, y_train)

print(rf_grid.best_params_, rf_grid.best_score_)


## Model 4: XGBoost

# Fit a XGBoost model
try_xgboost = xgb.XGBClassifier(max_depth=5, n_estimators=80, learning_rate=0.1, min_child_weight=1, 
                                gamma=0, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic")
try_xgboost.fit(X_train, y_train)
y_train_pred = try_xgboost.predict(X_train)
y_test_pred = try_xgboost.predict(X_test)

# AUC Score

print(roc_auc_score(y_test, y_test_pred))

# Tune parameters for xgboost model

parameters_xgb = {"max_depth": [8, 10, 12]}
xgb_grid = GridSearchCV(estimator=xgb.XGBClassifier(n_estimators=200, min_child_weight=6, learning_rate=0.1, gamma=0,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   objective="binary:logistic"), 
                        param_grid=parameters_xgb, cv=5,
                        scoring='roc_auc')
xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_params_, xgb_grid.best_score_)

# Retrain XGBoost model with tuned parameters

model_xgboost = xgb.XGBClassifier(max_depth=8, n_estimators=200, learning_rate=0.1, min_child_weight=6, 
                                gamma=0, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic")
model_xgboost.fit(X_train, y_train)
y_train_predict = model_xgboost.predict(X_train)
y_test_predict = model_xgboost.predict(X_test)

print(roc_auc_score(y_test, y_test_predict))

print("Training set scores:")
print_results(y_train, y_train_predict)

print("Training set scores:")
print_results(y_test, y_test_predict)




