from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import Imputer


df = pd.read_csv('heloc_dataset_v1.csv')

df["is_duplicate"]= df.duplicated()
df = df.drop(df[df['is_duplicate'] == True].index)
df[df['is_duplicate'] == True]

df = df.replace(-9, np.nan)
df = df.dropna()
df = df.replace(-8, np.nan)
df = df.replace(-7, np.nan)

imputer = Imputer(strategy="most_frequent")
for i in range(1,24):
    df.iloc[:, [i]] = imputer.fit_transform(df.iloc[:, [i]])
    

data_dict = {}
data_dict['data_original'] = df
data_dict['features_colnames'] = df.columns
data_dict['y'] = pd.factorize(df['RiskPerformance'])[0]

categorical1 = pd.get_dummies(data_dict['data_original'][data_dict['features_colnames'][10]])
categorical2 = pd.get_dummies(data_dict['data_original'][data_dict['features_colnames'][11]])
categorical = pd.concat([categorical1, categorical2], axis=1, ignore_index=True)

numeric = pd.concat(([data_dict['data_original'][data_dict['features_colnames'][1:10]],
                      data_dict['data_original'][data_dict['features_colnames'][12:24]]]), axis=1, ignore_index=True)
data_dict['X'] = pd.concat([categorical, numeric], axis=1, ignore_index=True)

X = data_dict['X']
y = data_dict['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

data_dict['X_train'] = X_train.reset_index(drop=True)
data_dict['X_test'] = X_test.reset_index(drop=True)
data_dict['y_train'] = y_train
data_dict['y_test'] = y_test


pipe_svc = Pipeline([('standard', StandardScaler()), 
                     ('SVC', SVC(C = 1,
                                 cache_size = 200,
                                 class_weight = None,
                                 coef0 = 0.0,
                                 decision_function_shape = 'ovr',
                                 degree = 3,
                                 gamma = 'auto_deprecated',
                                 kernel = 'rbf',
                                 max_iter = -1,
                                 probability = False,
                                 random_state = 1,
                                 shrinking = True,
                                 tol = 0.001,
                                 verbose = False))])

pipe_svc.fit(X_train,y_train)
# pipe_logistic = Pipeline([('standard', StandardScaler()), 
#                           ('logistic regression',
#                            LogisticRegression())])
# pipe_knn = Pipeline([('standard', StandardScaler()), 
#                      ('KNN',KNeighborsClassifier())])
pipe_rf = Pipeline([('standard', StandardScaler()), 
                    ('random forest', RandomForestClassifier(bootstrap = True,
                                                             class_weight = None,
                                                             criterion = 'gini',
                                                             max_depth = None,
                                                             max_features = 0.2,
                                                             max_leaf_nodes = None,
                                                             min_impurity_decrease = 0.0,
                                                             min_impurity_split = None,
                                                             min_samples_leaf = 1,
                                                             min_samples_split = 2,
                                                             min_weight_fraction_leaf = 0.0,
                                                             n_estimators = 100,
                                                             n_jobs = None,
                                                             oob_score = False,
                                                             random_state = 1,
                                                             verbose = 0,
                                                             warm_start = False))])

pipe_rf.fit(X_train,y_train)
pipe_boost = Pipeline([('adaboost', AdaBoostClassifier(algorithm = 'SAMME.R',
                                                       base_estimator = None,
                                                       learning_rate = 0.1,
                                                       n_estimators = 100,
                                                       random_state = 1))])
pipe_boost.fit(X_train,y_train)




pickle.dump(data_dict['X_train'], open("X_train.sav", "wb"))
pickle.dump(data_dict['X_test'], open("X_test.sav", "wb"))
pickle.dump(y_test, open("y_test.sav", "wb"))
pickle.dump(y_train, open("y_train.sav", "wb"))
pickle.dump(pipe_svc, open("pipe_svc.sav", "wb"))
# pickle.dump(pipe_logistic, open("pipe_logistic.sav", "wb"))
# pickle.dump(pipe_knn, open("pipe_knn.sav", "wb"))
pickle.dump(pipe_rf, open("pipe_rf.sav", "wb"))
pickle.dump(pipe_boost, open("pipe_boost.sav", "wb"))

