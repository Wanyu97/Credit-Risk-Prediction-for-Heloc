
# coding: utf-8

# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


# In[3]:


df = pd.read_csv('heloc_dataset_v1.csv')


# In[4]:



# In[5]:


df = df.replace(-9,np.nan)
df = df.dropna()


# In[6]:


df = df.replace(-8, np.nan)
df = df.replace(-7, np.nan)

imputer = Imputer(strategy="most_frequent")
for i in range(1,24):
    df.iloc[:, [i]] = imputer.fit_transform(df.iloc[:, [i]])


# In[7]:


data_dict = {}
data_dict['data_original'] = df
data_dict['features_colnames'] = df.columns
data_dict['y'] = pd.factorize(df['RiskPerformance'])[0]

categorical1 = pd.get_dummies(data_dict['data_original'][data_dict['features_colnames'][10]])
categorical2 = pd.get_dummies(data_dict['data_original'][data_dict['features_colnames'][11]])
categorical = pd.concat([categorical1,categorical2],axis = 1,ignore_index=True)

numeric = pd.concat(([data_dict['data_original'][data_dict['features_colnames'][1:10]],data_dict['data_original'][data_dict['features_colnames'][12:24]]]),axis = 1,ignore_index=True)
data_dict['X'] = pd.concat([categorical,numeric],axis = 1,ignore_index=True)



# In[8]:


data_dict['X']


# In[9]:


categorical


# In[10]:


numeric


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    
# The function `init_classifiers` returns a list of classifiers to be trained on the datas
def init_classifiers():
    return([(SVC(), model_names[0], param_grid_svc), 
            (LogisticRegression(), model_names[1], param_grid_logistic),
            (KNeighborsClassifier(), model_names[2], param_grid_knn),
            (GaussianNB(), model_names[3], param_grid_nb),
            (DecisionTreeClassifier(), model_names[4], param_grid_tree),
            (RandomForestClassifier(), model_names[6], param_grid_rf),
            (AdaBoostClassifier(), model_names[7], param_grid_boost)
           ])

# 'model_names' contains the names  that we will use for the above classifiers
model_names = ['SVM','LR','KNN','NB','Tree','QDA','RF','Boosting']


def evaluate_model(data_dict, model, model_name, params):
    np.random.seed(1)
    #split data
    X = data_dict['X']
    y = data_dict['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    data_dict['X_train'] = X_train
    data_dict['X_test'] = X_test
    data_dict['y_train'] = y_train
    data_dict['y_test'] = y_test
    #scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    grid_search = GridSearchCV(model, params, cv=3) 
    grid_search.fit(X_train,y_train)
    
    

    model_selection = grid_search.best_estimator_ 
    predictions = model_selection.predict(X_test)

    accuracy = (y_test==predictions).mean() 
    model_dic = {} 
    model_dic['Classifier'] = model_name
    model_dic['Best Parameter'] = grid_search.best_estimator_.get_params()
    model_dic['Test Score'] = accuracy
    model_dic['CV Score'] = grid_search.best_score_ 
    model_dic['Time'] = grid_search.refit_time_
    
    
    return model_dic

model_names = ['SVM','LR','KNN','RF','Boosting']

    
# The function `init_classifiers` returns a list of classifiers to be trained on the datas
def init_classifiers():
    return([(SVC(), model_names[0], param_grid_svc), 
            (LogisticRegression(), model_names[1], param_grid_logistic),
            (KNeighborsClassifier(), model_names[2], param_grid_knn),
            (RandomForestClassifier(), model_names[3], param_grid_rf),
            (AdaBoostClassifier(), model_names[4], param_grid_boost)
           ])

# 'model_names' contains the names  that we will use for the above classifiers

# the training parameters of each model
param_grid_svc = [{'C':[0.1,1],'kernel':['rbf','linear','poly'], 'max_iter':[-1],'random_state':[1]}]
param_grid_logistic = [{'C':[0.1,1], 'penalty':['l1','l2'],'random_state':[1]}]
param_grid_knn = [{},{'n_neighbors':[1,2,3,4]}]
param_grid_rf = [{'random_state':[1]},{'n_estimators':[30,50,70,100],'max_features':[0.2, 0.3, 0.5, 0.8], 'bootstrap':[True],'random_state':[1]}]
param_grid_boost = [{'random_state':[1]},{'n_estimators':[30,50,70,100],'learning_rate':[0.1,1],'random_state':[1]}]


# In[12]:


res_list = []
classifiers = init_classifiers()
for i in range(len(classifiers)):
    res_list.append(evaluate_model(data_dict, classifiers[i][0], classifiers[i][1], classifiers[i][2]))
    
df_model_comparison = pd.DataFrame(res_list, columns = ['Classifier','Best Parameter','Dataset','Test Score','CV Score']).sort_values(by=['Classifier','Dataset']).reset_index(drop=True)
    
    
df_model_comparison


# In[13]:


df_model_comparison['Best Parameter'][3]


# In[ ]:




