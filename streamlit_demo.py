import streamlit as st
import pickle
import numpy as np
from sklearn import metrics
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
# Load the pipeline and data
# pipe = pickle.load(open('pipe_logistic.sav', 'rb'))
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))
X_train = pickle.load(open('X_train.sav', 'rb'))

dic = {0: 'Bad', 1: 'Good'}

#Function to test certain index of dataset
def test_demo(index):
    values = X_test.iloc[index].astype('float64')  # Input the value from dataset

    # Create four sliders in the sidebar
    a = st.sidebar.slider('External Risk Estimate', 0.0, 100.0, values[16], 1.0)
    b = st.sidebar.slider('Months Since Oldest Trade Open', 0.0, 604.0, values[17], 1.0)
    c = st.sidebar.slider('Months Since Most Recent Trade Open', 0.0, 207.0, values[18], 1.0)
    d = st.sidebar.slider('Average Months in File', 4.0, 332.0, values[19], 1.0)
    e = st.sidebar.slider('Number Satisfactory Trades', 0.0, 100.0, values[20], 1.0 )
    f = st.sidebar.slider('Number Trades 60+ Ever', 0.0, 100.0, values[21], 1.0 )
    g = st.sidebar.slider('Number Trades 90+ Ever', 0.0, 100.0, values[22], 1.0 )
    h = st.sidebar.slider('Percent Trades Never Delinquent', 0.0, 100.0, values[23], 1.0 )
    i = st.sidebar.slider('Months Since Most Recent Delinquency', 0.0, 100.0, values[24], 1.0 )
    j = st.sidebar.slider('Number of Total Trades', 0.0, 104.0, values[25], 1.0 )
    k = st.sidebar.slider('Number of Trades Open in Last 12 Months', 0.0, 19.0, values[26], 1.0 )
    l = st.sidebar.slider('Percent Installment Trades', 0.0, 100.0, values[27], 1.0 )
    m = st.sidebar.slider('Months Since Most Recent Inq excl 7days', 0.0, 100.0, values[28], 1.0 )
    n = st.sidebar.slider('Number of Inq Last 6 Months', 0.0, 100.0, values[29], 1.0 )
    o = st.sidebar.slider('Number of Inq Last 6 Months excl 7days', 0.0, 100.0, values[30], 1.0 )
    p = st.sidebar.slider('Net Fraction Revolving Burden', 0.0, 232.0, values[31], 1.0 )
    q = st.sidebar.slider('Net Fraction Installment Burden',0.0, 471.0, values[32], 1.0 )
    r = st.sidebar.slider('Number Revolving Trades with Balance', 0.0, 100.0, values[33], 1.0 )
    s = st.sidebar.slider('Number Intallment Trades with Balance', 0.0, 100.0, values[34], 1.0 )
    t = st.sidebar.slider('Number Bank/Natl Trades w high utilization ratio', 0.0, 100.0, values[35], 1.0 )
    u = st.sidebar.slider('Percent Trades with Balance', 0.0, 100.0, values[36], 1.0 )
    
    #Print the prediction result
    alg = ['Boosting', 'Random Forest', 'Support Vector Machine']
    classifier = st.selectbox('Which algorithm?', alg)

    if classifier == 'Boosting':
        
        pipe = pickle.load(open('pipe_boost.sav', 'rb'))
        res = pipe.predict(np.array([0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)
        
        st.text('Support Vector Machine Chosen')
        
    elif classifier == 'Random Forest':
        pipe = pickle.load(open('pipe_rf.sav', 'rb'))
        res = pipe.predict(np.array([0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)
        
        st.text('Random Forest Chosen')

    else:
        pipe = pickle.load(open('pipe_svc.sav', 'rb'))
        res = pipe.predict(np.array([0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)
        
        st.text('SVM Chosen')

    
# title
st.title('Heloc Prediction')
# show data
if st.checkbox('Show dataframe'):
    st.write(X_test)
st.write(X_train.reset_index()) # Show the dataset

number = st.text_input('Choose a row of information in the dataset:', 30)  # Input the index number

test_demo(int(number))  # Run the test function
