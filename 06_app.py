# phaly to sari important libraries import kar lhy "phlay": unknown word.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt        
from sklearn import datasets            
from sklearn.model_selection import train_test_split         
from sklearn.decomposition import PCA                        
from sklearn.svm import SVC                                
from sklearn. neighbors import KNeighborsClassifier           
from sklearn.ensemble import RandomForestClassifier            
from sklearn.metrics import accuracy_score                      

# app ki heading
st.write ("""
# explore different ml models and datasets
daikhty han kon sa best ha in may say ?                          
""")

# data set k name ak box may dallk sidebar py lage do              
dataset_name = st.sidebar.selectbox(
    'select dataset',
    ('i_ris','breast cancer','wine')
)
# or sis k nichy calssifie  ka name ak daby may da do 
classifire_name = st.sidebar. selectbox(
    'select classifire',
    ('KNN','SVM','random forest')
    
)
# ab ham nahyy ak function definne krne hay dateset ko laod krnay k lia 
def get_dateset(dataset_name):
    data =None
    if dataset_name == 'iris':
        data =datasets. load_iris()
    elif dataset_name =="wine":
        data = datasets.load_wine()
    else: 
        data =datasets.load_breast_cancer()
    x =data.data
    y= data.target
    
    return x,y
# ab is function ko bula lay gay or x,y variable k equal rakh lay g 
x,y = get_dateset(dataset_name)
# ab hum apny dataset ki shape ko ap pay print ka day gay
st.write('shape of dataset:', x.shape)
st.write('number of classes:',len(np.unique(y)))


# next hum different classifire k parameter ko user input may add kary gy
def add_parameter_ui(classifier_name):
    params = dict()     # create an empty dictionary
    if classifier_name == 'svm':
        c = st.sidebar.slider('c',0.01,10.0)
        params ['c'] = c   # its the degree of corrredt classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('k',1,15)
        params ['k'] = K   # its the numver of nearest neighbour 
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        params ['max_depth'] = max_depth    # depth of every tree that grow in random forest
        n_estimators  = st.sidebar ('n_estimators', 1,100)
        params['n_estimators'] = n_estimators     # number of tree
        return params
    
    
    # ab is function ko bula lay gay or params variable k equal rakh lay gy 
    params  = add_parameter_ui(classifier_name)


# ab hum classifier bnay gay base on classifier_name and params
def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'svm':
        clf = svc (c=params['c'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = RandomForestClassifier(n_estimatores+ params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
        return clf
    
    # ab is function ko bula lay gya or clf variable k equal rakh gy
    clf = get_classifier(classifier_name,params)
    
    
    # ab hum apny dataset ko test and train data may split kar layty ham by 80/20 ratio 
    x_train, x_test, y_train,Y_test = train_test_split (x,y_size =0.2 , random_state=1234)
    
    # abb hum nay apny classifier ki treaining karni hay
    clf.fit(x_train , y_train)
    y_pred = clf.predict(x_test)
    
    
    # model ka accuracy score chek kar lay gy or ise sayy app pay print kar day gy
    acc =accuracy_score(Y_test,y_pred)
    st.write(f'classifier ={classifier_name}')
    st.write(f'accuracy =',acc )
    ##  PLOT DATASET  ##
    # ab hum  apny sary sary features ko 2 dimenssionnal plot pay draw kar gy   using   "PCA"
    pca = pca(2)
    x_projected = pca.fit_tran =transform(x)
    # ab hum apn data 0 or 1 diminssion may slice kar dy gya
    x1  =  x_projected[:,0]
    x2  = x_projected[:,1]
    
    fig =plt. figure()
    plt.scatter(x1, x2,
                c=y,alpha=0.8,
                cmap='viredis')
    plt.xcorr('principal component 1')
    plt.ylabel('principa; component 2')
    plt.colorbar()
    # plt .show()
    st.pyplot(fig)