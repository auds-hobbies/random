#!/usr/bin/env python
# coding: utf-8

# ### Hobby: Credit loan defaulters  
# 
# <div class="alert alert-block alert-info">
# <span style='font-family:Georgia'> 
# <b>Imbalanced classification: <b>    
#     
#     
# - re-sampling with SMOTE  
# - evaluation of classifier with ROC AUC and Class accuracy
# 
# </div>

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score,make_scorer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns

from collections import Counter
from matplotlib import pyplot
from numpy import mean, std

from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline #as imbPipeline


from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,RandomForestRegressor ,GradientBoostingClassifier,BaggingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, auc, accuracy_score,log_loss,roc_auc_score,r2_score,roc_curve


from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline #as imbPipeline

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier 

import xgboost as xgb

import scipy.stats as stats
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler
from sklearn.feature_selection import SelectFromModel
# plotting 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# https://www.nltk.org/data.html 
from joblib import dump
from joblib import load
from sklearn.svm import LinearSVC


# In[2]:


file_credit = "credit.csv"


# ## Load data

# In[3]:



def load_summarize(file_path):
    df0 = pd.read_csv(file_credit)

    print(f"Shape = {df0.shape}")

    missing_cols = [col for col in df0.columns if df0[col].isna().sum()>0]
    print(f"Columns with missing values = {missing_cols}\n")

    df0.head()
    
    # describe the dataset
    pd.set_option('precision', 3)
    print(df0.describe())

    # summarize the class distribution
    target = df0.values[:,-1]
    counter = Counter(target)
    for k,v in counter.items():
        percentage = v / len(target) * 100
        print('\nTarget class distribution: \nClass=%s, Count=%d, Percentage=%.3f%%' % (k, v, percentage))
    
    return df0

df0 = load_summarize(file_credit)
df0.head()


# In[4]:


df0.hist(bins=25,figsize=(10,10))
# show the plot
pyplot.show()


# In[5]:


df = df0.copy()

# summary of different types of variables in dataset
discrete = [var for var in df.columns if df[var].dtype!='O' and df[var].nunique()<df.shape[0]]
continuous = [var for var in df.columns if df[var].dtype!='O' and var not in discrete]
# combination 
string = [discrete, continuous]
# categorical
categorical = [var for var in df.columns if df[var].dtype=='O' and var not in string]

print(f'Total number of variables = {len(df.columns)}\n')
print('# of discrete variables = {}'.format(len(discrete)))
print('# of continuous variables = {}'.format(len(continuous)))
print('# of categorical variables = {}\n'.format(len(categorical)))
#print('There are {} mixed variables'.format(len(mixed)))
print(f'Categorical variables = {categorical}\n')


# # Label encoding; Split  
# 
# - label encoding on categorical features 
# - splitting data into train and test sets

# In[6]:


def transforms_le(df):
    le_a11 = LabelEncoder()
    le_a34 = LabelEncoder()
    le_a43 = LabelEncoder()
    le_a65 = LabelEncoder()
    le_a75 = LabelEncoder()
    le_a93 = LabelEncoder()
    le_a101 = LabelEncoder()
    le_a121 = LabelEncoder()
    le_a143 = LabelEncoder()
    le_a152 = LabelEncoder()
    le_a173 = LabelEncoder()
    le_a192 = LabelEncoder()
    le_a201 = LabelEncoder()
    
    df['a11_enc'] = le_a11.fit_transform(df.A11)
    df['a34_enc'] = le_a34.fit_transform(df.A34)
    df['a43_enc'] = le_a43.fit_transform(df.A43)
    df['a65_enc'] = le_a65.fit_transform(df.A65)
    df['a75_enc'] = le_a75.fit_transform(df.A75)
    df['a93_enc'] = le_a93.fit_transform(df.A93)
    
    df['a101_enc'] = le_a101.fit_transform(df.A101)
    df['a121_enc'] = le_a121.fit_transform(df.A121)
    df['a143_enc'] = le_a143.fit_transform(df.A143)
    df['a152_enc'] = le_a152.fit_transform(df.A152)
    df['a173_enc'] = le_a173.fit_transform(df.A173)
    df['a192_enc'] = le_a192.fit_transform(df.A192)
    df['a201_enc'] = le_a201.fit_transform(df.A201)
    
    return df

df1 = transforms_le(df)


# In[7]:


def split_resample(df1):
    df2 = df1.drop(categorical, axis=1)
    df2 = df2[['6', '1169', '4', '4.1', '67', '2', '1', 'a11_enc', 'a34_enc',
           'a43_enc', 'a65_enc', 'a75_enc', 'a93_enc', 'a101_enc', 'a121_enc',
           'a143_enc', 'a152_enc', 'a173_enc', 'a192_enc', 'a201_enc','1.1']]

    pre_X = df2.drop(df2.columns[-1],axis=1)
    pre_y = df2.iloc[:,-1]

    pre_X_train, pre_X_test, pre_y_train, pre_y_test = train_test_split(pre_X, pre_y, stratify=df2['1.1'],
                                                        test_size=0.25,
                                                        #train_size=0.25,
                                                        random_state=3)

    print(f"x-train shape = {pre_X_train.shape}, \ny-test shape = {pre_y_train.shape}")
    print(f"x-test shape = {pre_X_test.shape}, \ny-test shape = {pre_y_test.shape}\n\n")
    
    print('Train set:')
    sme = SMOTEENN(random_state=41)
    X_train_res,y_train_res = sme.fit_resample(pre_X_train, pre_y_train)
    print('Original dataset = %s' % (Counter(pre_y_train)))
    print('Resampled dataset = %s \n' % (Counter(y_train_res)))
    
    print('Test set:')
    X_test_res,y_test_res = sme.fit_resample(pre_X_test, pre_y_test)
    print('Original test set = %s' % (Counter(pre_y_test)))
    print('Resampled test set = %s \n' % (Counter(y_test_res)))
    
    return pre_X_train,pre_y_train,pre_X_test,pre_y_test,X_train_res,y_train_res, X_test_res, y_test_res


pre_X_train,pre_y_train,pre_X_test,pre_y_test,X_train_res,y_train_res, X_test_res, y_test_res = split_resample(df1)


# # Modelling    
# 
# 
# - modelling with multiple algorithms 
# - evaluation of models with AUC and Class Accuracy

# In[8]:


def fit_predict_evaluate(X_train_res,y_train_res, X_test_res, y_test_res,model):
    # train
    model.fit(X_train_res,y_train_res)
    predictions_train = model.predict(X_train_res)
    pred_prob_train = model.predict_proba(X_train_res)
    logloss_score_train = log_loss(y_train_res, pred_prob_train)
    auc0_train = roc_auc_score(y_train_res, pred_prob_train[:, 0])
    auc1_train = roc_auc_score(y_train_res, pred_prob_train[:, 1]) # 
    class_accuracy_score_train = accuracy_score(y_train_res, predictions_train)
    prob_accuracy_score_train = accuracy_score(y_train_res, predictions_train)
    
    # test
    model.fit(X_test_res, y_test_res)
    predictions_test = model.predict(X_test_res)
    pred_prob_test = model.predict_proba(X_test_res)
    logloss_score_test = log_loss(y_test_res, pred_prob_test)
    auc0_test = roc_auc_score(y_test_res, pred_prob_test[:, 0])
    auc1_test = roc_auc_score(y_test_res, pred_prob_test[:, 1]) # 
    class_accuracy_score_test = accuracy_score(y_test_res, predictions_test)
    prob_accuracy_score_test = accuracy_score(y_test_res, predictions_test)
        
    return predictions_train, auc1_train, class_accuracy_score_train,            predictions_test, auc1_test, class_accuracy_score_test


def instantiate_models():
    models, names = list(), list()
    # Dummy classifier 
    models.append(DummyClassifier(strategy='uniform'))
    names.append('Dummy')
    # LR 
    models.append(LogisticRegression(max_iter=2000, solver='lbfgs')) # solver='liblinear'
    names.append('LR')
    # XGBoost 
    #models.append(XGBClassifier())
    #names.append('XGB')
    # XGBoost - Regressor 
    #models.append(XGBRegressor())
    #names.append('XGB_Regressor')
    # LDA 
    models.append(LinearDiscriminantAnalysis())
    names.append('LDA')
    # GNB 
    models.append(GaussianNB())
    names.append('GNB')
    # MNB 
    models.append(MultinomialNB())
    names.append('MNB')
    # GPC
    models.append(GaussianProcessClassifier())
    names.append('GPC')
    # KNN Classifier 
    models.append(KNeighborsClassifier())
    names.append('KNN')
    # Decision tree classifier 
    models.append(DecisionTreeClassifier())
    names.append('CART')
    # ensemble methods 
    models.append(AdaBoostClassifier())
    names.append('AdaB')
    # Gradient boosting classifier 
    models.append(GradientBoostingClassifier())
    names.append('GB')
    # RF
    models.append(RandomForestClassifier(n_estimators=10))
    names.append('RF')
    # Extra trees
    models.append(ExtraTreesClassifier(n_estimators=10))
    names.append('ET')
    # SVC
    #models.append(SVC(C=1.5)) #Probability=True, gamma='Auto', C=1.5
    #names.append('SVM')
    
    return models, names     


models, names = instantiate_models()
lst_predictions_train = []
lst_predictions_prob = []
lst_logloss_score = []
lst_auc0 = []
lst_auc1_train = []
lst_class_accuracy_score_train = []
lst_prob_accuracy_score_train = []
#test 
lst_predictions_test = []
#lst_predictions_prob = []
#lst_logloss_score = []
#lst_auc0 = []
lst_auc1_test = []
lst_class_accuracy_score_test = []
lst_prob_accuracy_score = []

# evaluate each model 
print(f"\nResults - \n")
for i in range(len(models)):
    # evaluate the model and store results    
    predictions_train, auc1_train, class_accuracy_score_train,     predictions_test, auc1_test, class_accuracy_score_test = fit_predict_evaluate(X_train_res,y_train_res,                                                                                  X_test_res, y_test_res,models[i])
    
    lst_predictions_train.append(predictions_train)
    #lst_predictions_prob.append(predictions_prob)
    #lst_logloss_score.append(logloss_score)
    #lst_auc0.append(auc0)
    lst_auc1_train.append(auc1_train)
    lst_class_accuracy_score_train.append(class_accuracy_score_train)
    #lst_prob_accuracy_score.append(prob_accuracy_score)
    
    # test 
    lst_predictions_test.append(predictions_test)
    #lst_predictions_prob.append(predictions_prob)
    #lst_logloss_score.append(logloss_score)
    #lst_auc0.append(auc0)
    lst_auc1_test.append(auc1_test)
    lst_class_accuracy_score_test.append(class_accuracy_score_test)
    #lst_prob_accuracy_score.append(prob_accuracy_score)
    
auc_df_train = pd.DataFrame(zip(names,lst_auc1_train,lst_class_accuracy_score_train ),columns=['model','auc1_train','class_accuracy_score_train'])
auc_df2_train = auc_df_train[['model','auc1_train','class_accuracy_score_train']]

auc_df_test = pd.DataFrame(zip(names,lst_auc1_test,lst_class_accuracy_score_test ),columns=['model_test','auc1_test','class_accuracy_score_test'])
auc_df2_test = auc_df_test[['model_test','auc1_test','class_accuracy_score_test']]

df_train = auc_df2_train.copy()
#print(f"\nnew df shape = {df_train.shape}")
#print(f"Missing values = {[col for col in df_train.columns if df_train[col].isna().sum()>0]}")
#print(df_train)

df_test = auc_df2_test.copy()
#print(f"\nnew df shape = {df_test.shape}")
#print(f"Missing values = {[col for col in df_test.columns if df_test[col].isna().sum()>0]}")
#print(df_test)

df_evaluation = pd.concat([df_train, df_test],axis=1, sort=False)
df_evaluation = df_evaluation.drop(columns=['model_test'], axis=1)
df_evaluation.columns = ['model','AUC_train','Accuracy_train','AUC_test','Accuracy_test']
df_evaluation


# In[ ]:




