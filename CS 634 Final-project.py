#!/usr/bin/env python
# coding: utf-8

# In[315]:


import numpy as np
import matplotlib.pyplot as mlt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix,make_scorer
from sklearn.neighbors import KNeighborsClassifier


# In[316]:


dataset = pd.read_csv('/Users/himanipatel/Downloads/archive/chipotle_stores.csv')


# dataset.head()

# In[317]:


dataset.info()


# In[318]:


dataset.describe()


# In[319]:


A=dataset.drop(['address'],axis=1)
B=dataset['state'].copy()


# In[320]:


A.head()


# In[321]:


B.head()


# In[322]:


A["latitude"]=A["latitude"].fillna(A["latitude"].mean())


# In[323]:


A["longitude"]=A["longitude"].fillna(A["longitude"].mode()[0])


# In[324]:


A.info()


# In[325]:


##feature scaling

num_col = A._get_numeric_data().columns
col=A.columns
cat_col=list(set(col) - set(num_col))

for col in cat_col:
    le = preprocessing.LabelEncoder()
    A[col] = le.fit_transform(A[col])
A.shape


# In[326]:


##splitting the dataset into the training set and the test set

A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.30, random_state = 0)


# In[327]:


sc = StandardScaler()
A_train = sc.fit_transform(A_train)
A_test = sc.transform(A_test)


# In[328]:


##training the Random Forest model on the training set

rndm_frst=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rndm_frst.fit(A_train , B_train)


# In[329]:


##predecting the test set results
B_rndm = rndm_frst.predict(A_test)


# In[330]:


##making the confusion matrix

cm = confusion_matrix(B_test, B_rndm)
print(cm)


# In[331]:



def tn(B_test,B_rndm): return confusion_matrix(B_test,B_rndm)[0,0]
def tn(B_test,B_rndm): return confusion_matrix(B_test,B_rndm)[0,1]
def tn(B_test,B_rndm): return confusion_matrix(B_test,B_rndm)[1,1]
def tn(B_test,B_rndm): return confusion_matrix(B_test,B_rndm)[1,0]
def tpr(B_test,B_rndm): 
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round((no_tp / (no_tp + no_fn)),2)
def tnr(B_test,B_rndm): 
    no_tn = confusion_matrix(B_test,B_rndm)[0,0]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    return round((no_tn / (no_tn + no_fp)),2)
def fpr(B_test,B_rndm): 
    no_tn = confusion_matrix(B_test,B_rndm)[0,0]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    return round((no_fp / (no_tn + no_fp)),2)
def fnr(B_test,B_rndm): 
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round((no_fn / (no_tp + no_fn)),2)
def Recall(B_test,B_rndm):
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round((no_tp / (no_tp + no_fn)),2)
def Precision(B_test,B_rndm):
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    return round((no_tp / (no_tp + no_fp)),2)
def F1Score(B_test,B_rndm):
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round(((2*no_tp) / ((2*no_tp) + no_fp+no_fn)),2)
def Accuracy(B_test,B_rndm):
    no_tn = confusion_matrix(B_test,B_rndm)[0,0]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round(((no_tp + no_tn) / (no_tp + no_fp + no_fn + no_tn)),2)
def Error(B_test,B_rndm):
    no_tn = confusion_matrix(B_test,B_rndm)[0,0]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round(((no_fp + no_fn) / (no_tp + no_fp + no_fn + no_tn)),2)
def BACC(B_test,B_rndm):
    no_tn = confusion_matrix(B_test,B_rndm)[0,0]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round(0.5*((no_tp / (no_tp + no_fn))+(no_tn / (no_fp + no_tn))),2)
def TSS(B_test,B_rndm):
    no_tn = confusion_matrix(B_test,B_rndm)[0,0]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round((no_tp / (no_tp + no_fn))-(no_fp / (no_fp + no_tn)),2)
def HSS(B_test,B_rndm):
    no_tn = confusion_matrix(B_test,B_rndm)[0,0]
    no_fp = confusion_matrix(B_test,B_rndm)[0,1]
    no_tp = confusion_matrix(B_test,B_rndm)[1,1]
    no_fn = confusion_matrix(B_test,B_rndm)[1,0]
    return round((2*((no_tp * no_tn)-(no_fp * no_fn)))/(((no_tp + no_fn)*(no_fn + no_tn))+((no_tp + no_fp)*(no_fp + no_tn))),2)

    

#def tpr(ans_tp,ans_fn): return ans_tp/(ans_tp+ans_fn)
scoring = {'tp': make_scorer(tp),'tn': make_scorer(tn),'fp': make_scorer(fp),'fn': make_scorer(fn),'tpr': make_scorer(tpr),
           'tnr':make_scorer(tnr),'fpr':make_scorer(fpr),'fnr':make_scorer(fnr),'recall':make_scorer(Recall),
           'precision':make_scorer(Precision),'F1Score':make_scorer(F1Score),
           'Accuracy':make_scorer(Accuracy),'Error':make_scorer(Error),'BACC':make_scorer(BACC),'TSS':make_scorer(TSS),
           'HSS':make_scorer(HSS)}


# In[332]:


## Splits the dataset in the Kfolds
cv = KFold(n_splits=10,random_state=1,shuffle=True)
scores = cross_validate(rndm_frst,A_train,B_train,scoring = scoring,cv=cv)
print(scores)


# In[333]:


column = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Fold 6','Fold 7','Fold 8','Fold 9','Fold 10']
row = ['TP','TN','FP','FN','TPR','TNR','FPR','FNR','Recall','Precision','Error','BACC','TSS','HSS','F1 Score','Accuracy']
 

data_rf = [value for value in scores.values()]
data_rf = data_rf[3:]


# In[334]:


for i in range(len(data_rf)):
    avg = round(sum(data_rf[i])/len(data_rf[i]),2)
    temp = list(data_rf[i])
    temp.append(avg)
    data_rf[i]=temp
print(data_rf)


# In[335]:


import matplotlib.pyplot as plt
fig3,ax3 = mlt.subplots()


fig3.patch.set_visible(False)
ax3.axis('off')


the_table3=ax3.table(cellText=data_svm, rowLabels=row, colLabels=column,loc='center',colWidths=[0.25 for x in column])
the_table3.auto_set_font_size(False)
the_table3.set_fontsize(13)
fig3.tight_layout()
plt.show()


# In[336]:


###SVM


# In[337]:


from sklearn.svm import SVC
from sklearn import svm
svm_clf=SVC(kernel='linear',random_state=0)
svm_clf.fit(A_train,B_train)


# In[338]:


B_pred = classifier.predict(A_test)


# In[339]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(B_test, B_pred)
print(cm)


# In[340]:


#Apply model on test data
y_pred_test=svm_clf.predict(A_test)
y_pred_test


# In[341]:


#Accuracy calculation
from sklearn import metrics
metrics.accuracy_score(B_test,B_test)


# In[342]:


#Create confusion matrix
conf=metrics.confusion_matrix(B_test,B_test)
conf


# In[343]:


#Precision, Recall, FScore
pr_rcl=metrics.precision_recall_fscore_support(B_test,B_test,average='weighted')
pr_rcl


# In[344]:


print(scores.values)


# In[345]:


data_svm = [value for value in scores.values()]
data_svm = data_svm[2:]


# In[346]:


fig3,ax3 = mlt.subplots()


fig3.patch.set_visible(False)
ax3.axis('off')


the_table3=ax3.table(cellText=data_svm, rowLabels=row, colLabels=column,loc='center',colWidths=[0.25 for x in column])
the_table3.auto_set_font_size(False)
the_table3.set_fontsize(13)
fig3.tight_layout()
plt.show()


# In[92]:


##LSTM


# In[290]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense,RNN,LSTM,Activation,Dropout
from keras.models import Sequential


# In[291]:


A.shape


# In[292]:


B.shape


# In[293]:


A_train = np.reshape(A_train, (A_train.shape[0],A_train.shape[1],1))


# In[305]:


A_train = np.reshape(A_train, (A_train.shape[0],A_train.shape[1],1))


# In[306]:


model = Sequential() # initializing model

model.add(LSTM(units=60, return_sequences=False, input_shape=(A_train.shape[1],1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(A_train, B_train, epochs=101, batch_size=4000,validation_split=0.3)

