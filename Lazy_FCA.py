import pandas as pd
import numpy as np
import timeit
# In[2]:
'''
data = pd.read_csv('2008_200_300.csv')
data = data.dropna()
data = data.drop(data[data.Better == 0].index)

data['Age_c'] = pd.qcut(data['Age'], 5)
data['Leuc_c'] = pd.qcut(data['Leuc'], 5)
data['Leber_c'] = pd.cut(data['Leber'], 8)
data['Milz_c'] = pd.cut(data['Milz'], 5)
data['height_c'] = pd.qcut(data['height'], 5)
data['weight_c'] = pd.qcut(data['weight'], 5)

data = data.drop(['ID','Age','Leuc','Leber','Milz','height','weight'], 1)

data = pd.get_dummies(data, columns = ['Sex', 'Immun', 'CNS', 'Mediastinum', 'Zytogen', 'Region', 'Geb_month', 'Diag_month', 'syndrome', 'Age_c', 'Leuc_c', 'Leber_c', 'Milz_c', 'height_c', 'weight_c'])

data.loc[data.Better == 300, 'Better'] = 'positive'
data.loc[data.Better == 200, 'Better'] = 'negative'

data.to_csv('2008_200_300_new.csv',index=False)

newdata = pd.read_csv('2008_200_300_new.csv')

from sklearn.cross_validation import KFold

kf = KFold(len(newdata), n_folds=10, shuffle=True, random_state=None)
for k, (train, test) in enumerate(kf):
    newdata.iloc[train].to_csv('train'+str(k+1)+'.csv',index=False)
    newdata.iloc[test].to_csv('test'+str(k+1)+'.csv',index=False)
'''
# In[3]:

def parse_file(name):
    df = pd.read_csv(name, sep=',')
    df = df.replace(to_replace='positive', value=1)
    df = df.replace(to_replace='negative', value=0)
    y = np.array(df['SeriousDlqin2yrs'])
    del df['SeriousDlqin2yrs']
    return np.array(df).astype(int), y

# In[11]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# In[13]:
def pred_data(i, part):
    X_train, y_train = parse_file('train' + str(i) + '.csv')
    X_test, y_test = parse_file('test' + str(i) + '.csv')
    X_train_pos = X_train[y_train == 1]
    X_train_neg = X_train[y_train == 0]
    
    y_pred = [] 
    k=1   
    for test_obj in X_test:
        print(k*100/len(X_test))
        k+=1
        pos = 0
        neg = 0
    
        
        for pos_obj in X_train_pos:
            if np.sum(test_obj == pos_obj)> int(len(pos_obj)*part):
                pos += 1
                
        for neg_obj in X_train_neg:
            if np.sum(test_obj == neg_obj)  > int(len(neg_obj)*part):
                neg += 1       
                
                
        for pos_obj in X_train_pos:
            for neg_obj in X_train_neg:
                if (neg_obj>=np.minimum(pos_obj,test_obj)).all() and (neg_obj<=np.maximum(pos_obj,test_obj)).all():
                    pos += 1
                else: 
                    break
                        
        print(pos)
                
        for neg_obj in X_train_neg:
            for pos_obj in X_train_pos:
                if (pos_obj>=np.minimum(test_obj,neg_obj)).all() and (pos_obj<=np.maximum(test_obj,neg_obj)).all():
                    neg += 1 
                else:
                    break
                
        print(neg)
            
        pos = pos / float(len(X_train_pos))

        neg = neg / float(len(X_train_neg))
        if (pos > neg):
            y_pred.append(1)
        else:
            y_pred.append(0)
            
    y_pred = np.array(y_pred)
    #print y_pred
    
    '''
    TP = np.sum(y_test * y_pred)
    TN = np.sum(y_test + y_pred == 0)
    FP = np.sum((y_test  == 0) * (y_pred == 1))
    FN = np.sum((y_test  == 1) * (y_pred == 0))
    TPR = float(TP) / np.sum(y_test == 1)
    TNR = float(TN) / np.sum(y_test == 0)
    FPR = float(FP) / (TP + FN)
    NPV = float(TN) / (TN + FN)
    FDR = float(FP) / (TP + FP)
    '''
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print("Dataset {}".format(i))
    #print "True Positive: {}\nTrue Negative: {}\nFalse Positive: {}\nFalse Negative: {}\nTrue Positive Rate: {}\nTrue Negative Rate: {}\n\
    #Negative Predictive Value: {}\nFalse Positive Rate: {}\nFalse Discovery Rate: {}\nAccuracy: {}\nPrecision: {}\nRecall: {}".format(TP, TN, FP, FN, TPR, TNR, FPR, NPV, FDR, acc, prec, rec)
    print("Accuracy: {}\nPrecision: {}\nRecall: {}".format(acc, prec, rec))
    print("===========")
    print(i)
    return(acc, prec, rec)
acc1=[]
rec1=[]
prec1=[]

start = timeit.default_timer()
for i in range(0, 1):
    acc, prec, rec = pred_data(i+1, 0.91)
    acc1.append(acc)
    prec1.append(prec)
    rec1.append(rec)
stop = timeit.default_timer()
time = stop - start
print("Accuracy: {}\nPrecision: {}\nRecall: {}\nTime: {}".format(sum(acc1)/len(acc1), sum(prec1)/len(prec1),sum(rec1)/len(rec1), time))