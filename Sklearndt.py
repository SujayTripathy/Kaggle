from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

###########READING THE DATA########################
training=pd.read_csv("train_final.csv")                                                     ##Reading the data
testing=pd.read_csv("test_final.csv")


##########TRAINING DATA##########################
# age_Scale=preprocessing.scale(training["age"],copy=False)                                                  ##Normalizing the Numeric features
# training['age']=age_Scale
# fnl_Scale=preprocessing.scale(training["fnlwgt"],copy=False)
# training['fnlwgt']=fnl_Scale
# edu_Scale=preprocessing.scale(training["education.num"],copy=False)
# training['education.num']=edu_Scale
# hours_Scale=preprocessing.scale(training["hours.per.week"],copy=False)   
# training['hours.per.week']=hours_Scale                                  
# gain_Scale=preprocessing.robust_scale(training["capital.gain"],copy=False)                                 ##Normalizing the numeric features with outliers
# training['capital.gain']=gain_Scale
# loss_Scale=preprocessing.robust_scale(training["capital.loss"],copy=False)
# training['capital.loss']=loss_Scale
# training['workclass'].replace('[?]',str(training["workclass"].mode()[0]),inplace=True,regex=True)       ## Replacing the unknown values with the mode of the feature
# training['occupation'].replace('[?]',str(training["occupation"].mode()[0]),inplace=True,regex=True)
le=preprocessing.LabelEncoder()
le.fit(training['workclass'].unique())
training['workclass']=le.transform(training['workclass'])
le.fit(training['occupation'].unique())
training['occupation']=le.transform(training['occupation'])
le.fit(training['education'].unique())
training['education']=le.transform(training['education'])
le.fit(training['marital.status'].unique())
training['marital.status']=le.transform(training['marital.status'])
le.fit(training['relationship'].unique())
training['relationship']=le.transform(training['relationship'])
le.fit(training['race'].unique())
training['race']=le.transform(training['race'])
le.fit(training['sex'].unique())
training['sex']=le.transform(training['sex'])
le.fit(training['native.country'].unique())
training['native.country']=le.transform(training['native.country'])


###############################TESTING DATA###################################
# age_Scale=preprocessing.scale(training["age"],copy=False)                                                  ##Normalizing the Numeric features
# training['age']=age_Scale
# fnl_Scale=preprocessing.scale(training["fnlwgt"],copy=False)
# training['fnlwgt']=fnl_Scale
# edu_Scale=preprocessing.scale(training["education.num"],copy=False)
# training['education.num']=edu_Scale
# hours_Scale=preprocessing.scale(training["hours.per.week"],copy=False)   
# training['hours.per.week']=hours_Scale                                  
# gain_Scale=preprocessing.robust_scale(training["capital.gain"],copy=False)                                 ##Normalizing the numeric features with outliers
# training['capital.gain']=gain_Scale
# loss_Scale=preprocessing.robust_scale(training["capital.loss"],copy=False)
# training['capital.loss']=loss_Scale
# testing['workclass'].replace('[?]',str(testing["workclass"].mode()[0]),inplace=True,regex=True)       ## Replacing the unknown values with the mode of the feature
# testing['occupation'].replace('[?]',str(testing["occupation"].mode()[0]),inplace=True,regex=True)
le=preprocessing.LabelEncoder()
le.fit(testing['workclass'].unique())
testing['workclass']=le.transform(testing['workclass'])
le.fit(testing['occupation'].unique())
testing['occupation']=le.transform(testing['occupation'])
le.fit(testing['education'].unique())
testing['education']=le.transform(testing['education'])
le.fit(testing['marital.status'].unique())
testing['marital.status']=le.transform(testing['marital.status'])
le.fit(testing['relationship'].unique())
testing['relationship']=le.transform(testing['relationship'])
le.fit(testing['race'].unique())
testing['race']=le.transform(testing['race'])
le.fit(testing['sex'].unique())
testing['sex']=le.transform(testing['sex'])
le.fit(testing['native.country'].unique())
testing['native.country']=le.transform(testing['native.country'])


#####Tree model formation and prediction#####
# clf=tree.DecisionTreeClassifier()
# clf.fit(training.drop(['income>50K'],axis=1),training['income>50K'])
#results=clf.predict(testing.drop(['ID'],axis=1))
# probs= clf.predict_proba(testing.drop(['ID'],axis=1))
# probs=probs[:,1]


#####EnsembleForest#######
param_grid = {
        'n_estimators': range(100, 200, 2),
        'max_depth': range(10, 50, 2),
    }
rf=RandomForestClassifier(random_state=42)
clf=RandomizedSearchCV(estimator=rf,param_distributions=param_grid,cv=5,n_iter=100,verbose=2,random_state=42,n_jobs=-1)
clf.fit(training.drop(['income>50K'],axis=1),training['income>50K'])
probs= clf.predict_proba(testing.drop(['ID'],axis=1))

#####ID Column###
ID=[]
for i in range(23842):
    ID.append(i+1)


# # ###Forming Dataframe and converting to CSV file####
d={'ID':ID,"Prediction":probs[:,1]}                                                            
df=pd.DataFrame(d)
df.to_csv("Submission21.csv",index=False)
