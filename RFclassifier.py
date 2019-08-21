import os
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import metrics
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

#Baseline model classifier
df1 = df[[ 'Age', 'Family_Diabetes','Family_Size','Systolic_BP1', 'Diastolic_BP1', 'Systolic_BP2',
          'Diastolic_BP2', 'Weight', 'Height', 'BodyMassIndex','GripStrength_left',
          'GripStrength_right', 'Albumin', 'Creatinine', 'Globulin','Glucose', 'Total_Protein',
          'White_blood_cells', 'Red_blood_cells', 'Hemoglobin', 'Blood_platelets',
          'HepatitisA_antibody', 'Insulin', 'Cholesterol', 'Blood_lead','Blood_cadmium',
          'Blood_mercury', 'Blood_selenium', 'Blood_manganese','Vitamin_B12','Diabetes']]
df1.shape #(5011, 31)

#Primitive classifier
"""
df1 = df[[ 'Age', 'Family_Diabetes','Systolic_BP1', 'Systolic_BP2',
          'Weight', 'Height', 'BodyMassIndex','GripStrength_left',
          'GripStrength_right','Diabetes']]
df1.shape #(5011, 10)
"""
#Optimised classifier
"""
df1 = df[[ 'Age', 'Family_Diabetes','Systolic_BP1', 'Systolic_BP2',
          'Weight', 'Height', 'BodyMassIndex','GripStrength_left',
          'GripStrength_right', 'Albumin', 'Creatinine', 'Globulin','Glucose',
          'White_blood_cells', 'Red_blood_cells', 'Hemoglobin', 'Blood_platelets',
          'Insulin', 'Cholesterol','Vitamin_B12','Diabetes']]
df1.shape #(5011, 21)
"""

df1 = df1[df1.Diabetes != 3]
df1.loc[:, 'Diabetes'].replace([2], [0], inplace=True)
df1.dropna(axis=0)
df1.shape #(4885, 31)

X = df1.values[:, 0:30]
X = np.array(X)
X = X[:, ~np.isnan(X).any(axis=0)]
Y = np.array(df1['Diabetes'])
Y = Y[:, ~np.isnan(Y).any(axis=0)]

X.shape #(4885, 30)

Y.shape #(4885, 1)

#Training the Random Forest Classifier
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42,
                                                    stratify=None, shuffle=True)

y_train = np.array(y_train)
y_train.shape #(3272, 1)

clf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy', max_depth= None,
                             max_features = 'auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                             min_impurity_split=None, min_samples_leaf=1, min_samples_split=10,
                             min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=-1, oob_score=False,
                             random_state= None, verbose=0, warm_start=False)

y_train = pd.DataFrame(y_train)
clf.fit(x_train, y_train.values.ravel())

score = clf.score(x_train, y_train)
print('Train',score) #Train 0.9975

score = clf.score(x_test,y_test)
print('Test',score) #Test 0.9256

importances = clf.feature_importances_
print(importances)
print(sum(importances))
sort = sorted(importances, reverse=True)
print(sum(sort[0:4]))

std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)

indices = np.argsort(importances)[::-1]
df2 = df1[[ 'Age', 'Family_Diabetes','Family_Size','Systolic_BP1', 'Diastolic_BP1', 'Systolic_BP2',
          'Diastolic_BP2', 'Weight', 'Height', 'BodyMassIndex','GripStrength_left',
          'GripStrength_right', 'Albumin', 'Creatinine', 'Globulin','Glucose', 'Total_Protein',
          'White_blood_cells', 'Red_blood_cells', 'Hemoglobin', 'Blood_platelets',
          'HepatitisA_antibody', 'Insulin', 'Cholesterol', 'Blood_lead','Blood_cadmium',
          'Blood_mercury', 'Blood_selenium', 'Blood_manganese','Vitamin_B12']]
df2.shape #(4885, 30)

feature_names = df2.columns
print(feature_names)
"""
Index(['Age', 'Family_Diabetes', 'Family_Size', 'Systolic_BP1',
       'Diastolic_BP1', 'Systolic_BP2', 'Diastolic_BP2', 'Weight', 'Height',
       'BodyMassIndex', 'GripStrength_left', 'GripStrength_right', 'Albumin',
       'Creatinine', 'Globulin', 'Glucose', 'Total_Protein',
       'White_blood_cells', 'Red_blood_cells', 'Hemoglobin', 'Blood_platelets',
       'HepatitisA_antibody', 'Insulin', 'Cholesterol', 'Blood_lead',
       'Blood_cadmium', 'Blood_mercury', 'Blood_selenium', 'Blood_manganese',
       'Vitamin_B12'],
      dtype='object')
"""

print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
"""
Feature ranking:
1. feature Glucose (0.283641)
2. feature Family_Diabetes (0.077572)
3. feature Age (0.070841)
4. feature BodyMassIndex (0.051930)
5. feature Albumin (0.049400)
6. feature Weight (0.036071)
7. feature Systolic_BP1 (0.030398)
8. feature Blood_platelets (0.029481)
9. feature GripStrength_left (0.027068)
10. feature GripStrength_right (0.025609)
11. feature Systolic_BP2 (0.023352)
12. feature Creatinine (0.022871)
13. feature Vitamin_B12 (0.022418)
14. feature Red_blood_cells (0.022165)
15. feature White_blood_cells (0.021382)
16. feature Globulin (0.021162)
17. feature Cholesterol (0.020617)
18. feature Height (0.019889)
19. feature Hemoglobin (0.018720)
20. feature Total_Protein (0.018558)
21. feature Insulin (0.015159)
22. feature Diastolic_BP2 (0.014773)
23. feature Diastolic_BP1 (0.013900)
24. feature Blood_lead (0.013294)
25. feature Blood_selenium (0.013204)
26. feature Family_Size (0.011496)
27. feature Blood_manganese (0.009546)
28. feature Blood_mercury (0.009014)
29. feature Blood_cadmium (0.006059)
30. feature HepatitisA_antibody (0.000411)

"""

#Plotting the rankings
plt.subplots(figsize=(15,8))
sns.barplot(importances, feature_names, palette='inferno')


#Testing
y_predRF = clf.predict(x_test)

print("Precision score:")
print(round(precision_score(y_test, y_predRF, average='binary'), 3)) #0.694

print("\nAccuracy score:")
print(round(accuracy_score(y_test, y_predRF), 4)) #0.9256

cm = confusion_matrix(y_test, y_predRF)
print(cm)
"""
[[1432   23]
 [  87   71]]
"""
