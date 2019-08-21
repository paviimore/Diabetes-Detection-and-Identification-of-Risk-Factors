import os
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import optimizers
from keras import metrics
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import log_loss, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("data_clean.csv", sep=';')
df = df.drop(['Unnamed: 0'],axis=1)
cols = list(df)
df.shape #(5011, 48)

df1 = df[['Age', 'Family_Diabetes','Family_Size','Systolic_BP1', 'Diastolic_BP1', 'Systolic_BP2',
          'Diastolic_BP2', 'Weight', 'Height', 'BodyMassIndex','GripStrength_left',
          'GripStrength_right', 'Albumin', 'Creatinine', 'Globulin','Glucose', 'Total_Protein',
          'White_blood_cells', 'Red_blood_cells', 'Hemoglobin', 'Blood_platelets',
          'HepatitisA_antibody', 'Insulin', 'Cholesterol', 'Blood_lead','Blood_cadmium',
          'Blood_mercury', 'Blood_selenium', 'Blood_manganese','Vitamin_B12','Diabetes']]
df1.shape #(5011, 31)

df1 = df1[df1.Diabetes != 3]
df1.loc[:, 'Diabetes'].replace([2], [0], inplace=True)
df1.shape #(4885, 31)

df1.dropna(axis=0)
df1.shape #(4885, 31)

X = df1.values[:, 0:30]
X = np.array(X)
X = X[:, ~np.isnan(X).any(axis=0)]
Y = np.array(df1['Diabetes'])
Y = Y[:, ~np.isnan(Y).any(axis=0)]

X.shape #(4885, 30)
Y.shape #(4885, 1)

#Building the Neural Network classifier
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=30))
classifier.add(Dropout(0.5))
#Second  Hidden Layer
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dropout(0.5))
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dropout(0.5))
#classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
#classifier.add(Dropout(0.5))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
adam=optimizers.Adam(lr=0.0001, amsgrad=False)
classifier.compile(optimizer = adam , loss='binary_crossentropy', metrics =['accuracy'])
classifier.summary()

#Fitting the data to the training dataset

early_stp = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=0, mode='auto',
                        baseline=None, restore_best_weights=True)
history = classifier.fit(X_train, y_train, validation_split=0.33, epochs=200, batch_size=10,
                         verbose=0, callbacks=[early_stp])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(' model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('NNacc4.png', dpi=300, bbox_inches='tight')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('NNloss4.png', dpi=300, bbox_inches='tight')
plt.show()

eval_model = classifier.evaluate(X_test, y_test) #[0.1655, 0.9405]

eval_model=classifier.evaluate(X_train, y_train) #[0.1584, 0.9482]

y_pred=classifier.predict(X_test)

#Comparing predicted values to the threshold = 0.5
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0

cm = confusion_matrix(y_test, y_pred)
print(cm)
"""
[ [101(TP)   24(FP)]
 [72(FN)   1416(TN)] ]
"""
print("Precision score:")
print(round(precision_score(y_test, y_pred, average='binary'), 3)) #0.808

print("Recall score:")
print(round(recall_score(y_test, y_pred, average='binary'), 3)) #0.584

print("\nAccuracy score:")
print(round(accuracy_score(y_test, y_pred), 4)) #0.9405


#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,'g',label='Optimised Network AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('ROC.png', dpi=300, bbox_inches='tight')
plt.show()
