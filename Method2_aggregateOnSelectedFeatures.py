import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Functions import *
from DataDefined import *

df = pd.read_csv('Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')

df, label_mappings = preprocessing(df, one_val_col, to_be_scaled_col, two_val_col, some_val_num_col, some_val_cat_col)

relevant_features = ['Vehicle_speed', 'Acceleration_speed_-_Longitudinal', 'Acceleration_speed_-_Lateral', 
                     'Indication_of_brake_switch_ON/OFF','Steering_wheel_speed', 'Steering_wheel_angle', 
                     'Flywheel_torque', 'Class', 'Ride number']


df_aggregated = aggregate(df, relevant_features)

traindataset = df_aggregated[df_aggregated['Class'] != 2]
testdataset = df_aggregated[df_aggregated['Class'] == 2]

X_train = traindataset.drop(['Class', 'Ride number'], axis=1)
y_train = traindataset['Class']

X_test = testdataset.drop(['Class', 'Ride number'], axis=1)
y_test = testdataset['Class']

oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)

oc_svm.fit(X_train)

#y_pred = oc_svm.predict(X_test)
y_pred_train = oc_svm.predict(X_train)
y_pred = [0 if x == -1 else 1 for x in y_pred]
y_pred_train = [0 if x == -1 else 1 for x in y_pred_train]

y_transf_train = [1] * len(y_train)
y_transf_test = [0] * len(y_test)

print(classification_report(y_transf_test, y_pred))