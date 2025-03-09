import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Functions import *
from DataDefined import *

df = pd.read_csv('Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')

df, label_mappings = preprocessing(df, one_val_col, to_be_scaled_col, two_val_col, some_val_num_col, some_val_cat_col)


## Model training

relevant_features = ['Vehicle_speed', 'Acceleration_speed_-_Longitudinal', 'Acceleration_speed_-_Lateral', 'Indication_of_brake_switch_ON/OFF','Steering_wheel_speed', 'Steering_wheel_angle', 'Flywheel_torque'] 

#X = df.drop(['Class'], axis=1)
#Y = df['Class']
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

X_train, X_test, y_train, y_test = split_train_test_self(df, relevant_features, )

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

print(classification_report(y_test, predictions))