import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Functions import *
from DataDefined import *

df = pd.read_csv('Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')

# Dropping the columns with only one value
df = df.drop(df[one_val_col],axis=1)


#plot the to be normalised data
#im sorry for how fucked up this one is :)
plot_distr(df, to_be_scaled_col, 3)

## if we want to fumble with the distribution a bit, unblock the following code:
#col_to_change = to_be_scaled_col #for now I've set them to all of them, change this if you don't want this
#df[col_to_change] = np.log1p(df[col_to_change])

#### Transforming data

##Scaling the numerically distributed columns to a normal distribution between -1 and 1
df = scale_to_norm_dist(df, to_be_scaled_col)

##Scaling the two valued columns to 0 and 1
df = scale_to_two_val(df, two_val_col)

##Scaling the values with only a couple numerical values between 0 and 1
#Oke dit werkt dus nu niet en ik heb geen idee waarom
for col in some_val_num_col:
    df[col] = df[col]/(max(df[col]))

##Encoding the categorical numbers between 0 and 1
df, label_mappings = encode_scale(df, some_val_cat_col)

## Classifying the rides, with ride 1 gets label 0.
df = addRideNumbers(df)


## Model training

rf = RandomForestClassifier(random_state=42)


#relevant_features = ['Vehicle_speed', 'Acceleration_speed_-_Longitudinal', 'Acceleration_speed_-_Lateral', 'Indication_of_brake_switch_ON/OFF','Steering_wheel_speed', 'Steering_wheel_angle', 'Flywheel_torque'] 

#X = df[df[relevant_features]]
#Y = df['Class']

X = df.drop(['Class'], axis=1)
Y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

print(classification_report(y_test, predictions))