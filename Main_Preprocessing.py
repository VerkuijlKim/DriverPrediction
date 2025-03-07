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

#plot the to be scaled columns
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

relevant_features = ['Vehicle_speed', 'Acceleration_speed_-_Longitudinal', 'Acceleration_speed_-_Lateral', 'Indication_of_brake_switch_ON/OFF','Steering_wheel_speed', 'Steering_wheel_angle', 'Flywheel_torque'] 

#X = df.drop(['Class'], axis=1)
#Y = df['Class']
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

X_train, X_test, y_train, y_test = split_train_test_self(df, relevant_features, )

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

print(classification_report(y_test, predictions))

###############
#aggegate data

relevant_features = ['Vehicle_speed', 'Acceleration_speed_-_Longitudinal', 'Acceleration_speed_-_Lateral', 
                     'Indication_of_brake_switch_ON/OFF','Steering_wheel_speed', 'Steering_wheel_angle', 
                     'Flywheel_torque', 'Class', 'Ride number']

df_rel_feat = df[relevant_features].copy(deep=False)

rel_features = ['Vehicle_speed', 'Acceleration_speed_-_Longitudinal', 'Acceleration_speed_-_Lateral', 
                     'Indication_of_brake_switch_ON/OFF','Steering_wheel_speed', 'Steering_wheel_angle', 
                     'Flywheel_torque']

# voor alle columns average and std
df_to_concat = []

for driver in df_rel_feat['Class'].unique():
    df_rel_driver = df_rel_feat[df_rel_feat['Class'] == driver]
    for ridenr in df_rel_driver['Ride number'].unique():
        df_rel_driver_rel_nr = df_rel_driver[df_rel_driver['Ride number'] == ridenr]
        col_names = []
        col_values = []
        for col in rel_features:
            #mean
            curr_avr = df_rel_driver_rel_nr[col].mean()
            #std
            curr_std = df_rel_driver_rel_nr[col].std()
            #min
            curr_min = df_rel_driver_rel_nr[col].min()
            #max
            curr_max = df_rel_driver_rel_nr[col].max()
            col_names.extend([col + ' mean', col + ' std', col + ' min', col + ' max'])
            col_values.extend([curr_avr, curr_std, curr_min, curr_max])
        col_names.extend(['Class', 'Ride number'])
        col_values.extend([(driver), (ridenr)])
        df_aggregated_driver_nr = pd.DataFrame(data=[col_values], columns=col_names)
        df_aggregated_driver_nr.head(5)
        df_to_concat.append(df_aggregated_driver_nr)

df_aggregated = pd.concat(df_to_concat, axis=0, ignore_index=True)

df_aggregated.head(15)

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