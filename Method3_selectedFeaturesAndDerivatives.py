import pandas as pd 
from Functions import *
from DataDefined import *


df = pd.read_csv('Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')

## add derivatives
#Column names we want the derivative of
to_diff_col = ['Steering_wheel_speed', 'Accelerator_Pedal_value', 'Vehicle_speed']

df = add_derivatives(df, to_diff_col)


df, label_mappings = preprocessing(df, one_val_col, to_be_scaled_col, two_val_col, some_val_num_col, some_val_cat_col)


