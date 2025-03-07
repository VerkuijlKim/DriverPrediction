from DataDefined import *
from Functions import *
import pandas as pd

df = pd.read_csv('Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')


# Dropping the columns with only one value
df = df.drop(df[one_val_col],axis=1)


# Dropping PathOrder, as we can't use this information because we don't know what it stands for
df = df.drop('PathOrder', axis=1)


##Scaling the numerically distributed columns to a normal distribution between -1 and 1
df = scale_to_norm_dist(df, to_be_scaled_col)

##Scaling the two valued columns to 0 and 1
df = scale_to_two_val(df, two_val_col)


for col in some_val_num_col:
    df[col] = df[col]/df[col].max()