import pandas as pd 
import numpy as np
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
for col in some_val_num_col:
    df[col] = df[col]/max(df[col])

# 'PathOrder' is in this list but I am not sure we should keep it in 
##Encoding the categorical numbers between 0 and 1
df, label_mappings = encode_scale(df, some_val_cat_col)
