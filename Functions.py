import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

"""
Scale given columns that are numerically distributed to a
normal distribution with m=0 and sd=1.

@param df             dataframe
@param col_names      given colum names of the df, on which 
                       normalisation has to happen

@return df            return the updated df
"""
def scale_to_norm_dist(df, col_names):
    sc = StandardScaler()
    for col in col_names:
        df[col] = sc.fit_transform(df[col].values.reshape(-1, 1))
    return df

"""
For columns with exactly 2 values in total, change those
values to 0 and 1.

@param df             dataframe
@param col_names      given colum names of the df, on which 
                       normalisation has to happen

@return df            return the updated df
"""
def scale_to_two_val(df, col_names):
    for col in col_names:
        val = df[col].unique()
        val.sort()
        df[col] = [0 if i == val[0] else 0 for i in df[col]]

"""
For columns with numerical categories, encode them and
then scale them relative to eachother on a scale from
0 to 1.

@param df                 dataframe
@param col_names          given colum names of the df, on which 
                           normalisation has to happen

@return df                return the updated df
@return label_mappings    save the label_mappings, so you can 
                           look back to which label is which
                           category
"""
def encode_scale(df, col_names):
    label_mappings = {}
    for col in col_names:
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        if col != 'Class':
            df[col] = df[col]/max(df[col])
    return df, label_mappings

"""
Plot the histplot of the distribution of given columns
of the df.

@param df             dataframe
@param col_names      given colum names of the df, on which 
                       normalisation has to happen
@param nr_col=3       nr of columns in the subplots

@output               histplots for all the given columns
"""
def plot_distr(df, columns, nr_col):
    nr_row = len(columns)//nr_col
    
    fig, axs = plt.subplots(nr_row, nr_col, figsize=(20,12))  # 12 rows, 3 columns

    for i in range(0, len(columns)):
        col_loc = i%3
        row_loc = i//3

        plot = sns.histplot(df[columns[i]], kde=False, bins=30, ax=axs[row_loc, col_loc])
        axs[row_loc, col_loc].set_title(columns[i])

    plt.tight_layout()
    plt.show()


"""
Adds a classification to which ride a certain row belongs to. Every
driver has driven multiple rides of the same route, so this function
adds a column for every row to specificy which ride it was, in order 
to use this to better split the data for model testing and training
later.

@param df             dataframe

@return df            dataframe with the added column
"""
def addRideNumbers(df):
    drivers = df['Class'].unique()

    df_drivers_to_concat = []

    for driver in drivers:
        df_driver = df[df['Class'] == driver]

        list_ride_nr = []  
        current_ride = -1

        for timestamp in df_driver['Time(s)']:
            if timestamp == 1:
                current_ride += 1
                list_ride_nr.append(current_ride)
        df_driver.insert(49, 'Ride number', list_ride_nr)

        df_drivers_to_concat.append(df_driver)

    return pd.concat(df_drivers_to_concat, axis=0, ignore_index=True)

"""
Splits the data into a training and test set. The test set 
contains a full ride of every single one of the drivers. The 
number of the ride is given with the index.

@param df             dataframe
@param features       list of column names of the dataframe 
                      we want to keep, + Class and Ride number
@param index          indicates which ride nr is taken 
                      for the test set

@return X_train       train data
@return X_test        test data
@return y_train       train labels
@return y_test        test labels
"""
def split_train_test_self(df, features, index):
    features.extend(['Class', 'Ride number'])
    
    df_rel = df[features].copy(deep=False)
    df_testset_to_concat = []
    drivers = df['Class'].unique()

    #split it up into two dataframes (train + test)
    for driver in drivers:
        #create test set
        test_set = df_rel.loc[(df_rel['Class'] == driver) & (df_rel['Ride number'] == index)]
        #add test set to df_testset
        df_testset_to_concat.append(test_set)
        #drop testset from og df
        df_rel.drop(df_rel.loc[(df_rel['Class'] == driver) & (df_rel['Ride number'] == index)].index, inplace=True)

    df_testset = pd.concat(df_testset_to_concat, axis=0, ignore_index=True)
    
    #sample both of them with frac=1
    df_testset = df_testset.sample(frac = 1, random_state=42)
    df_trainingset = df_rel.sample(frac = 1, random_state=42)

    #define X_train, X_test, y_train, y_test
    y_train = df_trainingset['Class'].copy(deep=False)
    X_train = df_trainingset.drop(['Class', 'Ride number'], axis=1)
    
    y_test = df_testset['Class'].copy(deep=False)
    X_test = df_testset.drop(['Class', 'Ride number'], axis=1)

    return X_train, X_test, y_train, y_test