import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



def preprocessing(df, one_val_col, to_be_scaled_col, two_val_col, some_val_num_col, some_val_cat_col):
    """
    Performs all of the normalisation functions on the
    columns they apply to.

    @param df                   dataframe
    @param one_val_col          column names of columns with only 1 value
    @param to_be_scaled_col     column names of columns that have to be scaled
                                to a standard normal distribution
    @param two_val_col          column names of columns with exactly 2 values
    @param some_val_num_col     column names of columns with a couple of distinct
                                values
    @param some_val_cat_col     column names of columns with categorical values

    @return df                  return the updated df
    @return label_mappings      returns the mappings of the LabelEncoder, to be
                                able to access them later
    """
    # Dropping the columns with only one value
    df = df.drop(df[one_val_col],axis=1)
    # Dropping PathOrder, as we can't use this information because we don't know what it stands for
    df = df.drop('PathOrder', axis=1)

    #plot the to be scaled columns
    plot_distr(df, to_be_scaled_col, 3)

    ## if we want to fumble with the distribution a bit, unblock the following code:
    #col_to_change = to_be_scaled_col #for now I've set them to all of them, change this if you don't want this
    #df[col_to_change] = np.log1p(df[col_to_change])

    ##Scaling the numerically distributed columns to a normal distribution between -1 and 1
    df = scale_to_norm_dist(df, to_be_scaled_col)
    ##Scaling the two valued columns to 0 and 1
    df = scale_to_two_val(df, two_val_col)

    ##Scaling the values with only a couple numerical values between 0 and 1
    for col in some_val_num_col:
        df[col] = df[col]/df[col].max()
    ##Encoding the categorical numbers between 0 and 1
    df, label_mappings = encode_scale(df, some_val_cat_col)

    ## Classifying the rides, with ride 1 gets label 0.
    df = addRideNumbers(df)

    return df, label_mappings


def scale_to_norm_dist(df, col_names):
    """
    Scale given columns that are numerically distributed to a
    normal distribution with m=0 and sd=1.

    @param df           dataframe
    @param col_names    given colum names of the df, on which 
                        normalisation has to happen

    @return df          return the updated df
    """
    sc = StandardScaler()
    for col in col_names:
        df[col] = sc.fit_transform(df[col].values.reshape(-1, 1))
    return df


def scale_to_two_val(df, col_names):
    """
    For columns with exactly 2 values in total, change those
    values to 0 and 1.

    @param df           dataframe
    @param col_names    given colum names of the df, on which 
                        normalisation has to happen

    @return df          return the updated df
    """
    for col in col_names:
        val = df[col].unique()
        val.sort()
        df[col] = [0 if i == val[0] else 1 for i in df[col]]
    return df


def encode_scale(df, col_names):
    """
    For columns with numerical categories, encode them and
    then scale them relative to eachother on a scale from
    0 to 1.

    @param df                   dataframe
    @param col_names            given colum names of the df, on which 
                                normalisation has to happen

    @return df                  return the updated df
    @return label_mappings      save the label_mappings, so you can 
                                look back to which label is which
                                category
    """
    label_mappings = {}
    for col in col_names:
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        if col != 'Class':
            df[col] = df[col]/max(df[col])
    return df, label_mappings


def plot_distr(df, columns, nr_col):
    """
    Plot the histplot of the distribution of given columns
    of the df.

    @param df           dataframe
    @param col_names    given colum names of the df, on which 
                        normalisation has to happen
    @param nr_col=3     nr of columns in the subplots

    @output             histplots for all the given columns
    """
    nr_row = len(columns)//nr_col
    
    fig, axs = plt.subplots(nr_row, nr_col, figsize=(20,12))  # 12 rows, 3 columns

    for i in range(0, len(columns)):
        col_loc = i%3
        row_loc = i//3

        plot = sns.histplot(df[columns[i]], kde=False, bins=30, ax=axs[row_loc, col_loc])
        axs[row_loc, col_loc].set_title(columns[i])

    plt.tight_layout()
    plt.show()



def addRideNumbers(df):
    """
    Adds a classification to which ride a certain row belongs to. Every
    driver has driven multiple rides of the same route, so this function
    adds a column for every row to specificy which ride it was, in order 
    to use this to better split the data for model testing and training
    later.

    @param df             dataframe

    @return df            dataframe with the added column
    """
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
        df_driver.insert(len(df.columns), 'Ride number', list_ride_nr)

        df_drivers_to_concat.append(df_driver)

    return pd.concat(df_drivers_to_concat, axis=0, ignore_index=True)




def add_derivatives(df, col_names):
    """
    Adds the derivatives of the given columns 
    to the dataframe

    @param df           dataframe
    @param col_names    column names of the df for which
                        to add the derivatives

    @return df          return updated df
    """
    for col in col_names:
        new_col_name = col + '_derivative'
        df[new_col_name] = df[col].diff()
    return df


def split_train_test_self(df, features, index):
    """
    Splits the data into a training and test set. The test set 
    contains a full ride of every single one of the drivers. The 
    number of the ride is given with the index.

    @param df           dataframe
    @param features     list of column names of the dataframe 
                        we want to keep, + Class and Ride number
    @param index        indicates which ride nr is taken 
                        for the test set

    @return X_train     train data
    @return X_test      test data
    @return y_train     train labels
    @return y_test      test labels
    """
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

def aggregate(df, column_names):
    """
    Adds features (mean, std, min, max) to selected
    columns of the dataframe. Calculates these values
    per driver per ride. Returns only the features 
    of the selected columns, as the features 
    summarize the dataset.

    @param df               dataframe
    @param column_names     list of column names of the dataframe 
                            we want to calculate the values for, 
                            + Class and Ride number

    @return df_aggregated   dataframe of added features
    """
    df_rel_feat = df[column_names].copy(deep=False)

    # voor alle columns average and std
    df_to_concat = []

    for driver in df_rel_feat['Class'].unique():
        df_rel_driver = df_rel_feat[df_rel_feat['Class'] == driver]
        for ridenr in df_rel_driver['Ride number'].unique():
            df_rel_driver_rel_nr = df_rel_driver[df_rel_driver['Ride number'] == ridenr]
            col_names = []
            col_values = []
            for col in column_names:
                if col != 'Class' and col != 'Ride number':
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

    return df_aggregated