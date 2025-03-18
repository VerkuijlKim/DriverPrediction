
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim


###############################################################################
# Functions for data preprocessing
###############################################################################

def addRideNumbers(df):
    """ 
    Assigns a ride number to each driver based on when the timer restarts to 1.
    """
    drivers = df['Class'].unique()
    df_drivers_to_concat = []

    for driver in drivers:
        df_driver = df[df['Class'] == driver].copy()
        df_driver['Ride number'] = (df_driver['Time(s)'] == 1).cumsum()
        df_drivers_to_concat.append(df_driver)

    return pd.concat(df_drivers_to_concat, axis=0, ignore_index=True)


def normalize_data(train_df, test_df, columns_to_normalize):
    """
    Normalizes specified columns in train and test data using StandardScaler.
    """
    scaler = StandardScaler()
    # Fit on train data and transform both train and test data.
    train_df[columns_to_normalize] = scaler.fit_transform(train_df[columns_to_normalize])
    test_df[columns_to_normalize] = scaler.transform(test_df[columns_to_normalize])
    
    return train_df, test_df


def one_hot_encode(train_df, test_df, columns_to_encode):
    """
    Applies one-hot encoding to specified categorical columns in train and test data.
    """
    for col in columns_to_encode:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        # Fit on train data and transform both train and test data.
        encoded_train = encoder.fit_transform(train_df[[col]])
        encoded_test = encoder.transform(test_df[[col]])
        
        # Change column names.
        encoded_columns = encoder.get_feature_names_out([col])
        train_encoded_df = pd.DataFrame(encoded_train, columns=encoded_columns, index=train_df.index)
        test_encoded_df = pd.DataFrame(encoded_test, columns=encoded_columns, index=test_df.index)
        
        # Drop original column and merge encoded columns.
        train_df = train_df.drop(columns=[col]).join(train_encoded_df)
        test_df = test_df.drop(columns=[col]).join(test_encoded_df)

    return train_df, test_df


def label_encode(train_df, test_df, columns_to_encode):
    """
    Applies label encoding on the specified categorical columns in train and test data.
    """
    encoders = {}

    for col in columns_to_encode:
        encoder = LabelEncoder()
        train_df[col] = encoder.fit_transform(train_df[col])
        encoders[col] = encoder

    for col in columns_to_encode:
        test_df[col] = encoders[col].transform(test_df[col]) 

    return train_df, test_df


###############################################################################
# New features
###############################################################################


def driver_features(df, columns):
    """
    Creates a new DataFrame with the mean, std and max for each driver and ride number 
    for the specified columns.
    """
    aggregation = {
        col: ['mean', 'std', 'max'] for col in columns
    }
    df_aggregated = df.groupby(['Class', 'Ride number'])[columns].agg(aggregation).reset_index()
    df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]

    return df_aggregated


def add_delta(df, columns):
    """
    This function adds delta (difference) columns for each specified column.
    It also sets the delta to 0 when the timestamp is 1, which indicates a new ride/driver.
    """
    for col in columns:
        delta_col_name = f"{col}_delta"
        df[delta_col_name] = df[col].diff() 
        df.loc[df['Time(s)'] == 1, delta_col_name] = 0.0
    
    return df


###############################################################################
# Function for splitting the data
###############################################################################


def split_train_test_ood(df, driver_nr, frac):
    """
    For out-of-distribution (OOD).
    Moves one entire driver's data into the test set.
    Additionally, includes samples from other drivers in the test set.
    """
    test_df = df[df['Class'] == driver_nr]
    remaining_df = df[df['Class'] != driver_nr]
    sampled_test_df = remaining_df.groupby('Class').sample(frac=frac, random_state=42)
    test_df = pd.concat([test_df, sampled_test_df])
    train_df = remaining_df.drop(sampled_test_df.index)

    return train_df, test_df

def split_train_test_ood_train1class(df, driver_nr_ood, driver_nr_train, frac):
    """
    Moves one entire driver's data into the test set.
    Includes samples from one driver in test set, rest of this
    driver's data becomes the training set.
    """

    test_df = df[df['Class'] == driver_nr_ood]
    df_driver_train = df[df['Class'] == driver_nr_train]
    sampled_test_df = df_driver_train.groupby('Class').sample(frac=frac, random_state=42)
    test_df = pd.concat([test_df, sampled_test_df])
    train_df = df_driver_train.drop(sampled_test_df.index)

    return train_df, test_df    


###############################################################################
# Models
###############################################################################


class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  
        self.fc2 = nn.Linear(92, 64)
        self.fc3 = nn.Linear(64,32)  #new
        self.fc4 = nn.Linear(32,25)  #new
        self.fc5 = nn.Linear(25, 10)   # Output layer: 10 classes
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x)) #
        x = self.relu(self.fc4(x)) #
        x = self.fc5(x)
        return x