
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
import torch.nn as nn
import numpy as np
import scipy.stats as stats


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
# Sliding windows functions
###############################################################################


def create_sliding_windows(data, window_size=60, overlap=40):
    """
    Create sliding windows from the input data.
    
    Args:
        data: pandas DataFrame where each row is a 1-second sample
        window_size: size of each window in seconds
        overlap: overlap between consecutive windows in seconds
    
    Returns:
        List of pandas DataFrames, each representing a window
    """
    stride = window_size - overlap
    n_samples = len(data)
    windows = []
    
    # Start indices for each window
    start_indices = range(0, n_samples - window_size + 1, stride)
    
    for start_idx in start_indices:
        end_idx = start_idx + window_size
        window = data.iloc[start_idx:end_idx].copy()
        windows.append(window)
    
    return windows


# def create_sliding_windows_for_all_rides(data, window_size=60, overlap=40):
#     """
#     Create sliding windows for all rides in the input data.
    
#     Args:
#         data: pandas DataFrame where each row is a 1-second sample
#         window_size: size of each window in seconds
#         overlap: overlap between consecutive windows in seconds
    
#     Returns:
#         List of pandas DataFrames, each representing a window
#     """
 
#     # Initialize list to store all windows
#     all_windows = []
    
#     # Group data by 'Ride number' and create sliding windows for each ride
#     for ride_number in sorted(data['Ride number'].unique()):
#         ride_data = data[data['Ride number'] == ride_number]
#         ride_windows = create_sliding_windows(ride_data, window_size, overlap)
#         all_windows.extend(ride_windows)
    
#     return all_windows


def extract_features_from_window(df):
    """Extract relevant features from a single time window DataFrame"""
    features = []
    
    # Statistical features for each column
    for column in df.columns:
        # Skip non-numeric or irrelevant columns
        if df[column].dtype in [np.float64, np.int64]:
            features.append(df[column].mean())
            features.append(df[column].std())
            features.append(df[column].min())
            features.append(df[column].max())
            features.append(df[column].skew())
            features.append(df[column].kurt())
            features.append(np.ptp(df[column]))
            features.append(stats.median_abs_deviation(df[column]))
            features.append(np.mean(np.abs(df[column] - 1)))         
    
    return np.array(features)


###############################################################################
# Evaluation functions
###############################################################################


def evaluate_ood_performance(ood_scores, ood_labels):
    auroc = roc_auc_score(ood_labels, ood_scores)
    fpr, tpr, thresholds = roc_curve(ood_labels, ood_scores)
    target_index = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[target_index]
    return auroc, fpr95
    