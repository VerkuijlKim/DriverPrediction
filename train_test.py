import pandas as pd
from itertools import permutations

from DataDefined2 import *
from Functions2 import *
from ood_models import *


def load_data():
    """
    Load data and apply (part of) the pre-processing functions.
    """
    df = pd.read_csv('Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')
    columns_to_drop = one_val_col + irrelevant_col
    df = df.drop(columns_to_drop, axis=1)
    df = addRideNumbers(df)
    df = add_delta(df, delta_col)
    encoder = LabelEncoder()
    df['Class'] = encoder.fit_transform(df['Class'])


def train_test(df):
    """
    Trains and tests each model for the 90 possible class comparisons.
    Returns the results in a DataFrame.
    Estimated running time: 15-20 min.
    """
    models = {
    'IsolationForest': IsolationForestModel(),
    'OneClassSVM': OCSVM(),
    'LOF': LocalOutlierFactorModel()
    }

    # All possible ordered pairs of class comparisons, where one is ID and the other OOD
    number_of_classes = list(range(10))
    class_comparisons = list(permutations(number_of_classes, 2))

    results = []

    for CLASS_ID, CLASS_OOD in class_comparisons:
        print(f"Evaluating ID: {CLASS_ID} vs OOD: {CLASS_OOD}")
        
        # Extract ID and OOD data
        id_data = df[df['Class'] == CLASS_ID]
        ood_data = df[df['Class'] == CLASS_OOD]

        # Preprocessing that needs to be done here
        columns_to_remove = ["Time(s)", "Class", "PathOrder"]
        id_data = id_data.drop(columns=columns_to_remove)
        ood_data = ood_data.drop(columns=columns_to_remove)

        # Apply feature selection and sliding window
        id_windows = create_sliding_windows(id_data, window_size=60, overlap=20)
        ood_windows = create_sliding_windows(ood_data, window_size=60, overlap=20)

        n_id_samples = len(id_windows)
        train_idx = int(n_id_samples * 0.7)
        val_idx = train_idx + int(n_id_samples * 0.1)

        train_data = id_windows[:train_idx]
        val_data = id_windows[train_idx:val_idx]
        test_id_data = id_windows[val_idx:]
        train_data += val_data

        train_data = np.array([extract_features_from_window(window) for window in train_data])
        test_id_data = np.array([extract_features_from_window(window) for window in test_id_data])
        test_ood_data = np.array([extract_features_from_window(window) for window in ood_windows])

        # Evaluate for the three models
        for model_name, model in models.items():
            print(f"Evaluating with {model_name}...")

            if model_name == 'LOF': 
                local_outlier_factor = LocalOutlierFactorModel()
                local_outlier_factor.train_model(train_data)
                ood_scores, ood_labels, all_scores = local_outlier_factor.test_model(test_id_data, test_ood_data)    

            elif model_name == 'OneClassSVM':
                one_class_svm = OCSVM()
                one_class_svm.train_model(train_data)
                ood_scores, ood_labels, all_scores = one_class_svm.test_model(test_id_data, test_ood_data)

            elif model_name == 'IsolationForest':
                isolation_forest = IsolationForestModel()
                isolation_forest.train_model(train_data)
                ood_scores, ood_labels, all_scores = isolation_forest.test_model(test_id_data, test_ood_data)    
                
            auroc, fpr95 = evaluate_ood_performance(ood_scores, ood_labels)
            print(auroc)

            results.append({
                'CLASS_ID': CLASS_ID,
                'CLASS_OOD': CLASS_OOD,
                'Model': model_name,
                'AUROC': auroc,
                'FPR95': fpr95,
                'AllScores': all_scores,  # raw scores 
                'OOD labels': ood_labels,
            })

    results_df = pd.DataFrame(results)
    return results_df


def print_auroc_stats(df):
    iforest_stats = df[df['Model'] == 'IsolationForest']
    print('Isolation forest AUROC mean:', iforest_stats['AUROC'].mean())
    print('Isolation forest AUROC std:', iforest_stats['AUROC'].std())
    print('Isolation forest AUROC min:', iforest_stats['AUROC'].min())
    print('Isolation forest AUROC max:', iforest_stats['AUROC'].max(), '\n')

    ocsvm_stats = df[df['Model'] == 'OneClassSVM']
    print('OneClassSVM AUROC mean:', ocsvm_stats['AUROC'].mean())
    print('OneClassSVM AUROC std:', ocsvm_stats['AUROC'].std())
    print('OneClassSVM AUROC min:', ocsvm_stats['AUROC'].min())
    print('OneClassSVM AUROC max:', ocsvm_stats['AUROC'].max(), '\n')

    lof_stats = df[df['Model'] == 'LOF']
    print('LocalOutlierFactor AUROC mean:', lof_stats['AUROC'].mean())
    print('LocalOutlierFactor AUROC std:', lof_stats['AUROC'].std())
    print('LocalOutlierFactor AUROC min:', lof_stats['AUROC'].min())
    print('LocalOutlierFactor AUROC max:', lof_stats['AUROC'].max())


def print_fpr95_stats(df):
    iforest_stats = df[df['Model'] == 'IsolationForest']
    print('Isolation forest FPR95 mean:', iforest_stats['FPR95'].mean())
    print('Isolation forest FPR95 std:', iforest_stats['FPR95'].std())
    print('Isolation forest FPR95 min:', iforest_stats['FPR95'].min())
    print('Isolation forest FPR95 max:', iforest_stats['FPR95'].max(), '\n')

    ocsvm_stats = df[df['Model'] == 'OneClassSVM']
    print('OneClassSVM FPR95 mean:', ocsvm_stats['FPR95'].mean())
    print('OneClassSVM FPR95 std:', ocsvm_stats['FPR95'].std())
    print('OneClassSVM FPR95 min:', ocsvm_stats['FPR95'].min())
    print('OneClassSVM FPR95 max:', ocsvm_stats['FPR95'].max(), '\n')

    lof_stats = df[df['Model'] == 'LOF']
    print('LocalOutlierFactor FPR95 mean:', lof_stats['FPR95'].mean())
    print('LocalOutlierFactor FPR95 std:', lof_stats['FPR95'].std())
    print('LocalOutlierFactor FPR95 min:', lof_stats['FPR95'].min())
    print('LocalOutlierFactor FPR95 max:', lof_stats['FPR95'].max())


if __name__== "__main__":
    df = load_data()
    results_df = train_test(df)
    results_df.to_csv('comparison-data.csv')
    print_auroc_stats(results_df)
    print_fpr95_stats(results_df)