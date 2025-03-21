
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class IsolationForestModel():
    def __init__(self):
        super(IsolationForestModel, self).__init__()
              
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('isolation_forest', IsolationForest(
                n_estimators=100,         # Number of trees
                max_samples= 'auto',      # Samples used for training each tree
                contamination= 'auto',    # Proportion of outliers in the data
                random_state=42,          # For reproducibility
                n_jobs=-1                 # Use all available cores
            ))
        ])

    def train_model(self, train_data):
        self.pipeline.fit(train_data)

    def compute_ood_score(self, x):
        scores = self.pipeline.decision_function(x)
        scores = 1 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return scores # higher values => more likely OOD

    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        return scores, labels
    

class LocalOutlierFactorModel():
    def __init__(self):
        super(LocalOutlierFactorModel, self).__init__()
              
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('isolation_forest', LocalOutlierFactor(
                novelty=True,
                n_jobs=-1
                ))
        ])

    def train_model(self, train_data):
        self.pipeline.fit(train_data)

    def compute_ood_score(self, x):
        scores = self.pipeline.decision_function(x)
        scores = 1 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return scores # higher values => more likely OOD

    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        return scores, labels
    

class OCSVM():
    def __init__(self):
        super(OCSVM, self).__init__()

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('one_class_svm', OneClassSVM(
                kernel='rbf', 
                gamma=0.01, 
                nu= 0.05))
            ])
        
    def train_model(self, train_data):
        self.pipeline.fit(train_data)

    def compute_ood_score(self, x):
        scores = self.pipeline.decision_function(x)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return 1 - scores # higher values => more likely OOD
    
    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        return scores, labels