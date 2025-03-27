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
                random_state=42,
                n_jobs=-1
            ))
        ])

    def train_model(self, train_data):
        self.pipeline.fit(train_data)

    def compute_ood_score(self, x):
        scores = self.pipeline.decision_function(x)
        return scores

    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)

        # Normalize using the same min-max range for both ID and OOD scores.
        # This ensures that if two samples (one from ID, one from OOD) 
        # get the same raw score, they remain equally ranked after normalization.
        all_scores = np.concatenate([id_scores, ood_scores])
        min_val, max_val = np.min(all_scores), np.max(all_scores)
        normalized_id_scores = (id_scores - min_val) / (max_val - min_val)
        normalized_ood_scores = (ood_scores - min_val) / (max_val - min_val)

        # Higher score means more likely to be OOD
        normalized_id_scores = 1 - normalized_id_scores
        normalized_ood_scores = 1 - normalized_ood_scores

        scores = np.concatenate([normalized_id_scores, normalized_ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])   # 0 = inlier, 1 = outlier
        return scores, labels, all_scores
    

class LocalOutlierFactorModel():
    def __init__(self):
        super(LocalOutlierFactorModel, self).__init__()
              
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lof', LocalOutlierFactor(
                novelty=True,
                n_jobs=-1
                ))
        ])

    def train_model(self, train_data):
        self.pipeline.fit(train_data)

    def compute_ood_score(self, x):
        scores = self.pipeline.decision_function(x)   # large values correspond to inliers
        return scores

    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)

        # Normalize using the same min-max range for both ID and OOD scores.
        # This ensures that if two samples (one from ID, one from OOD) 
        # get the same raw score, they remain equally ranked after normalization.
        all_scores = np.concatenate([id_scores, ood_scores])
        min_val, max_val = np.min(all_scores), np.max(all_scores)
        normalized_id_scores = (id_scores - min_val) / (max_val - min_val)
        normalized_ood_scores = (ood_scores - min_val) / (max_val - min_val)

        # Higher score means more likely to be OOD
        normalized_id_scores = 1 - normalized_id_scores
        normalized_ood_scores = 1 - normalized_ood_scores
        
        scores = np.concatenate([normalized_id_scores, normalized_ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])   # 0 = inlier, 1 = outlier
        return scores, labels, all_scores
    

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
        scores = self.pipeline.decision_function(x)  # positive for inlier and negative for outlier
        return scores
    
    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)

        # Normalize using the same min-max range for both ID and OOD scores.
        # This ensures that if two samples (one from ID, one from OOD) 
        # get the same raw score, they remain equally ranked after normalization.
        all_scores = np.concatenate([id_scores, ood_scores])
        min_val, max_val = np.min(all_scores), np.max(all_scores)
        normalized_id_scores = (id_scores - min_val) / (max_val - min_val)
        normalized_ood_scores = (ood_scores - min_val) / (max_val - min_val)

        # Higher score means more likely to be OOD
        normalized_id_scores = 1 - normalized_id_scores
        normalized_ood_scores = 1 - normalized_ood_scores

        scores = np.concatenate([normalized_id_scores, normalized_ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])   # 0 = inlier, 1 = outlier
        return scores, labels, all_scores
