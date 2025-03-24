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
        scores = 1 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return scores   # higher score -> more likely OOD

    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)
       
        # id_LRs = (1 - id_scores) / id_scores  # P_Hp / P_Hd for ID data
        # ood_LRs = (1 - ood_scores) / ood_scores  # P_Hp / P_Hd for OOD data
        # LRs = np.concatenate([id_LRs, ood_LRs])

        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        return scores, labels
    

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
        scores = - self.pipeline.decision_function(x)   # flip sign because higher should mean more likely OOD
        scores = 1 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return scores   # higher score -> more likely OOD

    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)

        # print('id score:', id_scores)
        # print('ood score:', ood_scores)

        # id_LRs = (1 - id_scores) / id_scores  # P_Hp / P_Hd for ID data
        # ood_LRs = (1 - ood_scores) / ood_scores  # P_Hp / P_Hd for OOD data
        # LRs = np.concatenate([id_LRs, ood_LRs])
        
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])   # 0 = inlier, 1 = outlier
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
        return 1 - scores   # higher score -> more likely OOD
    
    def test_model(self, id_data, ood_data):
        id_scores = self.compute_ood_score(id_data)
        ood_scores = self.compute_ood_score(ood_data)

        # id_LRs = (1 - id_scores) / id_scores  # P_Hp / P_Hd for ID data
        # ood_LRs = (1 - ood_scores) / ood_scores  # P_Hp / P_Hd for OOD data
        # LRs = np.concatenate([id_LRs, ood_LRs])
        
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        return scores, labels