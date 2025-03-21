import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class OCSVM():
    def __init__(self, cfg):
        super(OCSVM, self).__init__()

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('one_class_svm', OneClassSVM(kernel=cfg.model.kernel, gamma=cfg.model.gamma, nu=cfg.model.nu))
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