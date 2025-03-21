import hydra
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score

from utils import id_ood_val_loaders, init_model, init_optimiser, id_ood_val_data


def train_test_model(cfg, data):
    if cfg.model.is_DL:
        train_loader, val_loader, test_id_loader, ood_loader = id_ood_val_loaders(cfg, data)

        model = init_model(cfg)
        optimiser = init_optimiser(model, cfg)

        model.train_model(train_loader, val_loader, optimiser, cfg)
        ood_scores, ood_labels = model.test_model(test_id_loader, ood_loader)

        evaluate_ood_performance(ood_scores, ood_labels)

    else:
        train_data, test_id_data, test_ood_data = id_ood_val_data(cfg, data)

        model = init_model(cfg)
        model.train_model(train_data)
        ood_scores, ood_labels = model.test_model(test_id_data, test_ood_data)        

        evaluate_ood_performance(ood_scores, ood_labels)


def evaluate_ood_performance(ood_scores, ood_labels):
    auroc = roc_auc_score(ood_labels, ood_scores)
    print(f"AUROC: {auroc:.4f}")
    fpr, tpr, thresholds = roc_curve(ood_labels, ood_scores)
    target_index = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[target_index]
    print(f"FPR95: {fpr95:.4f}")


@hydra.main(config_path="../cybercrime_2025_driving/conf", config_name="config_main", version_base=None)
def main(cfg): 
    data = pd.read_csv('data/Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')

    train_test_model(cfg, data)

if __name__ == "__main__":
    main()
