import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde_density_and_LR(scores, labels):
    id_scores = scores[labels == 0]  # inlier scores (Hp)
    ood_scores = scores[labels == 1]  # outlier scores (Hd)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # KDE plot
    sns.kdeplot(id_scores, label="Inlier (ID)", fill=True, color='blue', ax=ax[0])
    sns.kdeplot(ood_scores, label="Outlier (OOD)", fill=True, color='red', ax=ax[0])
    ax[0].set_xlabel('Score')
    ax[0].set_ylabel('Density')
    ax[0].set_title('Distribution of scores for ID and OOD')
    ax[0].legend()

    # KDE for calculating the LR
    id_kde = gaussian_kde(id_scores, bw_method='silverman')  
    ood_kde = gaussian_kde(ood_scores, bw_method='silverman')

    # Range of scores
    unique_scores = np.linspace(min(scores), max(scores), 1000)

    # Probability density for ID and OOD 
    id_density = id_kde(unique_scores)  
    ood_density = ood_kde(unique_scores)

    # Give zero values a really small value (only for OOD because denominator)
    ood_density = np.clip(ood_density, 1e-6, None)

    # Calculate and plot the likelihood ratio
    likelihood_ratios = id_density / ood_density
    ax[1].plot(unique_scores, likelihood_ratios, color='b', label='Likelihood Ratio (KDE)')
    ax[1].set_xlabel('Score')
    ax[1].set_ylabel('Likelihood Ratio (LR)')
    ax[1].set_title('Likelihood Ratio for Each Score (using KDE)')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def startup(df):
    """
    Final implementation
    """

    # Startup animation
    print(
    r"""
    .-----------------------------------------------------------------------------.
    | ____       _                  ____               _ _      _   _             |
    ||  _ \ _ __(_)_   _____ _ __  |  _ \ _ __ ___  __| (_) ___| |_(_) ___  _ __  |
    || | | | '__| \ \ / / _ \ '__| | |_) | '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \ |
    || |_| | |  | |\ V /  __/ |    |  __/| | |  __/ (_| | | (__| |_| | (_) | | | ||
    ||____/|_|  |_| \_/ \___|_|    |_|   |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_||
    '-----------------------------------------------------------------------------'
    """)

    model_choice = -1

    while model_choice not in ['1', '2', '3']:
        model_choice = input("Welcome!\n"
                             "The dataset contains 10 drivers, were we have trained 3 models on a One-vs-One comparison,\n"
                             "where 1 driver is in-distribution and the other is out-of-distribution.\n"
                             "Here you can see the specific performance metrics for a specific comparison!\n"
                        "Which model would you like to use?\n"
                        "1. One-class SVM\n"
                        "2. Isolation Forest\n"
                        "3. Local Outlier Factor\n"
                        "\x1b[1;32;40m" + "--> " + '\x1b[0m' + "Answer: "
                        )
        
    id_choice = -1
    ood_choice = -1

    while id_choice not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        id_choice = input("What driver number (0-9) should be in-distribution?\n"
                          "\x1b[1;32;40m" + "--> " + '\x1b[0m' + "Answer: ")
        
    while ood_choice not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        ood_choice = input("What driver number (0-9) should be out-of-distribution?\n"
                          "\x1b[1;32;40m" + "--> " + '\x1b[0m' + "Answer: ")

    if ood_choice == id_choice:
        print("\x1b[1;31;40m" + "The in-distribution and out-of-distribution drivers cannot be the same." + '\x1b[0m')
        return 0
        
    match model_choice:
        case '1':
            df =  df[df['Model'] == 'OneClassSVM']
        case '2':
            df = df[df['Model'] == 'IsolationForest']
        case '3':
            df = df[df['Model'] == 'LOF']

    df = df[df['CLASS_ID'] == int(id_choice)]
    df = df.loc[df['CLASS_OOD'] == int(ood_choice)]

    print('AUROC:', df['AUROC'].values)
    print('FPR95:', df['FPR95'].values)

    # Retrieve labels and scores
    labels = np.array(df['OOD labels'].iloc[0].strip("[]").split(), dtype=float)
    labels = labels.astype(int)
    scores = np.array(df['AllScores'].iloc[0].strip("[]").split(), dtype=float)

    # Plot density and LR
    plot_kde_density_and_LR(scores, labels)


if __name__== "__main__":
    df = pd.read_csv(r'comparison_normalized.csv')
    startup(df)