import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_scores(folder_path, figsize=(15,5)):
    losses = []
    accuracies = []
    f1_scores = []
    loss_std = []
    acc_std = []
    f1_std = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            temp_matrix = np.load(file_path)
            means = temp_matrix.mean(axis=0)
            stds = temp_matrix.std(axis=0)

            losses.append(means[0])
            accuracies.append(means[1])
            f1_scores.append(means[2])

            loss_std.append(stds[0])
            acc_std.append(stds[1])
            f1_std.append(stds[2])

    n = len(losses)
    plt.figure(figsize=figsize)
    plt.subplot(1,3,1)
    plt.errorbar([k for k in range(n)], losses, yerr=loss_std, color='blue')
    plt.xticks([k for k in range(n)], labels=['config'+str(k) for k in range(n)])
    plt.title(f'Test loss for the {n} configurations.')
    plt.subplot(1,3,2)
    plt.errorbar([k for k in range(n)], accuracies, yerr=acc_std, color='orange')
    plt.xticks([k for k in range(n)], labels=['config'+str(k) for k in range(n)])
    plt.title(f'Test accuracy for the {n} configurations.')
    plt.subplot(1,3,3)
    plt.errorbar([k for k in range(n)], f1_scores, yerr=f1_std, color='green')
    plt.xticks([k for k in range(n)], labels=['config'+str(k) for k in range(n)])
    plt.title(f'Test f1_score for the {n} configurations.')
    plt.tight_layout()
    plt.savefig('../figures/config_comparison.png')
    plt.show()
    return None

if __name__ == "__main__":
    PATH_RESULTS = '../models_performances/'
    PATH_FIGURES = '../figures/'
    df_config = pd.read_csv(PATH_RESULTS+'configs.csv', engine='python')
    print('\nTable of configs :')
    print(df_config.head(10))
    compare_scores(PATH_RESULTS)
