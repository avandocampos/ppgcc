import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
from utils import (
    load_iris_uci,
    load_vertebral_column_uci,
    generate_artificial_dataset,
    holdout_evaluation,
)
from classifier import GaussianBayesClassifier


def plot_accuracy_rejection_curve(X, y, classifier_class, Wr_values, dataset_name=None, num_trials=20, test_size=0.3, random_state=42):
    accuracies = []
    rejection_rates = []
    
    for Wr in Wr_values:
        classifier = classifier_class(rejection_cost=Wr)
        
        mean_accuracy, std_accuracy, _ = holdout_evaluation(X, y, classifier, num_trials=num_trials, test_size=test_size, random_state=random_state)
        
        total_predictions = len(X) * (1 - test_size) * num_trials
        total_rejected = np.sum([classifier.predict(X) == -1 for _ in range(num_trials)])
        mean_rejection_rate = total_rejected / total_predictions
        
        accuracies.append(mean_accuracy)
        rejection_rates.append(mean_rejection_rate)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rejection_rates, accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'Accuracy-Rejection (AR) Curve for {dataset_name}')
    plt.xlabel('Rejection Rate')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def evaluate_dataset(dataset_name, dataset_loader, num_trials=20):

    X, y = dataset_loader()
    
    classifier = GaussianBayesClassifier(rejection_cost=0.1)
    
    mean_accuracy, std_accuracy, mean_conf_matrix = holdout_evaluation(X, y, classifier, num_trials=num_trials)
    
    print(f"Results for {dataset_name}:")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print("Mean Confusion Matrix:")
    print(mean_conf_matrix)

    return mean_accuracy, std_accuracy, mean_conf_matrix


if __name__ == "__main__":

    Wr_values = [0.04, 0.12, 0.24, 0.36, 0.48]
    X, y = generate_artificial_dataset()
    
    plot_accuracy_rejection_curve(X, y, GaussianBayesClassifier, Wr_values, dataset_name='Artificial Dataset')
