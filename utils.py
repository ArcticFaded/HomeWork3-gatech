import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sps
from scipy.linalg import pinv
from scipy.stats import kurtosis
from sklearn.metrics import silhouette_samples, silhouette_score

def kurtotic(X1):
    return np.abs(np.array(kurtosis(X1))).mean()

def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

    
def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

def cluster_sample_silhouette_score(X, clusterLabels):
    return silhouette_samples(X, clusterLabels)

def cluster_silhouette_score(X, clusterLabels):
    return silhouette_score(X, clusterLabels)

def cluster_acc(Y,clusterLabels):
    Y = Y.squeeze(1)
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return accuracy_score(Y,pred)

def classification_reducer_report(grid_search, X_test, y_test):
    print('{classifer} \n {report}'.format(
                classifer='Keras Model', 
                report=classification_report(np.round(grid_search.predict(X_test)), y_test)
            ))

def graph_reducer_results(grid_search, components, batch_size, epochs, algorithm):
    # plot.grid_search(grid_search.cv_results_, change='param_clf__batch_size', kind='bar')
    mean_scores = np.array(grid_search.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(batch_size), -1, len(epochs))


    # select score for best epochs
    mean_scores = mean_scores.max(axis=0)
    print(mean_scores)
    bar_offsets = (np.arange(len(epochs)) *
                (len(components) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(components, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=str(label) + ' n_components', color=COLORS[i])

    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Number of Epocs')
    plt.xticks(bar_offsets + len(components) / 2, epochs)
    plt.ylabel('Heart Disease Classification Accuracy')
    plt.ylim((0.5, 1))
    plt.legend(loc='upper left')

    plt.savefig(algorithm + '_reduction_Keras')

    grid_search.best_params_