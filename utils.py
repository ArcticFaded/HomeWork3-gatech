import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sps
from scipy.linalg import pinv
from scipy.stats import kurtosis
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, adjusted_mutual_info_score
from collections import Counter
import matplotlib.cm as cm

def ami(X1, X2):
    return adjusted_mutual_info_score(X1, X2)

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
    Y_ = Y.squeeze(1)
    print(Y_.shape, clusterLabels.shape)
    assert (Y_.shape[0] == clusterLabels.shape[0])
    pred = np.empty_like(Y_)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y_[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return accuracy_score(Y_,pred)

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

def plot_silhouette_score(X, SS, SSS, k, dataset, cluster_labels_KMeans, cluster_labels_GMM):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    sample_silhouette_values = SSS[k][dataset]['Kmeans']
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (k + 1) * 10])
    
    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels_KMeans == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        
    ax1.set_title("The silhouette plot for KMeans.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=SS[k][dataset]['Kmeans'], color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
    sample_silhouette_values = SSS[k][dataset]['GMM']
    
    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels_GMM == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax2.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax2.set_title("The silhouette plot for GMM.")
    ax2.set_xlabel("The silhouette coefficient values")
    ax2.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax2.axvline(x=SS[k][dataset]['GMM'], color="red", linestyle="--")

    ax2.set_yticks([])  # Clear the yaxis labels / ticks
    ax2.set_xticks([-0.05, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans and GMM clustering on data "
                  "with n_clusters = %d" % k),
                 fontsize=14, fontweight='bold')
    plt.savefig(dataset + '_Silhouetter_' + str(k))

def plot_cluster_accuracy(dataset, acc, clusters):
    gmm = [value[dataset]['GMM'] for key, value in acc.items()]
    kmm = [value[dataset]['Kmeans'] for key, value in acc.items()]
    plt.figure()

    plt.plot(clusters, gmm, label='GMM')
    plt.plot(clusters, kmm, label='KMeans')
    plt.ylabel('Accuracy within clusters')
    plt.xlabel('Number of clusters')
    plt.title('Clustering Accuracy on the '+ dataset +' Dataset')
    plt.legend()
    plt.savefig('Cluster_analysis_accuracy_' + dataset)

def plot_cluster_information(dataset, adjMI, clusters):
    gmm = [value[dataset]['GMM'] for key, value in adjMI.items()]
    kmm = [value[dataset]['Kmeans'] for key, value in adjMI.items()]
    plt.figure()

    plt.plot(clusters, gmm, label='GMM')
    plt.plot(clusters, kmm, label='KMeans')
    plt.ylabel('Information between clusters')
    plt.xlabel('Number of clusters')
    plt.title('Mutual Information on the '+ dataset +' Dataset')
    plt.legend()
    plt.savefig('Cluster_analysis_information_' + dataset)

def KMeans_ELBOW(dataset, SSE, clusters):
    kmm_SSE = [value[dataset] for key, value in SSE.items()]

    plt.figure()
    plt.plot(clusters, kmm_SSE, label='KMM-SSE')
    plt.ylabel('minimized SSE')
    plt.xlabel('Number of clusters')
    plt.title('Elbow method on the ' + dataset + ' Dataset')
    plt.legend()
    plt.savefig('Cluster_analysis_elbow_' + dataset)

def BICandAIC(dataset, ll, clusters):
    gmm_LL_AIC = [value[dataset]['AIC'] for key, value in ll.items()]
    gmm_LL_BIC = [value[dataset]['BIC'] for key, value in ll.items()]

    plt.figure()
    plt.plot(clusters, gmm_LL_AIC, label='GMM-AIC')
    plt.plot(clusters, gmm_LL_BIC, label='GMM-BIC')
    plt.ylabel('likelyhood and Log likelyhood')
    plt.xlabel('Number of clusters')
    plt.title('Probability of a model given X on the ' + dataset + ' Dataset')
    plt.legend()
    plt.savefig('Cluster_analysis_prob_' + dataset)

def plotReducerAndCluster(mean_scores, components):
    bar_offsets_ = (np.arange(len(components)) *
                (len(components) + 1) + .5)
    plt.figure(figsize=(20,10))
    COLORS = np.arange(len(components))
    for i, (label, reducer_scores) in enumerate(zip(components, mean_scores)):
        plt.bar(bar_offsets_ + i, reducer_scores.flatten(), label=str(label) + ' components in Kmeans')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Number of components')
    plt.xticks(bar_offsets_ + len(components) / 2, components)
    plt.ylabel('Heart Disease Classification Accuracy')
    plt.ylim((0.5, 1))
    plt.legend(loc='upper left')

    plt.show()
