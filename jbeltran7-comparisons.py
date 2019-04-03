import sklearn
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from models import create_model
from preprocessing import preprocess_bank_data, preprocess_heart_data
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from utils import cluster_sample_silhouette_score, cluster_silhouette_score, classification_reducer_report, \
    graph_reducer_results, plot_silhouette_score, plot_cluster_accuracy, \
    plot_cluster_information, KMeans_ELBOW, BICandAIC, cluster_acc, plotReducerAndCluster, \
    ami
from sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.metrics.scorer import make_scorer
from collections import defaultdict
# plt.style.use('ggplot')


def clustering_creation(df_final, target_column, dataset):
    
    X = df_final.loc[:, df_final.columns != target_column]
    Y = df_final.loc[:, df_final.columns == target_column]

    clusters = np.linspace(2, len(X.columns), 3, dtype=np.int64, endpoint=True)
    SSE = defaultdict(dict)
    ll = defaultdict(lambda: defaultdict(dict))
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    SS = defaultdict(lambda: defaultdict(dict))
    SSS = defaultdict(lambda: defaultdict(dict))
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)

    for k in clusters:
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(X)
        gmm.fit(X)
        SSE[k][dataset] = km.score(X)
        ll[k][dataset]['AIC'] = gmm.aic(X)
        ll[k][dataset]['BIC'] = gmm.bic(X) 
        SS[k][dataset]['Kmeans'] = cluster_silhouette_score(X, km.predict(X))
        SS[k][dataset]['GMM'] = cluster_silhouette_score(X, gmm.predict(X))
        SSS[k][dataset]['Kmeans'] = cluster_sample_silhouette_score(X, km.predict(X))
        SSS[k][dataset]['GMM'] = cluster_sample_silhouette_score(X, gmm.predict(X))
        acc[k][dataset]['Kmeans'] = cluster_acc(Y,km.predict(X))
        acc[k][dataset]['GMM'] = cluster_acc(Y,gmm.predict(X))
        adjMI[k][dataset]['Kmeans'] = ami(Y.squeeze(1),km.predict(X))
        adjMI[k][dataset]['GMM'] = ami(Y.squeeze(1),gmm.predict(X))
        print(k)

        cluster_labels_km = km.predict(X)
        cluster_labels_gm = gmm.predict(X)

        plot_silhouette_score(X, SS, SSS, k, dataset, cluster_labels_km, cluster_labels_gm)
    plot_cluster_accuracy(dataset, acc, clusters)
    plot_cluster_information(dataset, adjMI, clusters)
    KMeans_ELBOW(dataset, SSE, clusters)
    BICandAIC(dataset, ll, clusters)

def reducer_creation(df_final, target_column, reducer, dataset):
    X = df_final.loc[:, df_final.columns != target_column]
    Y = df_final.loc[:, df_final.columns == target_column]

    my_scorer = make_scorer(cluster_acc, greater_is_better=True)
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)

    components = np.linspace(2, len(X.columns) - 1, 5, dtype=np.int64, endpoint=True)
    estimators = [('reduce_dim', reducer), ('clf', km)]
    param_grid = [dict(reduce_dim__n_components=components, clf__n_clusters=components)]
    pipe = Pipeline(estimators)
    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=my_scorer)
    grid_search.fit(X, Y)
    mean_scores = np.array(grid_search.cv_results_['mean_test_score']).reshape(len(components), -1, len(components))
    plotReducerAndCluster(mean_scores, components)


    estimators = [('reduce_dim', reducer), ('clf', gmm)]
    param_grid = [dict(reduce_dim__n_components=components, clf__n_components=components)]
    pipe = Pipeline(estimators)
    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=my_scorer)
    grid_search.fit(X, Y)
    
    mean_scores = np.array(grid_search.cv_results_['mean_test_score']).reshape(len(components), -1, len(components))
    plotReducerAndCluster(mean_scores, components)

    


def classifer_creation_reduction(df_final, target_column, reducer):
    X = df_final.loc[:, df_final.columns != target_column]
    Y = df_final.loc[:, df_final.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    
    model = KerasClassifier(build_fn=create_model, verbose=0)
    batch_size = [8, 16]
    epochs = [1]#[15, 25, 35, 40]
    components = np.linspace(2, len(X.columns) - 1, 3, dtype=np.int64, endpoint=True)
    estimators = [('reduce_dim', reducer), ('clf', model)]
    param_grid = [dict(reduce_dim__n_components=[components[0]], clf__input_dim=[components[0]], clf__batch_size=batch_size, clf__epochs=epochs),
                dict(reduce_dim__n_components=[components[1]], clf__input_dim=[components[1]], clf__batch_size=batch_size, clf__epochs=epochs),
                dict(reduce_dim__n_components=[components[2]], clf__input_dim=[components[2]], clf__batch_size=batch_size, clf__epochs=epochs)]
    pipe = Pipeline(estimators)
    grid_search = GridSearchCV(pipe, param_grid=param_grid)

    grid_search.fit(X_train, y_train)
    return grid_search, X_train, y_train, X_test, y_test, batch_size, epochs, components


# read csv    
heart = pd.read_csv('./data/heart.csv')
bank = pd.read_csv('./data/bank.csv', sep=';')
# main areas

heart_final, heart_scalers = preprocess_heart_data(heart)
bank_final, heart_scalers = preprocess_bank_data(bank)

pca = PCA()
ica = FastICA(max_iter =1000, tol=1)
svd = TruncatedSVD()
sparse = SparseRandomProjection()
gaussian = GaussianRandomProjection()

reducers = {
    # 'PCA': pca, 
    # 'FastICA': ica, 
    'TruncatedSVD': svd, 
    'SparseProjection': sparse, 
    'GaussianProjection': gaussian
}

for algorithm, reducer_ in reducers.items():
    best_classifier, X_train, y_train, X_test, y_test, batch_size, epochs, components = classifer_creation_reduction(heart_final, 'target', reducer_)
    reducer_creation(heart_final, 'target', reducer_, '_heart')
    print(algorithm + '_heart')
    classification_reducer_report(best_classifier, X_test, y_test)
    graph_reducer_results(best_classifier, components, batch_size, epochs, algorithm + '_heart')

    best_classifier, X_train, y_train, X_test, y_test, batch_size, epochs, components = classifer_creation_reduction(bank_final, 'y', reducer_)
    reducer_creation(bank_final, 'y', reducer_, '_bank')
    print(algorithm + '_bank')
    classification_reducer_report(best_classifier, X_test, y_test)
    graph_reducer_results(best_classifier, components, batch_size, epochs, algorithm + '_bank')

# clustering_creation(heart_final, 'target', 'Heart')
# clustering_creation(bank_final, 'y', 'Bank')



