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
from utils import classification_reducer_report, graph_reducer_results
rom sklearn.decomposition import FastICA, PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
# plt.style.use('ggplot')



def classifer_creation_reduction(df_final, target_column, reducer):
    X = df_final.loc[:, df_final.columns != target_column]
    Y = df_final.loc[:, df_final.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    
    model = KerasClassifier(build_fn=create_model, verbose=0)
    batch_size = [8, 16]
    epochs = [1]#[15, 25, 35, 40]
    components = np.linspace(2, 25, 3, dtype=np.int64, endpoint=True)
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
ica = FastICA(max_iters=1000, tol=1)
svd = TruncatedSVD()
sparse = SparseRandomProjection()
gaussian = GaussianRandomProjection()

reducers = {
    'PCA': pca, 
    'FastICA': ica, 
    'TruncatedSVD': svd, 
    'SparseProjection': sparse, 
    'GaussianProjection': gaussian
}

for algorithm, reducer_ in reducers.items():
    best_classifier, X_train, y_train, X_test, y_test, batch_size, epochs, components = classifer_creation_reduction(heart_final, 'target', reducer_)
    print(algorithm + '_heart')
    classification_reducer_report(best_classifier, X_test, y_test)
    graph_reducer_results(best_classifier, components, batch_size, epochs, algorithm + '_heart')

    best_classifier, X_train, y_train, X_test, y_test, batch_size, epochs, components = classifer_creation_reduction(bank_final, 'y', reducer_)
    print(algorithm + '_bank')
    classification_reducer_report(best_classifier, X_test, y_test)
    graph_reducer_results(best_classifier, components, batch_size, epochs, algorithm + '_bank')




classification_accuracy(SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test)
print('==========================================new area==========================================')
SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test = classifer_creation(bank_final, 'y')
classification_accuracy(SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test)