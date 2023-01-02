# basic package
import os
import pandas as pd
import numpy as np
import warnings
#  necessary package
import pyod.models
import importlib
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

warnings.filterwarnings("ignore")

# dataset and model list / dict
all_models = {'iforest': 'IForest', 'ocsvm': 'OCSVM', 'abod': 'ABOD', 'cblof': 'CBLOF', 'cof': 'COF',
              'combination': 'aom', 'copod': 'COPOD', 'ecod': 'ECOD', 'feature_bagging': 'FeatureBagging',
              'hbos': 'HBOS', 'knn': 'KNN', 'lmdd': 'LMDD', 'loda': 'LODA', 'lof': 'LOF', 'loci': 'LOCI',
              'lscp': 'LSCP', 'mad': 'MAD', 'mcd': 'MCD', 'pca': 'PCA', 'rod': 'ROD', 'sod': 'SOD',
              'sos': 'SOS', 'vae': 'VAE', 'auto_encoder_torch': 'AutoEncoder', 'so_gaal': 'SO_GAAL',
              'mo_gaal': 'MO_GAAL', 'xgbod': 'XGBOD', 'deep_svdd': 'DeepSVDD'}

dataset_list = ['xjtu1-1']
model_dict = {'iforest': 'IForest', 'cof': 'COF', 'feature_bagging': 'FeatureBagging', 'lof': 'LOF'}
# save the results

# seed for reproducible results
seed = 324
res = {}
y_score = {}
for dataset in dataset_list:
    '''
    la: ratio of labeled anomalies, from 0.0 to 1.0
    realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
    noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
    '''

    # data = pd.read_csv(r'/data/yfy/FD-data/RUL/cost_rep.csv', header=None)
    data = pd.read_csv(r'/home/yfy/Desktop/project/AD/contrastive/CoST/training/XJTU/test_20230102_212555/cost_rep100.csv', header=None)
    data = np.array(data)
    y = np.zeros(data.shape[0])
    y[889:] = 1

    for k, v in model_dict.items():
        # model initialization
        o = importlib.import_module("pyod.models."+k)
        # clf = getattr(o, v)(random_state=seed, contamination=0.38)
        clf = getattr(o, v)(contamination=0.08)  # 0.08
        # training, for unsupervised models the y label will be discarded
        clf = clf.fit(data)

        # evaluation
        y_train_pred = clf.labels_
        index = np.where(y_train_pred == 1)
        res[v] = index[0]
        print('\n', v, index[0][0])   # the first 1/detection

        y_train_scores = clf.decision_scores_  # raw outlier scores
        y_score[v] = y_train_scores[index[0]]

        # get the prediction on the test data
        # y_test_pred = clf.predict(data)  # outlier labels (0 or 1)
        # y_test_scores = clf.decision_function(data)  # outlier scores

        # evaluate and print the results
        print("On Training Data:")
        evaluate_print(k, y, y_train_scores)
        # print("\nOn Test Data:")
        # evaluate_print(k, y, y_test_scores)

        # example of the feature importance

        # feature_importance = clf.feature_importances_
        # print("Feature importance", feature_importance)

        # visualize the results
        # visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
        #           y_test_pred, show_figure=True, save_figure=False)

print(1)
