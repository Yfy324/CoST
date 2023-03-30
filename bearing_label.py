# basic package
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
#  necessary package
import importlib
from pyod.utils.data import evaluate_print
from pyod.models.base import BaseDetector
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.cof import COF
from pyod.models.suod import SUOD
from pyod.models.feature_bagging import FeatureBagging
import random
from fpr import Fpr
from pyod.utils.example import visualize
from tasks.forecasting import SSD_score
random.seed(0)
warnings.filterwarnings("ignore")

# draw = np.load('/data/yfy/FD-data/RUL/RUL_ts/PHM2.npy', allow_pickle=True).item()
# draw = draw['Bearing2_1'].astype(float)
# d1 = draw[:, :, 0]
# d2 = draw[:, :, 1]
# mse = np.mean((d1+d2)**2, axis=1)
# # mse = np.mean(d1**2, axis=1)
# plt.figure()
# plt.plot(range(len(mse)), mse)
# plt.show()

data = pd.read_csv(r'/data/yfy/FD-data/RUL/bearing_rep/PHM2_7.csv')
data = np.array(data)
clf = IForest(random_state=324, contamination=0.1, max_samples=1.)
# clf = SUOD(contamination=0.1)
# clf = LOF(n_neighbors=180, contamination=0.06)
clf = clf.fit(data)
y_train_pred = clf.labels_
index = np.where(y_train_pred == 1)[0]
print('\n', index)
y_train_scores = clf.decision_scores_  # raw outlier scores
a =y_train_scores + 0.5
plt.figure()
# plt.plot(range(data.shape[0]), y_train_scores)
plt.plot(range(data.shape[0]), a)
plt.show()

plt.figure()
plt.scatter(range(data.shape[0]), y_train_pred)
plt.show()

fig2a = np.zeros(116)
fig2b = np.ones(229-116)
fig2c = np.concatenate((fig2a, fig2b))
stick = [0, 50, 100, 116, 150, 200, 230]

# plt.figure()
plt.subplot(212)
plt.xlim((0, 230))
plt.xticks(stick)
plt.plot(range(fig2c.shape[0]), fig2c)

plt.subplot(211)
plt.xlim((0, 230))
plt.xticks(stick)
plt.plot(range(data.shape[0]), a)

plt.savefig("test.png", dpi=300, format="png")
plt.show()

avg = 0
m_avg = []
for i in range(0, y_train_scores.shape[0]-9, 1):
    m_avg.append(np.mean(y_train_scores[i:i+10]))

m_avg = np.array(m_avg)
plt.figure()
plt.plot(range(m_avg.shape[0]), m_avg)
plt.show()


def ssd_fault_location():
    data = pd.read_csv(r'/data/yfy/FD-data/RUL/bearing_rep/PHM1_3.csv', header=None)
            # data = pd.read_csv(r'/home/yfy/Desktop/project/AD/contrastive/CoST/training/PHM/test_20230114_194905/cost_rep100.csv', header=None)
    data = np.array(data)
    # clf = SUOD(contamination=0.08,
    #            base_estimators=[LOF(n_neighbors=15),
    #                             LOF(n_neighbors=20),
    #                             # HBOS(n_bins=10), HBOS(n_bins=20), COPOD(),
    #                             IForest(n_estimators=50, max_samples=1.),
    #                             IForest(n_estimators=100, max_samples=1.),
    #                             IForest(n_estimators=150, max_samples=1.),
    #                             FeatureBagging()])
    clf = IForest(n_estimators=180, contamination=0.15, max_samples=1.)

    clf = clf.fit(data)
    y_train_pred = clf.labels_
    index = np.where(y_train_pred == 1)[0]
    y_train_scores = clf.decision_scores_  # raw outlier scores
    desc_score_indices = np.argsort(y_train_scores, kind="mergesort")[::-1]

    fault_location = SSD_score().get_loc(data, index)

    score = []
    for i in range(1, data.shape[0]):
        sin = y_train_scores[:i]
        sood = y_train_scores[i:]
        score.append(np.mean(sood) - np.mean(sin))

    indices = np.argsort(score, kind="mergesort")[::-1]

    print(1)


def cross_domain_suod():
    # dataset and model list / dict
    dataset_list = ['PHM1_1']
    model_dict = {'suod': 'SUOD',
                  # 'vae': 'VAE',
                  'iforest': 'IForest', 'cof': 'COF', 'feature_bagging': 'FeatureBagging',
                  'lof': 'LOF'}

    # save the results, seed for reproducible results
    seed = 324
    res = {}
    y_score = {}
    desc_score_indices = {}

    for dataset in dataset_list:
        '''
        la: ratio of labeled anomalies, from 0.0 to 1.0
        realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
        noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
        '''

        data1 = pd.read_csv(r'/data/yfy/FD-data/RUL/RUL_data/'+dataset+'.csv', header=None)
        data1 = np.array(data1)
        data2 = pd.read_csv(r'/data/yfy/FD-data/RUL/RUL_data/PHM1_2.csv', header=None)
        data2 = np.array(data2)
        X_train, y_train = data1[:, :-1], data1[:, -1:]
        X_test, y_test = data2[:, :-1], data2[:, -1:]

        for k, v in model_dict.items():
            # model initialization
            o = importlib.import_module("pyod.models."+k)
            # clf = getattr(o, v)(random_state=seed, contamination=0.1, max_samples=1.)  #, n_estimators=25)  # iforest conta=0.38/0.24
            # clf = getattr(o, v)(n_neighbors=180, contamination=0.06)  # LOF
            # clf = getattr(o, v)(contamination=0.4)  # 0.08
            # clf = SUOD(contamination=0.15,
            #            base_estimators=[LOF(n_neighbors=15),
            #                             LOF(n_neighbors=20),
            #                             # HBOS(n_bins=10), HBOS(n_bins=20), COPOD(),
            #                             IForest(n_estimators=50, max_samples=1.),
            #                             IForest(n_estimators=100, max_samples=1.),
            #                             IForest(n_estimators=150, max_samples=1.),
            #                             FeatureBagging()])

            clf = getattr(o, v)(contamination=0.15,
                                base_estimators=[LOF(n_neighbors=15),
                                                 LOF(n_neighbors=20),
                                                 # HBOS(n_bins=10), HBOS(n_bins=20), COPOD(),
                                                 IForest(n_estimators=50, max_samples=1.),
                                                 IForest(n_estimators=100, max_samples=1.),
                                                 IForest(n_estimators=150, max_samples=1.),
                                                 FeatureBagging()])


            # training, for unsupervised models the y label will be discarded
            clf = clf.fit(X_train)

            # evaluation
            y_train_pred = clf.labels_
            index = np.where(y_train_pred == 1)[0]
            res[v] = index
            print('\n', v, index[0])   # the first 1/detection

            y_train_scores = clf.decision_scores_  # raw outlier scores
            y_score[v] = y_train_scores[index]

            # desc_score_indices[v] = np.argsort(y_train_scores, kind="mergesort")[::-1]
            # scores = y_train_scores[desc_score_indices[v]]
            # high = np.mean(scores[:20])
            # low = np.mean(scores[-20:])


            # critic = Fpr()
            # fpr95 = critic.evaluate(y, y_train_scores)

            # plt.figure()
            # plt.plot(range(m_avg.shape[0]), m_avg)
            # plt.show()

            # get the prediction on the test data
            y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
            y_test_scores = clf.decision_function(X_test)  # outlier scores

            # evaluate and print the results


            print("On Training Data:")
            evaluate_print(k, y_train, y_train_scores)
            print("\nOn Test Data:")
            evaluate_print(k, y_test, y_test_scores)

            # example of the feature importance

            # feature_importance = clf.feature_importances_
            # print("Feature importance", feature_importance)

            # visualize the results
            visualize(clf, X_train, y_train, X_test, y_test, y_train_pred,
                      y_test_pred, show_figure=True, save_figure=False)

            print(1)


def get_label_files():
    # phm14和其他整合到一个文件
    # dir1 = r'/data/yfy/FD-data/RUL/phm_dict.npy'
    # data1 = np.load(dir1, allow_pickle=True).item()  # (time-steps, 2560, 2)
    # dir2 = r'/data/yfy/FD-data/RUL/phm_14.npy'
    # data2 = np.load(dir2, allow_pickle=True).item()  # (time-steps, 2560, 2)
    # data1['Bearing1_4'] = data2['Bearing1_4']
    # np.save('/data/yfy/FD-data/RUL/phm.npy', data1)

    rep_path = r'/data/yfy/FD-data/RUL/bearing_rep/'
    run_dir = os.path.join(os.getcwd(), 'training', 'RUL_data')
    os.makedirs(run_dir, exist_ok=True)

    bearing_names = os.listdir(rep_path)
    bearing_names.sort()
    RUL_dict = {'PHM1_1': [1006, 2433, 2739], 'PHM1_2': [641, 828], 'PHM1_3': [990, 1705, 2254],
                'PHM1_4': [832, 1085, 1386], 'PHM1_5': [1192, 2221, 2411], 'PHM1_6': [858, 1435, 2414],
                'PHM1_7': [978, 905, 1428], 'PHM2_1': [114, 250, 875], 'PHM2_2': [160, 293, 678],
                'PHM2_3': [251, 738, 1946], 'PHM2_4': [269, 407, 741], 'PHM2_5': [965, 1156, 2275],
                'PHM2_6': [325, 460, 687], 'PHM2_7': [120, 225, 192], 'PHM3_1': [490], 'PHM3_2': [1442, 1584],
                'XJTU1_1': [850, 905, 1428], 'XJTU1_2': [501, 897, 1750], 'XJTU1_3': [714, 1588, 1788],
                'XJTU1_4': [988, 1059, 1425], 'XJTU1_5': [407, 482, 604], 'XJTU2_1': [5310, 5424, 5652, 5794],
                'XJTU2_2': [553, 1008, 1806], 'XJTU2_3': [3365, 3793, 6053], 'XJTU2_4': [361, 492],
                'XJTU2_5': [1436, 2721, 3651]
                }

    for i in bearing_names:
        dir = os.path.join(rep_path, i)
        data = pd.read_csv(dir, header=None)
        # data = np.array(data)

        # 考虑不同退化过程
        # points = RUL_dict[i.split('.')[0]]
        # y = np.zeros(data.shape[0])
        # label = 1
        # for j in points:
        #     y[j:] = label
        #     label += 1

        points = RUL_dict[i.split('.')[0]][0]
        y = np.zeros(data.shape[0], dtype=int)
        y[points:] = int(1)
        # y.dtype = int

        # np.insert(data, data.shape[1], y, axis=1)
        # np.savetxt(f'{run_dir}/{i}', data,  delimiter=',')

        data.insert(data.shape[1], 'label', value=y)
        data.to_csv(f'{run_dir}/{i}', index=False, header=None)

        a = pd.read_csv(f'{run_dir}/{i}', header=None)


def suod_scores():
    # dataset and model list / dict
    all_models = {'iforest': 'IForest', 'ocsvm': 'OCSVM', 'abod': 'ABOD', 'cblof': 'CBLOF', 'cof': 'COF',
                  'combination': 'aom', 'copod': 'COPOD', 'ecod': 'ECOD', 'feature_bagging': 'FeatureBagging',
                  'hbos': 'HBOS', 'knn': 'KNN', 'lmdd': 'LMDD', 'loda': 'LODA', 'lof': 'LOF', 'loci': 'LOCI',
                  'lscp': 'LSCP', 'mad': 'MAD', 'mcd': 'MCD', 'pca': 'PCA', 'rod': 'ROD', 'sod': 'SOD',
                  'sos': 'SOS', 'vae': 'VAE', 'auto_encoder_torch': 'AutoEncoder', 'so_gaal': 'SO_GAAL',
                  'mo_gaal': 'MO_GAAL', 'xgbod': 'XGBOD', 'deep_svdd': 'DeepSVDD'}

    dataset_list = ['xjtu1-1']
    model_dict = {'suod': 'SUOD',
                  # 'vae': 'VAE',
                  'iforest': 'IForest', 'cof': 'COF', 'feature_bagging': 'FeatureBagging',
                  'lof': 'LOF'}
    # save the results

    # seed for reproducible results
    seed = 324
    res = {}
    y_score = {}
    desc_score_indices={}

    for dataset in dataset_list:
        '''
        la: ratio of labeled anomalies, from 0.0 to 1.0
        realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
        noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
        '''

        data = pd.read_csv(r'/data/yfy/FD-data/RUL/bearing_rep/PHM1_2.csv')
        # data = pd.read_csv(r'/data/yfy/FD-data/RUL/bearing_rep/XJTU1_2.csv')
        data = np.array(data)
        y = np.zeros(data.shape[0])
        y[120:] = 1
        # xtrain = np.concatenate((data[:100], data[-100:]), axis=0)
        # ytrain = np.zeros(100)
        # ytrain[100:] = 1

        for k, v in model_dict.items():
            # model initialization
            o = importlib.import_module("pyod.models."+k)
            # clf = getattr(o, v)(random_state=seed, contamination=0.1, max_samples=1.)  #, n_estimators=25)  # iforest conta=0.38/0.24
            # clf = getattr(o, v)(n_neighbors=180, contamination=0.06)  # LOF
            clf = getattr(o, v)(contamination=0.15,
                                base_estimators=[LOF(n_neighbors=15),
                                                 LOF(n_neighbors=20),
                                                 # HBOS(n_bins=10), HBOS(n_bins=20), COPOD(),
                                                 IForest(n_estimators=50, max_samples=1.),
                                                 IForest(n_estimators=100, max_samples=1.),
                                                 IForest(n_estimators=150, max_samples=1.),
                                                 FeatureBagging()])

            # clf = getattr(o, v)(contamination=0.4)  # 0.08
            # training, for unsupervised models the y label will be discarded
            clf = clf.fit(data)

            # evaluation
            y_train_pred = clf.labels_
            index = np.where(y_train_pred == 1)[0]
            res[v] = index
            print('\n', v, index[0])   # the first 1/detection

            y_train_scores = clf.decision_scores_  # raw outlier scores
            y_score[v] = y_train_scores[index]

            desc_score_indices[v] = np.argsort(y_train_scores, kind="mergesort")[::-1]
            scores = y_train_scores[desc_score_indices[v]]
            # high = np.mean(scores[:20])
            # low = np.mean(scores[-20:])

            avg = 0
            all_avg = []
            for i in range(len(y_train_scores)):
                avg = (y_train_scores[i] + i * avg) / (i + 1)
                all_avg.append(avg)

            all_avg = np.array(all_avg)

            avg = 0
            m_avg = []
            for i in range(0, y_train_scores.shape[0]-9, 1):
                # avg = (y_score[v][i] + i * avg) / (i + 1)
                m_avg.append(np.mean(y_train_scores[i:i+10]))

            m_avg = np.array(m_avg)

            gap_avg = []
            gap = 10
            re = y_train_scores.shape[0] % gap
            remain = y_train_scores.shape[0] - re - gap
            for i in range(0, remain, gap):
                g = np.mean(y_train_scores[i:i+gap])
                gap_avg.append(g)

            gap_avg.append(np.mean(y_train_scores[-re:]))
            gap_avg = np.array(gap_avg)

            # m_avg = []
            # for i in index:
            #     if i <= data.shape[0]-5:
            #         m_avg.append(np.mean(y_train_scores[i-5:i+5]))
            #
            # m_avg = np.array(m_avg)


            # critic = Fpr()
            # fpr95 = critic.evaluate(y, y_train_scores)

            # y1 = all_avg[:300]
            # y1 = y_train_scores[:300]
            # x1 = range(300)
            # x1 = range(800, data.shape[0], 1)
            plt.figure()
            plt.plot(range(data.shape[0]), y_train_scores)
            # plt.plot(x1, y1)
            plt.show()

            y3 = y_train_scores[1350:1600]
            x3 = range(1350, 1600, 1)
            # x3 = range(5000, y_train_scores.shape[0], 1)
            plt.figure()
            # plt.plot(range(m_avg.shape[0]), m_avg)
            plt.plot(x3, y3)
            plt.show()

            y3 = m_avg[1350:1600]
            x3 = range(1350, 1600, 1)
            # x3 = range(5000, m_avg.shape[0], 1)
            plt.figure()
            # plt.plot(range(m_avg.shape[0]), m_avg)
            plt.plot(x3, y3)
            plt.show()

            plt.figure()
            plt.plot(range(m_avg.shape[0]), m_avg)
            plt.show()

            # y2 = gap_avg[:81]
            # x2 = range(81)
            # # x2 = range(81, gap_avg.shape[0], 1)
            # plt.figure()
            # plt.plot(range(gap_avg.shape[0]), gap_avg)
            # # plt.plot(x2, y2)
            # plt.show()

            # plt.figure()
            # plt.hist(y_train_scores, bins=10)
            # plt.show()

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

