import csv

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


data = None


def prepare():
    global data
    data = pd.read_csv("features.zip")
    X = data.drop(["duration",
                   "radiant_win",
                   "tower_status_radiant",
                   "tower_status_dire",
                   "barracks_status_radiant",
                   "barracks_status_dire"], axis=1)
    X.fillna(0, inplace=True)
    y = data["radiant_win"]

    return X, y


def grid_search(X, y, clf_cls, cv, **params):
    def do(param, value):
        scores = 0
        clf = clf_cls(random_state=241, **{param: value})
        start_time = datetime.now()
        for train, test in cv:
            clf.fit(X[train], y[train])
            res = clf.predict_proba(X[test])[:, 1]
            scores += roc_auc_score(y[test], res)
        elapsed_time = datetime.now() - start_time
        score = scores / cv.n_folds
        print("%s=%s\n\
              \tscore=%s\n\
              \ttime elapsed: %s\n" % (param, value, score, elapsed_time))
        return score, clf, elapsed_time

    best_score = 0
    best_param = None
    best_clf = None
    best_elapsed_time = None
    for param, values in params.items():
        for value in values:
            score, clf, elapsed_time = do(param, value)
            if score > best_score:
                best_score = score
                best_param = {
                    param: value
                }
                best_clf = clf
                best_elapsed_time = elapsed_time
    return {
        'score': best_score,
        'param': best_param,
        'clf': best_clf,
        'elapsed_time': best_elapsed_time,
    }


def gradient(X, y, cv):
    print("GradientBoostingClassifier")
    grid = {'n_estimators': (10, 20, 30, 40, 50)}
    return grid_search(X, y, GradientBoostingClassifier, cv, **grid)


def logistic(X, y, cv):
    print("LogisticRegression")
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    return grid_search(X, y, LogisticRegression, cv, **grid)


def cut_X(X):
    features_to_cut = ('lobby_type',) +\
        tuple('r%s_hero' % i for i in range(1, 6)) +\
        tuple('d%s_hero' % i for i in range(1, 6))
    return X.drop(list(features_to_cut), axis=1)


def calc_heroes(X):
    radiant_heroes = reduce(lambda x, row: x + X[row],
                            ('r%d_hero' % i for i in range(2, 6)),
                            X['r1_hero'])
    dire_heroes = reduce(lambda x, row: x + X[row],
                         ('d%d_hero' % i for i in range(2, 6)),
                         X['d1_hero'])
    heroes = radiant_heroes + dire_heroes
    return heroes.unique().size


def bow(X, N):
    X_pick = np.zeros((data.shape[0], N))
    for i, match_id in enumerate(data.index):
        for p in range(1, 6):
            X_pick[i, data.ix[match_id, 'r%d_hero' % p] - 1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % p] - 1] = 1
    return X_pick


def predict_proba_test(clf):
    X_test = pd.read_csv("features_test.zip")
    X_test.fillna(0, inplace=True)
    X_cutted = cut_X(X_test)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cutted)

    probas = clf.predict_proba(X_scaled)
    with open("result.csv", 'w') as csvfile:
        fields = ('match_id', 'radiant_win')
        writer = csv.DictWriter(csvfile, fields)
        writer.writeheader()
        for i, predict in enumerate(probas[:, 1]):
            writer.writerow({
                'match_id': X_test['match_id'][i],
                'radiant_win': predict,
            })
    print("min=%s" % probas[:, 1].min())
    print("max=%s" % probas[:, 1].max())


if __name__ == "__main__":
    X, y = prepare()
    cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)

    bests = []
    bests.append(gradient(X.as_matrix(), y, cv))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    bests.append(logistic(X_scaled, y, cv))

    X_cutted = cut_X(X)
    scaler = StandardScaler()
    X_cutted_scaled = scaler.fit_transform(X_cutted)
    bests.append(logistic(X_cutted_scaled, y, cv))

    N = calc_heroes(X)
    print('Number of heroes: %s' % N)
    X_pick = bow(X, N)

    X_with_bow = np.concatenate((X_cutted_scaled, X_pick), axis=1)
    bests.append(logistic(X_with_bow, y, cv))

    print(bests)

    best = next(reversed(sorted(bests, key=lambda o: o['score'])))
    print("predicit with %s" % best['clf'])
    predict_proba_test(best['clf'])
