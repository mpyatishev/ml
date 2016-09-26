from datetime import datetime
from functools import reduce
# from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


data = None


def prepare():
    global data
    data = pd.read_csv("features.zip")
    X = data.drop(["duration", "radiant_win", "tower_status_radiant",
                   "tower_status_dire", "barracks_status_radiant",
                   "barracks_status_dire"], axis=1)
    X.fillna(0, inplace=True)
    y = data["radiant_win"]

    return X, y


def grid_search(X, y, clf_cls, cv, **params):
    for param, values in params.items():
        for value in values:
            print("%s=%s" % (param, value))
            scores = 0
            clf = clf_cls(random_state=241, **{param: value})
            start_time = datetime.now()
            for train, test in cv:
                clf.fit(X[train], y[train])
                res = clf.predict(X[test])
                scores += roc_auc_score(y[test], res)
            elapsed_time = datetime.now() - start_time
            print("\tscore=%s" % (scores / cv.n_folds))
            print("\ttime elapsed: %s" % elapsed_time)


def gradient(X, y, cv):
    print("GradientBoostingClassifier")
    grid = {'n_estimators': (10, 20, 30, 40, 50)}
    grid_search(X, y, GradientBoostingClassifier, cv, **grid)
    # for n_estimators in (10, 20, 30, 40, 50):
    #     print("n_estimators=%s" % n_estimators)
    #     scores = 0
    #     clf = GradientBoostingClassifier(n_estimators=n_estimators)
    #     start_time = datetime.now()
    #     for train, test in cv:
    #         clf.fit(X.iloc[train], y[train])
    #         res = clf.predict(X.iloc[test])
    #         scores += roc_auc_score(y[test], res)
    #     elapsed_time = datetime.now() - start_time
    #     print("\tscore=%s" % (scores / cv.n_folds))
    #     print("\ttime elapsed: %s" % elapsed_time)


def logistic(X, y, cv):
    print("LogisticRegression")
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    grid_search(X, y, LogisticRegression, cv, **grid)
    # clf = LogisticRegression()
    # gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=cv, verbose=1)
    # start_time = datetime.now()
    # gs.fit(X_scaled, y)
    # elapsed_time = datetime.now() - start_time
    # print("\tscores:")
    # pprint(gs.grid_scores_)
    # print("\ttime elapsed: %s" % elapsed_time)
    # print("\tbest_score=%s, best_params=%s" % (gs.best_score_,
    #                                            gs.best_params_))


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


if __name__ == "__main__":
    X, y = prepare()
    cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)

    gradient(X.as_matrix(), y, cv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logistic(X_scaled, y, cv)

    X_cutted = cut_X(X_scaled)
    logistic(X_cutted, y, cv)

    N = calc_heroes(X)
    X_pick = bow(X, N)

    X_with_bow = np.concatenate((X_cutted, X_pick), axis=1)
    logistic(X_with_bow, y, cv)
