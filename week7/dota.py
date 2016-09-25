from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def prepare():
    data = pd.read_csv("features.zip")
    X = data.drop(["duration", "radiant_win", "tower_status_radiant",
                   "tower_status_dire", "barracks_status_radiant",
                   "barracks_status_dire"], axis=1)
    X.fillna(0, inplace=True)
    y = data["radiant_win"]

    return X, y


def gradient(X, y, cv):
    for n_estimators in (10, 20, 30, 40, 50):
        print("n_estimators=%s" % n_estimators)
        scores = 0
        clf = GradientBoostingClassifier(n_estimators=n_estimators)
        start_time = datetime.now()
        for train, test in cv:
            clf.fit(X.iloc[train], y[train])
            res = clf.predict(X.iloc[test])
            scores += roc_auc_score(y[test], res)
        elapsed_time = datetime.now() - start_time
        print("\tscore=%s" % (scores / cv.n_folds))
        print("\ttime elapsed: %s" % elapsed_time)


def logistic(X, y, cv):
    print("LogisticRegression")
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    clf = LogisticRegression()
    gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=cv)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    start_time = datetime.now()
    gs.fit(X_scaled, y)
    elapsed_time = datetime.now() - start_time
    print("\tscores:")
    pprint(gs.grid_scores_)
    print("\ttime elapsed: %s" % elapsed_time)


def logistic_cut(X, y, cv):
    features_to_cut = ('lobby_type',) +\
        tuple('r%s_hero' % i for i in range(1, 6)) +\
        tuple('d%s_hero' % i for i in range(1, 6))
    X_cutted = X.drop(list(features_to_cut), axis=1)

    logistic(X_cutted, y, cv)


if __name__ == "__main__":
    X, y = prepare()
    cv = KFold(y.size, n_folds=5, shuffle=True)

    # gradient(X, y, cv)
    logistic(X, y, cv)
    logistic_cut(X, y, cv)
