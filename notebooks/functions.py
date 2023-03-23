



def fit_classificator(classificator, params, x_train, y_train, x_test, y_test):
    clf = classificator()

    grid = GridSearchCV(clf, params, cv=5, scoring="roc_auc")
    grid.fit(x_train, y_train)

    best_params = grid.best_params_

    clf = grid.best_estimator_

    return {
        "model": clf,
        "best_params": best_params,
        "score": round(get_probs_and_scores(clf, x_train, y_train, x_test, y_test)
        [
            "test_score"
        ], 3)
    }