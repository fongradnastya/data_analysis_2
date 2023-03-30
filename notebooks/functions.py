import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV

import pandas as pd

def plot_roc_curves(ax):
    """
    Строит ROC кривые для всех моделей
    :param ax: ось, на которой строим график
    """
    labels = tuple()
    for index, prob in enumerate(probs):
        fpr, tpr, _ = roc_curve(y_test, prob)
        area = round(auc(fpr, tpr), 3)
        labels += (names[index] + f" ({area=})",)
        axes[0].plot(fpr, tpr)
    ax.plot((-0.01, 1.01), (-0.01, 1.01), color="navy", linestyle="--")
    ax.set_title("ROC curves")
    ax.set_xlim((-0.01, 1.01))
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlabel("False positive rate", labelpad=15)
    ax.set_ylabel("True positive rate", labelpad=15)
    ax.legend(loc="lower right", labels=labels)


def get_probs_and_scores(predictor, x_train, y_train, x_test, y_test):
    train_pred_proba = predictor.predict_proba(x_train)
    test_pred_proba = predictor.predict_proba(x_test)
    return {
        "train_pred_proba": train_pred_proba[:, 1],
        "test_pred_proba": test_pred_proba[:, 1],
        "train_score": roc_auc_score(y_train, train_pred_proba[:, 1]),
        "test_score": roc_auc_score(y_test, test_pred_proba[:, 1])
    }


def make_roc_curve_plot(predictor, title, x_train, y_train, x_test, y_test):
    resp = get_probs_and_scores(predictor, x_train, y_train, x_test, y_test)

    train_curve = roc_curve(y_train, resp["train_pred_proba"])

    test_curve = roc_curve(y_test, resp["test_pred_proba"])

    # строим график ROC-кривых
    plt.figure(figsize=(12, 6))
    plt.plot(*train_curve[:2],
             label='Train ROC curve (AUC={:.2f})'.format(resp["train_score"]))
    plt.plot(*test_curve[:2],
             label='Validation ROC curve (AUC={:.2f})'.format(resp["test_score"]))

    # добавляем диагональную линию
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # настраиваем параметры графика
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()