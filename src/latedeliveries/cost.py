import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from latedeliveries.pipeline import RANDOM_STATE


def business_cost(y_true, y_pred):
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    flagged = np.sum(predicted == 1)
    missed_late = np.sum((actual == 1) & (predicted == 0))

    return int(4 * flagged + 15 * missed_late)


def optimize_threshold(model, X_train, y_train):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.arange(0.05, 1.0, 0.05)

    costs = [
        business_cost(y_train, probabilities >= threshold) for threshold in thresholds
    ]

    best_index = int(np.argmin(costs))

    return {
        "best_threshold": round(float(thresholds[best_index]), 2),
        "best_cost": costs[best_index],
        "thresholds": thresholds.tolist(),
        "costs": costs,
    }
