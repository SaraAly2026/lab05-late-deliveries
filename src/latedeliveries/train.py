import json
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression

from latedeliveries.cost import optimize_threshold
from latedeliveries.data import make_orders
from latedeliveries.evaluation import compare_models, evaluate_final_model
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_orders,
)
from latedeliveries.tuning import tune_model


def main():
    reports_dir = Path("reports")
    models_dir = Path("models")

    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    df = make_orders(n=3000, seed=7)
    train, test = split_orders(df)

    X_train = train.drop(columns=["late"])
    y_train = train["late"]

    X_test = test.drop(columns=["late"])
    y_test = test["late"]

    # Compare candidate models using training cross-validation.
    cv_results = compare_models(X_train, y_train)
    cv_results.to_csv(reports_dir / "cv_results.csv", index=False)

    # Tune logistic regression using training data only.
    search = tune_model(X_train, y_train)

    # Optimize the business-cost threshold without using test data.
    model = build_pipeline(
        LogisticRegression(
            C=search.best_params_["model__C"],
            class_weight=search.best_params_["model__class_weight"],
            max_iter=1000,
        )
    )

    threshold_results = optimize_threshold(model, X_train, y_train)
    threshold = threshold_results["best_threshold"]

    # Fit on training data and evaluate once on the held-out test data.
    model.fit(X_train, y_train)
    metrics = evaluate_final_model(model, X_test, y_test, threshold)

    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    joblib.dump(model, models_dir / "late_orders.joblib")

    model_info = {
        "model": "Logistic Regression",
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "data_seed": 7,
        "training_rows": len(train),
        "test_rows": len(test),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "threshold": threshold,
        "test_results": metrics,
    }

    (models_dir / "model_info.json").write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    print("Training completed successfully!")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best threshold: {threshold}")
    print(f"Optimized test cost: ${metrics['costs']['optimized_model']}")


if __name__ == "__main__":
    main()
