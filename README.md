# QuickBite — Predict Late Deliveries

## Project Overview

This project uses machine learning to predict which QuickBite orders may be delivered late. The goal is to identify orders that need attention while minimizing the cost of unnecessary vouchers and missed late deliveries.

The dataset contains 3,000 simulated orders generated using the provided `data.py` file. An order is classified as late when its delivery delay exceeds 10 minutes.

## Environment Setup

The project uses Python 3.11 and scikit-learn.

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -e .
```

## Run the Project

To reproduce the model training, cross-validation, hyperparameter tuning, threshold optimization, and final evaluation:

```bash
python -m latedeliveries.train
```

Generated outputs:

- `reports/cv_results.csv` — Cross-validation comparison
- `reports/metrics.json` — Final test metrics and business costs
- `models/late_orders.joblib` — Saved trained pipeline
- `models/model_info.json` — Model configuration and metadata

## Data Preparation

The dataset is split into 75% training and 25% testing using a stratified split with random state 42.

The model uses five features: `distance_km`, `items`, `hour`, `weather`, and `day_type`.

`prep_minutes` and `customer_rating` are excluded because they are not available when an order is placed. `order_id` is excluded because it is an identifier.

Numerical features are processed using median imputation and standard scaling. Categorical features are processed using one-hot encoding with support for unseen categories.

## Model Comparison and Tuning

Five models were compared using five-fold stratified cross-validation:

- Dummy Classifier
- Logistic Regression
- Balanced Logistic Regression
- Decision Tree
- Random Forest

Logistic Regression achieved a mean ROC-AUC of approximately 0.769. GridSearchCV tested six parameter combinations and selected `C=0.1` and `class_weight=None`, with a mean ROC-AUC of 0.7686.

The tuning results did not show a meaningful improvement compared with the original model, especially considering the variation across cross-validation folds.

## Business Cost and Threshold Optimization

QuickBite has two business costs:

- $4 for every order flagged as potentially late.
- $15 for every late order that is not flagged.

Out-of-fold training predictions were used to evaluate thresholds from 0.05 to 0.95.

The best threshold was **0.25**, with a training cross-validation cost of **$4,287**.

The test dataset was not used to select the threshold.

## Final Test Results

| Policy | Total Cost |
|---|---:|
| Flag None | $1,755 |
| Flag All | $3,000 |
| Optimized Model | $1,436 |

The optimized model reduced business cost by approximately 18.2% compared with flagging no orders.

The final test confusion matrix was:

| | Predicted On Time | Predicted Late |
|---|---:|---:|
| Actual On Time | 567 | 66 |
| Actual Late | 64 | 53 |

## Automated Testing

The project includes automated tests for data generation, reproducible splitting, feature selection, probability outputs, missing values, unseen categories, business cost calculations, threshold decisions, and saving and reloading the model.

Run the tests:

```bash
python -m pytest -v
```

Check code quality:

```bash
ruff check .
ruff format --check .
```

## Analysis Notebook

The notebook `notebooks/analysis.ipynb` includes:

- Late delivery prevalence
- Cross-validation model comparison
- Business cost versus prediction threshold
- Final test confusion matrix
- Business recommendation

## Recommendation and Limitations

I recommend testing the optimized Logistic Regression model in a limited pilot. It reduced the simulated business cost compared with the two baseline policies.

However, the model still missed 64 late deliveries in the test dataset. The data is simulated, so the results should not be treated as guaranteed performance on real orders. The model should be monitored and evaluated using actual delivery data before wider deployment.