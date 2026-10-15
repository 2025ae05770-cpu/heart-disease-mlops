"""Champion selection.

Primary criterion is mean cross-validated ROC-AUC (5 folds on the training split): it is
threshold-independent and far less noisy than the ~61-row hold-out set. Ties (to 3 d.p.)
are broken by hold-out recall, because missing a sick patient is the costlier error.
"""


def select_champion(results, metric="cv_roc_auc_mean", tie_breaker="test_recall"):
    if not results:
        raise ValueError("No model results to select from")
    name = max(results, key=lambda n: (round(results[n][metric], 3), results[n][tie_breaker]))
    return {"model_name": name, "metrics": results[name]}
