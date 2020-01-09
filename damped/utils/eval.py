from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import pandas as pd

from .confusion_matrix import print_confusion_matrix


def display_evaluation_result(args, total_labels, total_pred):
    """
    Print confusion_matrix
    Print per-label precisions, recalls, and F1-scores
    Print average Accuracy
    """

    cm = confusion_matrix(total_labels, total_pred, labels=args.label)
    print_confusion_matrix(cm, args.label_name)

    print("Accuracy:  {:.4f}".format(accuracy_score(total_labels, total_pred)))

    # compute per-label precisions, recalls, F1-scores, and supports
    # instead of averaging
    metrics = precision_recall_fscore_support(
        total_labels, total_pred, average=None, labels=args.label
    )

    df = pd.DataFrame(
        list(metrics),
        index=["Precision", "Recall", "Fscore", "support"],
        columns=args.label_name,
    )
    df = df.drop(["support"], axis=0)
    print(df.T)
