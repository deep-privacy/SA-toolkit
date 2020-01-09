#!/usr/bin/env python

import numpy as np


def print_confusion_matrix(
    cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None
):
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print()
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    print(empty_cell * 3 + "Predicted")
    # Print header
    fst_empty_cell = (
        (columnwidth - 3) // 2 * " " + "True" + (columnwidth - 3) // 2 * " "
    )

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell

    # Print header
    print("    " + fst_empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
    print()
