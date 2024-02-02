
import numpy as np
import pandas as pd

from fluxDataset import FluxDataset

def get_mean_pred(train_df : pd.DataFrame, group_by : str):
    if group_by not in train_df.columns:
        raise IndexError(f"{group_by} not found in train_df.columns")
 
    means = train_df.groupby([group_by]).mean(numeric_only=True)
    column_index = means.columns
    groups = train_df[group_by].unique()
    means = means.values

    def pred(test_df : pd.DataFrame):
        test_df = test_df.drop(columns=test_df.columns.difference(column_index))
        test_df = test_df.reindex(columns=column_index)
        test_df[column_index.difference(test_df.columns)] = 0

        data = test_df.values
        diff = data[None,:,:] - means[:,None,:]
        distances = np.linalg.norm(diff, axis=2)
        return groups[np.argmin(distances, axis=0)]
    
    return pred

def get_prediction_df(test_df : pd.DataFrame, train_df : pd.DataFrame, group_by, pred):
    if group_by not in train_df.columns.intersection(test_df.columns):
        print(f"Unable to group by {group_by}, not in columns")

    train_groups = train_df[group_by].unique()
    test_groups = test_df[group_by].unique()

    n_train = len(train_groups)
    n_test = len(test_groups)

    accuracies = np.zeros((n_test, n_train))
    for i, test_group in enumerate(test_groups):
        group_df = train_df[train_df[group_by] == test_group]
        for j, train_group in enumerate(train_groups):
            accuracies[i,j] = get_prediction_accuracy(group_df, train_group, pred)

    test_group_by = f"test_{group_by}"

    df = pd.DataFrame(data=accuracies, columns=train_groups)
    df[test_group_by] = test_groups 
    df = df.reindex(columns= [test_group_by] + list(train_groups))

    return df


def get_prediction_accuracy(data : pd.DataFrame, exp, pred):
    prediction = pred(data)
    return np.mean(prediction == exp)