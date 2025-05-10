from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

SEED = 42


def split_arrays(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataframe(dataframe):
    dataframe_train_val, dataframe_test = train_test_split(
        dataframe, test_size=0.2, random_state=42
    )

    dataframe_train, dataframe_val = train_test_split(
        dataframe_train_val, test_size=0.2, random_state=42
    )

    return dataframe_train, dataframe_val, dataframe_test


def stratified_split_dataframe(dataframe):
    sss = StratifiedShuffleSplit(2, train_size=0.6, test_size=0.4, random_state=SEED)
    train_index, test_index_b = next(sss.split(dataframe["images"], dataframe["masks"]))

    sssb = StratifiedShuffleSplit(2, train_size=0.5, test_size=0.5, random_state=SEED)
    val_index, test_index = next(
        sssb.split(test_index_b["images"], test_index_b["masks"])
    )

    return train_index, val_index, test_index
