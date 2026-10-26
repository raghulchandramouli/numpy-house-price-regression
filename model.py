"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    
    x = np.asarray(X, dtype=float)

    nan_mask = np.isnan(x)
    col_mean = np.nan_to_num(np.nanmean(x, axis=0))
    return np.where(nan_mask, col_mean, x)

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    
    x = np.array(X, copy=True)
    q1, q3 = np.percentile(x, 25, axis=0), np.percentile(x, 75, axis=0)
    iqr = q3 - q1

    lower, upper = (q1 - k * iqr), (q3 + k * iqr)
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.

    x = np.array(X, copy=True)
    return np.clip(x, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    
    n, d = np.asarray(numerator), np.asarray(denominator)
    return (n) / (d + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    
    x = np.asarray(X)
    return np.column_stack([x, col])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    

    labels = np.asarray(labels)
    unique = np.unique(labels)

    return (labels[:, None] == unique[None, :]).astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    
    x = np.asarray(X)
    mean, std = np.mean(x, axis=0), np.std(x, axis=0)
    return [mean ,np.where(std == 0, 1.0, std)]

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    
    return (np.asarray(X) - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    
    x = np.asarray(X)
    ones = np.ones((x.shape[0], 1))
    return np.hstack([ones, x])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    
    order = np.arange(0, n_samples)
    repro_seed = np.random.RandomState(seed)
    return repro_seed.permutation(order)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    
    n = len(indices)
    n_train, n_val = int(n * train_ratio), int(n * val_ratio)

    tr_idx, val_idx, te_idx = indices[0: n_train], indices[n_train: n_train + n_val], indices[n_train + n_val:]
    return tr_idx, val_idx, te_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    
    x_sub, y_sub = X[indices], y[indices]
    return (x_sub, y_sub)

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    
    A = X.T @ X
    b = X.T @ y

    theta = np.linalg.solve(A, b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    
    return np.dot(X,  theta)

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    
    residual = np.abs(y_true - y_pred)
    return float(np.mean(residual))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    
    residual = np.abs(y_true - y_pred)
    residual = np.square(residual)
    mean = np.mean(residual)
    root = np.sqrt(mean)
    return float(root)

# Step 17 - r_squared
import numpy as np

def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    res = np.sum((y_true - y_pred) ** 2)
    tot = np.sum((y_true - y_true.mean()) ** 2)

    if tot == 0:
        return 0.0

    return 1.0 - res / tot

# Step 18 - residual_summary
import numpy as np
def residual_summary(y_true, y_pred):
    r = np.asarray(y_true) - np.asarray(y_pred)

    return {
        "mean": np.mean(r),
        "std": np.std(r),
        "median_abs": np.median(np.abs(r))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    
    x_imp = impute_nan_with_mean(X)
    lower, upper= compute_iqr_bounds(x_imp, k=iqr_k)
    clips = clip_columns(x_imp, lower, upper)

    return clips

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    num = X_num[:, ratio_num_idx]
    den = X_num[:, ratio_den_idx]

    ratio = make_ratio_feature(num, den)

    assemble = append_column(X_num, ratio)

    if cat_labels is not None:
        encoded = one_hot_encode(cat_labels)
        assemble = np.hstack([assemble, encoded])

    return assemble

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # Step 1: Get number of samples
    n_samples = X.shape[0]

    # Step 2: Shuffle indices reproducibly
    indices = make_shuffled_indices(n_samples, seed)

    # Step 3: Partition into train, validation, and test
    train_idx, val_idx, test_idx = partition_indices(
        indices, train_ratio, val_ratio
    )

    # Step 4: Extract features and targets for each split
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)

    # Step 5: Return all six arrays
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

