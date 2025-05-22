import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

def load_and_prepare_data(csv_file):
    notes = pd.read_csv(csv_file)
    notes['latent_vector'] = notes['latent_vector'].apply(ast.literal_eval)
    X = np.vstack(notes['latent_vector'].values)
    y = notes['rating'].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalisation
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(), y_train.std()

    def norm_X(X): return (X - X_mean) / X_std
    def norm_y(y): return (y - y_mean) / y_std

    return {
        "X_train": norm_X(X_train), "y_train": norm_y(y_train),
        "X_val": norm_X(X_val),     "y_val": norm_y(y_val),
        "X_test": norm_X(X_test),   "y_test": norm_y(y_test),
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std
    }
