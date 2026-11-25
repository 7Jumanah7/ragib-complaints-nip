"""Data loading and leakage-safe splitting (Lab 3)."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .preprocessing import preprocess


def load_data(path="data/complaints.csv"):
    """Load complaints, clean them and add a numeric label column."""
    df = pd.read_csv(path)
    df["text"] = df["text"].map(preprocess)
    topics = sorted(df["topic"].unique())
    label_map = {topic: i for i, topic in enumerate(topics)}
    df["label"] = df["topic"].map(label_map)
    return df, topics, label_map


def group_split(df, test_size=0.5, random_state=7, group_column="customer_id"):
    """Split by customer so one customer never appears in both train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[group_column]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    shared = set(train_df[group_column]) & set(test_df[group_column])
    if shared:
        raise ValueError(f"Data leakage: shared customers {shared}")
    return train_df, test_df
