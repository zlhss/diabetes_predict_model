import sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

CATEGORICAL_COLS = [
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"

train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")


def encode_onehot(train_df, test_df):
    train_df = train_df.copy()
    test_df  = test_df.copy()

    # 删除无用列
    train_df.drop("gender", axis=1, inplace=True, errors="ignore")
    test_df.drop("gender", axis=1, inplace=True, errors="ignore")

    # 合并 one-hot，保证列一致
    full = pd.concat([train_df, test_df], axis=0)

    full = pd.get_dummies(
        full,
        columns=CATEGORICAL_COLS,
        dummy_na=False
    )

    train_encoded = full.iloc[:len(train_df)].reset_index(drop=True)
    test_encoded  = full.iloc[len(train_df):].reset_index(drop=True)

    return train_encoded, test_encoded


train_df, test_df = encode_onehot(train_df, test_df)

X = train_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
y = train_df[TARGET_COL].astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_aucs =[]

for train_idx, val_idx in skf.split(X,y):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])
    lr_model.fit(X_train,y_train)

    val_prob = lr_model.predict_proba(X_val)[:,1]

    auc = roc_auc_score(y_val,val_prob)

    lr_aucs.append(auc)

    print(f"LR AUC: {auc:.6f}")

print("LR Mean AUC:", np.mean(lr_aucs))

rf_aucs =[]

for train_idx, val_idx in skf.split(X,y):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train,y_train)

    val_prob = rf_model.predict_proba(X_val)[:,1]

    auc = roc_auc_score(y_val,val_prob)

    rf_aucs.append(auc)

    print(f"RF AUC: {auc:.6f}")

print("RF Mean AUC:", np.mean(rf_aucs))
