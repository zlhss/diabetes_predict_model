# XGBoost == 3.1.2
# 原生 xgb.train + DMatrix
# Optuna 调参 + Stratified CV + Early Stopping
# 输出概率，不转 0/1
# ✅ One-Hot Encoding
# ✅ 强制特征对齐（彻底解决 feature mismatch）
# ❗ 未使用 scale_pos_weight

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# =========================
# 1. One-Hot 编码函数
# =========================
CATEGORICAL_COLS = [
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"


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


# =========================
# 2. 读取数据
# =========================
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

train_df, test_df = encode_onehot(train_df, test_df)

# 拆分 X / y
X = train_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
y = train_df[TARGET_COL].astype(int)

X_test = test_df.drop(columns=[ID_COL], errors="ignore")

# 样本分布
neg = (y == 0).sum()
pos = (y == 1).sum()
print(f"neg={neg}, pos={pos}")
print("⚠️ One-Hot Encoding | 未使用 scale_pos_weight")


# =========================
# 3. Optuna 目标函数
# =========================
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": 42,

        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "lambda": trial.suggest_float("lambda", 0.1, 5.0),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for tr_idx, va_idx in skf.split(X, y):
        dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval   = xgb.DMatrix(X.iloc[va_idx], label=y.iloc[va_idx])

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        val_prob = model.predict(dval)
        aucs.append(roc_auc_score(y.iloc[va_idx], val_prob))

    return np.mean(aucs)


# =========================
# 4. 运行 Optuna
# =========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n===== Optuna Result =====")
print("Best AUC:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


# =========================
# 5. 用最优参数训练最终模型
# =========================
best_params = study.best_params.copy()
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "seed": 42,
})

# ⭐ 冻结训练特征模板（核心）
FEATURE_COLS = X.columns.tolist()

# ⭐ 强制对齐 test 特征
X_test = X_test.reindex(columns=FEATURE_COLS, fill_value=0)

dtrain_full = xgb.DMatrix(
    X,
    label=y,
    feature_names=FEATURE_COLS
)

dtest = xgb.DMatrix(
    X_test,
    feature_names=FEATURE_COLS
)

# 双保险断言（开发阶段强烈建议保留）
assert list(dtrain_full.feature_names) == list(dtest.feature_names)

final_model = xgb.train(
    params=best_params,
    dtrain=dtrain_full,
    num_boost_round=500
)


# =========================
# 6. 预测并保存（概率）
# =========================
test_prob = final_model.predict(dtest)

submission = pd.DataFrame({
    "id": test_df[ID_COL],
    TARGET_COL: test_prob
})

submission.to_csv("xgb_submission_optuna_onehot.csv", index=False)
print("\n✅ 输出完成：xgb_submission_optuna_onehot.csv")
