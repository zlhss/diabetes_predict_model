# XGBoost == 3.1.2
# 原生 xgb.train + DMatrix
# 固定 Optuna 最优参数 + Stratified CV + Early Stopping
# 输出概率，不转 0/1
# ✅ One-Hot Encoding
# ✅ 强制特征对齐（彻底解决 feature mismatch）
# ❗ 未使用 scale_pos_weight

from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

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
print("[INFO] One-Hot Encoding | scale_pos_weight is not used")


# =========================
# 3. 冻结 Optuna 最优参数
# =========================
BEST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "seed": 42,

    "max_depth": 7,
    "eta": 0.019070735300789274,
    "subsample": 0.8880712253010263,
    "colsample_bytree": 0.6056361014783196,
    "min_child_weight": 1,
    "gamma": 0.7357120004150447,
    "lambda": 2.5012429768287956,
}

# ⭐ 冻结训练特征模板（核心）
FEATURE_COLS = X.columns.tolist()


# =========================
# 4. 固定参数 5-fold CV
# =========================
def evaluate_best_xgb(X, y, feature_cols):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_aucs = []
    best_iterations = []
    best_num_boost_rounds = []

    print("\n===== Frozen XGBoost 5-Fold CV =====")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        dtrain = xgb.DMatrix(
            X.iloc[tr_idx],
            label=y.iloc[tr_idx],
            feature_names=feature_cols
        )
        dval = xgb.DMatrix(
            X.iloc[va_idx],
            label=y.iloc[va_idx],
            feature_names=feature_cols
        )

        model = xgb.train(
            params=BEST_PARAMS,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # XGBoost 的 best_iteration 是 0-based，预测范围需要 +1。
        best_iteration = model.best_iteration
        best_num_boost_round = best_iteration + 1

        val_prob = model.predict(
            dval,
            iteration_range=(0, best_num_boost_round)
        )
        fold_auc = roc_auc_score(y.iloc[va_idx], val_prob)

        fold_aucs.append(fold_auc)
        best_iterations.append(best_iteration)
        best_num_boost_rounds.append(best_num_boost_round)

        print(
            f"Fold {fold}: "
            f"ROC-AUC={fold_auc:.6f}, "
            f"best_iteration={best_iteration}, "
            f"num_boost_round={best_num_boost_round}"
        )

    mean_auc = float(np.mean(fold_aucs))
    median_num_boost_round = int(np.median(best_num_boost_rounds))

    print("\n===== CV Summary =====")
    print(f"Mean ROC-AUC: {mean_auc:.10f}")
    print(f"Best iterations: {best_iterations}")
    print(f"Median num_boost_round: {median_num_boost_round}")

    return {
        "fold_aucs": fold_aucs,
        "best_iterations": best_iterations,
        "best_num_boost_rounds": best_num_boost_rounds,
        "mean_auc": mean_auc,
        "median_num_boost_round": median_num_boost_round,
    }


cv_result = evaluate_best_xgb(X, y, FEATURE_COLS)


# =========================
# 5. 用 CV 最佳轮数中位数训练最终模型
# =========================
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
    params=BEST_PARAMS,
    dtrain=dtrain_full,
    num_boost_round=cv_result["median_num_boost_round"]
)

importance = final_model.get_score(importance_type="gain")

importance_df = pd.DataFrame(
    importance.items(),
    columns=["feature", "gain"]
).sort_values("gain", ascending=False)

print(importance_df.head(20))


# =========================
# 6. SHAP 可解释性图
# =========================
SHAP_SAMPLE_SIZE = 5000
SHAP_RANDOM_STATE = 42
FIGURES_DIR = Path(__file__).resolve().parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SHAP_BEESWARM_PATH = FIGURES_DIR / "shap_summary_beeswarm.png"
SHAP_BAR_PATH = FIGURES_DIR / "shap_summary_bar.png"

shap_sample = X.sample(
    n=min(SHAP_SAMPLE_SIZE, len(X)),
    random_state=SHAP_RANDOM_STATE
)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer(shap_sample)

plt.figure()
shap.summary_plot(
    shap_values,
    shap_sample,
    show=False,
    max_display=20
)
plt.savefig(SHAP_BEESWARM_PATH, dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(
    shap_values,
    shap_sample,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[OK] Saved SHAP beeswarm plot: {SHAP_BEESWARM_PATH}")
print(f"[OK] Saved SHAP bar plot: {SHAP_BAR_PATH}")


# =========================
# 7. 预测并保存（概率）
# =========================
test_prob = final_model.predict(
    dtest,
    iteration_range=(0, cv_result["median_num_boost_round"])
)

submission = pd.DataFrame({
    "id": test_df[ID_COL],
    TARGET_COL: test_prob
})

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "xgb_submission_optuna_onehot.csv"
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\n[OK] Saved: {SUBMISSION_PATH}")
