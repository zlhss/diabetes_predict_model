# diabetes_predict_model
🩺 Diabetes Prediction with XGBoost + Optuna

本项目使用 XGBoost 原生接口（xgb.train），结合 Optuna 超参数搜索 + Stratified K-Fold 交叉验证 + One-Hot Encoding，构建一个 二分类概率预测模型，用于预测个体是否被诊断为糖尿病（diagnosed_diabetes）。

项目重点放在 工程正确性与可复现性，而不仅是模型调包。

📌 项目特点（Highlights）

✅ 使用 XGBoost 原生 API（DMatrix + train）

✅ Optuna 自动调参（非 Grid / Random）

✅ Stratified K-Fold CV + Early Stopping

✅ One-Hot Encoding（train + test 合并，避免列不一致）

✅ 强制特征对齐（彻底解决 feature_names mismatch）

✅ 输出 预测概率，而非硬分类

✅ 工程级防护（断言、特征模板冻结）

📁 项目结构
.
├── data/
│   ├── train.csv
│   └── test.csv
│
├── xgb_optuna_cv_onehot_prob_fixed.py
├── xgb_submission_optuna_onehot.csv
└── README.md

📊 数据说明
目标变量（Label）

diagnosed_diabetes

0：未被诊断为糖尿病

1：已被诊断为糖尿病

主要特征类型

数值特征
如：age, bmi, cholesterol_total, blood_pressure 等

类别特征（One-Hot Encoding）

ethnicity

education_level

income_level

smoking_status

employment_status

⚙️ 环境依赖
Python >= 3.9
xgboost == 3.1.2
pandas
numpy
scikit-learn
optuna


安装依赖：

pip install xgboost pandas numpy scikit-learn optuna

🚀 使用方法
1️⃣ 准备数据

将数据放入 data/ 目录：

data/train.csv
data/test.csv


确保 train.csv 中包含 diagnosed_diabetes 列。

2️⃣ 运行训练 + 预测
python xgb_optuna_cv_onehot_prob_fixed.py

3️⃣ 输出结果

程序会生成：

xgb_submission_optuna_onehot.csv


格式如下：

id,diagnosed_diabetes
10001,0.73421
10002,0.12893
...


输出为 预测概率，可根据业务需求自行设定阈值。

🧠 方法说明（Methodology）
One-Hot Encoding 策略

将 train + test 合并后统一 One-Hot

避免：

训练集 / 测试集类别不一致

线上预测列缺失问题

特征工程安全机制（重点）

训练阶段冻结 FEATURE_COLS

预测阶段使用 reindex + fill_value=0 强制对齐

使用 assert 检查特征一致性

该机制 彻底解决 XGBoost 的 feature_names mismatch 问题

📈 模型训练策略

目标函数：binary:logistic

评估指标：AUC

交叉验证：Stratified 5-Fold

Early Stopping：50 rounds

超参数搜索：Optuna（30 trials）

🔍 可改进方向（Future Work）

 使用 CV 的 best_iteration 训练最终模型

 引入 scale_pos_weight 处理类别不平衡

 添加特征重要性分析（SHAP）

 拆分为 train.py / infer.py，用于部署

 尝试 LightGBM / CatBoost 对比实验

🎓 项目定位说明

本项目适合用于：

数据科学 / 机器学习课程作业

Kaggle / 天池类竞赛

实习 / 校招项目展示

理解 XGBoost 工程级用法

👤 作者

作者：张冷

技术方向：数据科学 / 机器学习 / 模型工程