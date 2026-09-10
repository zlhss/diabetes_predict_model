# 数据文件

数据来自 [Kaggle Diabetes Prediction Challenge / Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12/data)。根据官方说明，训练集和测试集由在 Diabetes Health Indicators Dataset 上训练的深度学习模型生成。

这些是机器学习竞赛生成数据，实验结果不代表真实医疗诊断结论。请在 Kaggle 数据页面按其访问要求获取文件，并遵守页面列出的数据使用条款。

在本目录放置：

| 文件 | 本地已核验规模 | 目标列 |
| --- | --- | --- |
| `train.csv` | 700,000 行、26 列 | 包含 `diagnosed_diabetes` |
| `test.csv` | 300,000 行、25 列 | 不包含 `diagnosed_diabetes` |

两个文件均包含 `id`。本项目不需要 `sample_submission.csv`，也不读取原始 Diabetes Health Indicators Dataset。

当前版本不分发 CSV；`.gitignore` 只允许本说明文件进入 `data/` 的版本管理。仓库早期提交中存在 CSV，本轮整理没有改写那些历史提交。请从仓库根目录执行脚本。
