# Life-Satisfaction

基于 PISA 2018 问卷数据的生活满意度分析项目，包含数据筛选与合并脚本，以及多模型回归、模型评估和可解释性分析流程。

## 项目结构

```text
.
├── Data/
│   ├── 个体/                  # 个体组 train.csv、test.csv
│   └── 集体/                  # 集体组 train.csv、test.csv
├── Data-Preprocessing/        # 原始问卷数据预处理脚本
└── Train/
	└── Predict.py             # 生活满意度回归分析主程序
```

## 数据说明

`Data/个体/` 和 `Data/集体/` 中提供了两组已处理的训练集和测试集。CSV 特征已编码、标准化并完成训练/测试划分，目标列为 `target`；这些文件适合下游建模或结果复核。

| 数据组 | 文件 | 样本数（约） | 字段数（含 target） |
| --- | --- | ---: | ---: |
| 个体 | `train.csv` / `test.csv` | 42,686 / 10,672 | 31 |
| 集体 | `train.csv` / `test.csv` | 57,814 / 14,454 | 31 |

注意：`Train/Predict.py` 当前并不读取上述 CSV。它读取 `DATA_FILE` 指定的原始 Excel 文件，默认目标变量为 `ST016Q01NA`，并在程序内部重新执行数据清洗、插补、特征工程和数据划分。若要使用仓库内的 CSV，需要另行编写读取 `target` 列并训练模型的流程，或调整现有训练程序。

原始 PISA 数据文件及国家列表、变量列表等外部输入未包含在仓库中。请确保你有权使用相应数据，并在运行预处理前准备好这些文件。

## 预处理流程

`Data-Preprocessing/` 下的脚本按数据准备阶段使用，文件顶部或 `main()` 中的 Windows 示例路径需要改成你的实际路径：

1. [01 按照国家将问卷结果从原始数据中筛选出来.py](Data-Preprocessing/01%20按照国家将问卷结果从原始数据中筛选出来.py)：读取 PISA `.sav` 原始文件和国家列表文本，按 `CNTRYID` 筛选国家/地区，输出各国家数据以及合并数据，格式包括 SAV、CSV 和 Excel。国家列表每行按“代码 | 英文名 | 中文名”填写。
2. [02 从问卷结果中筛选出需要的问题调查结果.py](Data-Preprocessing/02%20从问卷结果中筛选出需要的问题调查结果.py)：遍历国家数据目录中的 `.sav` 文件，保留每个文件前 14 列及变量列表中指定的列，并输出 SAV、CSV 和 Excel。变量列表为每行一个变量名。
3. [03 将各个问卷结果拼接起来.py](Data-Preprocessing/03%20将各个问卷结果拼接起来.py)：读取输入目录各子目录中的 Excel 文件，统一列后纵向合并，输出一个 Excel 文件。
4. [04 将两个调查问卷的筛选结果合并.py](Data-Preprocessing/04%20将两个调查问卷的筛选结果合并.py)：将文件 B 与文件 A 按 `CNTRYID` 左连接，保留文件 B 的所有行，并将文件 A 的字段合并进来。文件 A、B 和输出文件路径均在脚本顶部配置。

## 安装依赖

建议在虚拟环境中安装依赖。训练脚本使用下列 Python 包；预处理脚本还需要 `pyreadstat`：

```bash
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn packaging openpyxl gower shap lightgbm xgboost catboost torch pyreadstat
```

PyTorch 可使用 CPU；若环境配置了 CUDA，训练脚本会自动选择 GPU。仓库未提供锁定版本的依赖文件，第三方库版本可能影响兼容性。

## 运行训练与分析

运行前编辑 [Predict.py](Train/Predict.py) 顶部的配置：

- `DATA_FILE`：包含原始特征和目标变量的 Excel 文件路径。
- `TARGET_VARIABLE`：目标列名，默认 `ST016Q01NA`。
- `OUTPUT_DIR`：分析结果输出目录。

确认目标列为生活满意度评分（脚本仅保留 0 到 10 范围内的记录），并根据数据情况调整模型和分析参数后运行：

```bash
python Train/Predict.py
```

该脚本没有命令行参数；数据路径、目标变量、输出目录、随机种子及模型参数都在源文件顶部配置。脚本启动后会依次进行 Gower 距离 KNN 缺失值插补、特征编码和筛选、80/20 训练测试拆分、模型训练、五折交叉验证、SHAP 和 H-statistic 分析。插补和解释性分析可能耗时较长，尤其是数据量较大时。

## 主要输出

结果写入 `OUTPUT_DIR`，包括：

- `train.csv`、`test.csv`、`imputed_data.csv`：处理后的数据。
- `model_metrics_enhanced.csv`、`5fold_cross_validation_results.xlsx`：模型性能及交叉验证结果，指标包括 MAE、RMSE 和 R²。
- `comprehensive_analysis_enhanced.png`、`feature_importance_enhanced.png`、`performance_report_enhanced.txt`：综合分析图、特征重要性图和文本报告。
- `imputation_quality/`：插补前后质量比较结果。
- `shap_analysis/`：SHAP 可解释性分析结果。
- `h_statistic/`：特征交互分析结果。

## 注意事项

- 当前脚本中的输入和输出路径示例为 Windows 绝对路径；在 Linux、macOS 或其他 Windows 环境运行时，请先改成当前机器可访问的路径。
- 预处理脚本会读写较大的 PISA 文件。请预留足够的磁盘空间和内存，并检查 Excel 工作簿及目标列是否符合脚本预期。
- `Train/Predict.py` 在模块顶层直接执行完整分析流程，因此应在配置好路径后作为脚本运行，而不是直接导入。
- 由于Predict.py中绘制的插补质量前后对比图不够清晰，无法在论文中使用,要使用Interpolation quality verification.py程序，以及Predict.py输出的训练集数据科研绘制出论文中相同的对比图
