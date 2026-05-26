# 第 3 章 EquiformerV2 / Eqv2-Lite 代码清单与规整建议

生成日期：2026-04-29

## 范围

本文档只覆盖论文第 3 章相关代码：基于轻量化等变图网络 Eqv2-Lite / Equiformer-Light 的氢吸附自由能预测、完整 EquiformerV2 对照、结构-表格融合基线、消融实验、推理效率对比和第 3 章结果汇总。

第 4 章 HEA-Gen、RAG、LoRA、GRPO、大语言模型智能体和第 5 章系统实现不纳入本次清单。

## 当前索引文档

本次规整先建立文档索引，不移动历史实验目录：

| 文档 | 用途 |
| --- | --- |
| `README.md` | 项目入口，说明第三章主线、核心代码和重要结果目录。 |
| `docs/documentation_index.md` | 文档阅读入口，说明先看哪份文档。 |
| `docs/cleanup_status.md` | 当前规整进度，记录已完成与未触碰内容。 |
| `docs/project-organization.md` | 按代码、结果、文档、归档边界对当前项目做总分类。 |
| `docs/code_index.md` | 精简代码索引，方便快速判断脚本用途。 |
| `docs/results_manifest.md` | 论文结果到已有实验目录和结果文件的映射。 |
| `docs/artifacts.md` | 数据、checkpoint、结果文件的保留策略。 |
| `docs/archive_candidates.md` | 后续可归档文件候选清单。 |
| `docs/chapter3_revision_evidence.md` | 针对评阅意见的第 3 章证据汇总。 |
| `docs/parameter_reallocation_experiment.md` | 参数重分配补充实验设计与运行说明。 |
| `docs/tables/ch3_param_perf_speed.csv` | 参数量、性能与推理效率对比表。 |
| `docs/tables/ch3_architecture_ablation.csv` | Lmax、层数、径向基与模块消融汇总表。 |
| `docs/tables/ch3_local_environment_modeling.md` | 局部化学环境建模说明。 |
| `archive/README.md` | 后续归档规则，当前不移动历史结果。 |
| `src/README.md` | 源码目录说明。 |
| `scripts/README.md` | 脚本目录说明。 |
| `datasets/README.md` | 数据目录说明。 |
| `result/README.md` | 结果目录说明。 |
| `experiments/README.md` | 实验输出目录说明。 |

## 结论

第 3 章真正有用的主线代码可以归为六组：

1. 数据构建：`datasets/custom_data_processor_simplified.py` 和 `datasets/custom_hydrogen/`
2. Eqv2-Lite 核心模型：`src/standalone_equiformer_v2.py`
3. 完整 EquiformerV2 对照：`src/enhanced_equiformer_v2.py`
4. 训练与评估入口：`src/train_equiformer.py`、`src/train_enhanced_equiformer_v2.py`、`src/predict_enhanced_equiformer_v2.py`
5. 融合与消融：`scripts/train_test_tabular_fusion.py`、`scripts/run_eq_lite_ablation_depth.py`、`scripts/run_channels.py`、`scripts/run_radial_ablation.py`、`src/run_lmax_ablation.py`
6. 结果证据：`result/lab01/`、`result/eq_lite_ablation/`、`result/speed_benchmark/`、`experiments/2025-07-09_run1/`、`experiments/2025-09-10_run1/`、`experiments/20251021_*tabfusion*`、`实验结果/实验结果摘要.csv`

最需要保留并整理的是 Eqv2-Lite 主实现、Enhanced 原版对照、数据处理、融合模型和消融/速度实验。官方 OC20 旧入口、Catalysis-Hub 下载脚本、商业材料筛选与成本分析不属于第 3 章核心，可归档或移到后续章节/附录材料。

## 规整硬约束

后续如果开始实际移动或重命名文件，应遵守以下硬约束：

1. 不影响原实验结果：已有 checkpoint、LMDB、预测 CSV、metrics、图表和实验目录先不改名、不覆盖、不删除。任何新结构都应通过复制、兼容入口或 manifest 关联旧产物。
2. 保留旧 import 兼容：如果把 `src/standalone_equiformer_v2.py`、`src/enhanced_equiformer_v2.py` 等文件迁到更规范的包结构，旧文件应暂时保留为 re-export shim，避免历史脚本断裂。
3. 保留 LMDB pickle 兼容：当前 LMDB 反序列化依赖 `SimpleData` 类路径和 `CustomUnpickler`。迁移 Dataset/SimpleData 前必须验证旧 LMDB 能继续读取，或明确提供重建 LMDB 的脚本和 manifest。
4. 代码与资产分离：源码、配置、数据、checkpoint、实验结果、论文图表应分层管理。大文件不应无说明地混在源码主线。
5. 结果可追溯：每个论文指标必须能追溯到脚本、配置、输入数据、checkpoint、输出目录和结果文件。
6. 开源可读性优先：面向 GitHub 读者提供清晰 README、环境安装、最小复现命令、数据说明、结果说明、引用和许可证说明。
7. 先文档后迁移：先建立 README、manifest、复现说明和兼容策略，再移动代码。避免一次性大搬家导致无法定位回归。

## 推荐规整策略

建议采用渐进式规整，而不是直接删除或重排整个仓库：

1. 文档优先：补齐 `README.md`、`docs/project-organization.md`、`docs/artifacts.md`、`docs/results_manifest.md`。
2. 新结构并行：先创建规范目录，把整理后的入口放入新路径，同时保留旧文件作为兼容层。
3. 兼容验证：每迁移一组文件，验证旧脚本 import、LMDB 读取、checkpoint 加载和关键结果读取。
4. 历史归档：demo、mock、调试脚本移入 `archive/`，但不删除，且在归档 README 中说明来源和状态。
5. 最后清理缓存：只在确认无实验依赖后删除 `__pycache__`、`.DS_Store`、`Zone.Identifier` 等无效文件。

推荐最终形态是“开源规范结构 + 历史结果可追溯 + 旧入口可兼容”，而不是只追求目录干净。

## 保留：第 3 章核心代码

### 1. Eqv2-Lite / Equiformer-Light 主实现

| 文件 | 建议 | 理由 |
| --- | --- | --- |
| `src/standalone_equiformer_v2.py` | 必须保留 | 第 3 章主要创新点。该文件实现 StandaloneEquiformerV2，去掉 OCP 框架依赖，使用预处理图结构、自定义 LMDB Dataset、简化高斯距离展开和轻量化 Transformer block 配置。 |
| `src/train_equiformer.py` | 保留但需修正 | Eqv2-Lite 主训练脚本，产出 `result/lab01/`。当前硬编码 Windows 路径 `D:/mylab/equiformer/result/lab01`，建议改成项目相对路径。 |
| `result/lab01/` | 保留结果证据 | 存放 Eqv2-Lite 主实验 checkpoint、配置、训练历史、预测和图。对应论文主实验性能。 |
| `checkpionts/best_standalone_equiformer_v2_model.pt` | 保留但建议改目录名 | Standalone/Eqv2-Lite 训练权重。目录名 `checkpionts` 拼写错误，后续整理时建议迁到 `checkpoints/` 并保留兼容说明。 |

### 2. 完整 EquiformerV2 / Enhanced 对照

| 文件 | 建议 | 理由 |
| --- | --- | --- |
| `src/enhanced_equiformer_v2.py` | 必须保留 | 完整 EquiformerV2 对照实现，继承 `EquiformerV2_OC20`，用于证明 Eqv2-Lite 相对原版的性能与效率优势。 |
| `src/equiformer_v2/nets/equiformer_v2/*` | 必须保留 | Eqv2-Lite 和 Enhanced 都依赖这些 SO3/SO2/Transformer/径向基核心模块。 |
| `src/train_enhanced_equiformer_v2.py` | 必须保留 | Enhanced 原版训练、消融批跑和结果落盘入口。 |
| `src/predict_enhanced_equiformer_v2.py` | 保留 | 用训练好的 Enhanced 模型预测并输出丰富指标，服务对照实验和结构预测 CSV。 |
| `src/evaluate_equiformer_v2.py` | 保留但需修正 | Standalone/Eqv2-Lite 评估脚本，可加载 `best_standalone_equiformer_v2_model.pt` 并生成报告/图。当前更像旧入口，建议后续并入统一评估命令。 |
| `experiments/2025-09-10_run1/` | 保留结果证据 | Enhanced baseline、结构参数消融和融合前结构预测结果的主要来源。 |
| `checkpionts/best_enhanced_equiformer_v2.pt` | 保留但建议改目录名 | Enhanced 对照权重。 |

### 3. 数据集构建与标准化

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `datasets/custom_data_processor_simplified.py` | 必须保留 | 将 VASP 结构和 Excel 特征转成 LMDB，是第 2/3 章数据构建和 Eqv2-Lite 训练的数据入口。 |
| `datasets/custom_hydrogen/` | 必须保留 | 当前主实验实际使用的 LMDB 数据、归一化统计和数据摘要。 |
| `data/raw/25features_for_ML.xlsx` | 保留 | 原始表格特征数据来源。 |
| `data/raw/10features_for_ML.xlsx` | 保留但标注用途 | 早期/低维表格特征文件。若第 3 章最终只采用 25 features，应在数据说明中标明其历史用途。 |
| `data/raw/the_atomic_structure_for_ML_model.zip` | 保留 | 原始结构数据来源。 |
| `data/raw/the_atomic_structure_for_ML_model/` | 保留或可重建 | 解压后的 VASP 结构目录。若保留 zip 且可稳定重建，该目录可不进入 Git，但应在 `docs/artifacts.md` 中说明。 |
| `data/processed/cleaned_data.csv` | 保留 | 结构-表格融合脚本使用的表格特征 CSV。 |
| `data/processed/cleaned_data_no_outliers.csv` | 保留但标注用途 | 去异常值版本，需说明是否进入第 3 章正式实验。 |
| `data/processed/tabular_norm.json` | 保留 | 融合模型表格标准化参数。 |
| `datasets/custom_hydrogen/data_summary.json` | 保留 | 论文/答辩中说明数据规模和图结构统计很有用。 |
| `datasets/custom_hydrogen/normalization_stats.json` | 保留 | 训练和预测反归一化依赖。 |
| `datasets/custom_hydrogen_ocp/` | 归档或标注备用 | OCP 风格 LMDB 数据。若最终第 3 章不用该版本，应归档为备用数据格式，避免和主数据混淆。 |
| `configs/custom_hydrogen_config.yml` | 保留但需校验 | 早期/自定义氢吸附训练配置。建议说明是否仍能运行。 |
| `configs/custom_hydrogen_equiformer.yml` | 保留但需校验 | OCP/Equiformer 风格配置。建议作为配置证据保留，后续验证可运行性。 |

### 4. 结构-表格融合基线

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `scripts/train_test_tabular_fusion.py` | 必须保留 | 第 3 章“特征-结构混合融合基线模型”的主要实现，包含 TabularMLP、concat/gate FusionHead、训练、测试和 Top-K 排序。 |
| `src/equiformer_v2/nets/tabular_branch.py` | 保留但需去重 | 也实现了 TabularStandardizer、TabularMLP 和 FusionHead，和 `scripts/train_test_tabular_fusion.py` 功能重叠。建议后续决定一个作为正式库实现，另一个作为 CLI 入口。 |
| `experiments/struct_preds_real/` | 保留 | 真实结构模型预测输出，融合模型输入。 |
| `experiments/20251021_160614_tabfusion_run_real_equiformer_with_loss/` | 保留结果证据 | Concat 融合结果，摘要中 MAE 约 0.1049。 |
| `experiments/20251021_165725_tabfusion_run_gate_fusion/` | 保留结果证据 | Gate 融合较好结果，摘要中 MAE 约 0.1007。 |
| `experiments/20251021_165820_tabfusion_run_gate_fusion_100epochs/` | 保留结果证据 | Gate 100 epoch 结果，摘要中 MAE 约 0.1024。 |
| `scripts/extract_equiformer_predictions.py` | 保留但标注风险 | 生成融合所需结构预测 CSV，但脚本里用 80/10/10 顺序切分预测文件，缺少原始 split 映射，建议只作为历史复现实用脚本，不作为严谨主入口。 |

### 5. 消融实验

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `scripts/run_eq_lite_ablation_depth.py` | 必须保留 | Eqv2-Lite 层数消融，对应第 3 章核心架构消融。 |
| `scripts/run_channels.py` | 必须保留 | sphere_channels 消融，对应模型容量/通道数实验。 |
| `scripts/run_radial_ablation.py` | 保留 | 高斯径向基数量消融，对应距离展开设计验证。 |
| `src/run_lmax_ablation.py` | 保留 | lmax 球谐阶数消融，对应等变表达能力与复杂度权衡。 |
| `scripts/run_parameter_reallocation_ablation.py` | 新增保留 | 参数重分配补充实验，对比 FFN、attention、edge、radial 和 balanced 容量分配。 |
| `result/eq_lite_ablation/depth/` | 保留结果证据 | 层数消融 metrics、预测和曲线。 |
| `result/eq_lite_ablation/channels/` | 保留结果证据 | 通道数消融结果。 |
| `result/eq_lite_ablation/radial/` | 保留结果证据 | 径向基数量消融结果。 |
| `result/eq_lite_ablation/lmax/` | 保留结果证据 | lmax 消融结果。 |
| `experiments/ablation/` | 保留但可拆分 | 其中有 Enhanced/模块消融脚手架、绘图和汇总。建议保留 `grids/`、`analysis/`、`plots/` 和结果 CSV，但把 demo/test 脚本归入 archive。 |
| `scripts/collect_ablation_results.py` | 保留 | 汇总 `experiments/2025-09-10_run1/ablation` 指标。 |
| `scripts/generate_ablation_plots.py` | 保留 | 生成论文可用的消融图和模型对比图。 |
| `experiments/ablation/plots/ablation_overview.md` | 保留 | 消融实验说明文本，可转成论文附录或项目 README。 |

### 6. 推理效率对比

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `result/speed_benchmark/README.md` | 必须保留 | 说明 Eqv2-Lite 与原版 EquiformerV2 的效率对比实验。 |
| `result/speed_benchmark/eq_lite.json` | 必须保留 | Eqv2-Lite 最佳配置。记录 num_layers=3、num_heads=4、num_gaussians=256 等轻量化设置。 |
| `result/speed_benchmark/eqv2_best.json` | 必须保留 | 原版 EquiformerV2 对照配置。 |
| `result/speed_benchmark/infer_speed_test.py` | 必须保留 | 推理速度测试脚本。 |
| `result/speed_benchmark/infer_speed_results.json` | 保留结果证据 | 当前推理速度实验结果。 |
| `result/speed_benchmark/generate_comparison.py` | 保留 | 生成推理效率对比汇总，只读取 `infer_speed_results.json`。 |
| `result/speed_benchmark/run_all_tests.py` | 保留 | 一键运行推理效率脚本和汇总脚本。 |

速度实验统一限定为推理效率对比。`result/speed_benchmark/README.md`、`run_all_tests.py` 和 `generate_comparison.py` 已按该范围整理。

## 保留：第 3 章结果汇总与论文作图素材

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `/home/leo494/mylab/实验结果/实验结果摘要.csv` | 必须保留 | 第 3 章最清楚的实验摘要，记录 Standalone、Enhanced、微调、Concat、Gate 的 R2/MAE/RMSE。 |
| `/home/leo494/mylab/实验结果/实验结果汇总.json` | 必须保留 | 更完整的结果元数据。 |
| `/home/leo494/mylab/Eq-Lite-总结汇报.pdf` | 保留 | 与 Eqv2-Lite 总结相关的汇报材料。 |
| `/home/leo494/mylab/Eq原版消融实验.pdf` | 保留 | 原版/消融汇报材料。 |
| `experiments/model_comparison/` | 保留但需转码检查 | 包含模型对比 CSV 和图；CSV 当前有乱码迹象，建议确认编码后保留。 |
| `experiments/architecture_visualization/` | 保留 | 分支架构、融合机制、训练流程图，可用于论文图或答辩。 |
| `archive/report_helpers/visualize_branch_architecture.py` | 已归档 | 生成结构-表格融合架构图。 |
| `archive/public_packaging_helpers/prepare_public_results.py` | 已归档 | 汇总公开结果、生成报告和增强指标图。 |
| `archive/report_helpers/detailed_analysis_report.py` | 已归档 | 读取 `experiments/model_comparison/model_comparison.csv` 并生成详细分析报告。 |

结果建议按用途分组，避免正式论文结果、辅助图表、demo 和调试输出混在一起误引用：

| 分组 | 含义 | 当前例子 |
| --- | --- | --- |
| 正式论文结果 | 可追溯到真实数据、正式 split、正式脚本和结果文件 | `实验结果/实验结果摘要.csv` 中的 Standalone、Enhanced、Concat、Gate；`result/lab01/`；`experiments/20251021_160614_*`；`experiments/20251021_165820_*` |
| 论文辅助证据 | 可用于图表、消融或附录，但不一定是主表结果 | `result/eq_lite_ablation/`、`experiments/ablation/plots/`、`result/speed_benchmark/` |
| 历史/调试结果 | 可帮助追溯开发过程，不建议进入主论文表格 | demo fusion、mock data、test_grid、test_module_visual |
| 需复核结果 | 指标异常或数据对齐存在风险，需要复核后再决定是否引用 | `experiments/20251021_135737_tabfusion_run_real_data/` |

## 归档：与第 3 章有关但不应放在主线入口

这些文件有历史价值或可帮助复现实验，但不建议作为第 3 章主入口暴露。

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `scripts/process_real_data.py` | 保留辅助脚本 | 真实融合数据处理辅助脚本，仍留在 `scripts/`。 |
| `scripts/run_real_fusion_experiment.py` | 保留辅助脚本 | 自动化串联训练/提取/融合，仍留在 `scripts/`。 |
| `archive/demos/example_tabular_fusion.py` | 已归档 | 示例脚本，不是论文正式实验。 |
| `experiments/20251021_135230_tabfusion_run_demo/` | 归档 | demo 融合实验。 |
| `experiments/20251021_135335_tabfusion_run_demo/` | 归档 | demo 融合实验。 |
| `experiments/20251021_135737_tabfusion_run_real_data/` | 谨慎归档 | 该结果 MAE 异常低，可能来自模拟预测或数据泄漏，不建议作为论文主结果。 |
| `experiments/struct_preds/` | 归档 | 早期结构预测文件。正式融合建议使用 `experiments/struct_preds_real/`。 |
| `archive/old_ablation_runners/test_grid_lmax.py` | 已归档 | 测试/调试脚本。 |
| `archive/old_ablation_runners/test_module_ablation_visual.py` | 已归档 | 可视化调试脚本。 |
| `archive/old_ablation_runners/run_module_ablation_fixed.py` | 已归档 | 模块消融批跑脚本。 |
| `archive/public_packaging_helpers/organize_experiment_results.py` | 已归档 | 结果整理脚本，主线复现不依赖。 |
| `scripts/README_ablation_depth.md` | 归档或合并进正式 README | 深度消融说明，内容可并入第 3 章 README。 |

## 移到其他章节或附录：不是第 3 章 EquiformerV2 主线

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `archive/later_chapter_application/screen_materials.py` | 已归档到后续应用 | HER 商用材料筛选，不是第 3 章模型算法核心，可用于第 5 章或应用展示。 |
| `archive/later_chapter_application/add_material_cost.py` | 已归档到后续应用 | 成本估算与商业筛选，属于材料发现应用，不属于 Eqv2-Lite 算法本体。 |
| `archive/later_chapter_application/metal_price_dict.py` | 已归档到后续应用 | 成本估算依赖。 |
| `/home/leo494/mylab/Jiang/` | 移到应用结果 | 最终 HER 商用筛选交付物，不是第 3 章算法代码。 |
| `data/processed/cathub_H_adsorption.csv` | 归到数据扩展 | Catalysis-Hub 下载数据，当前不属于第 3 章主实验数据。 |
| `README.md` | 已重写 | 当前 README 已是第 3 章 Eqv2-Lite 项目入口。 |
| `requirements.txt` | 已重写 | 当前文件已改为轻量工具依赖说明，完整训练环境见 env 文件。 |

## 删除候选：缓存、系统附属文件、重复包

这些文件不建议直接作为论文项目交付内容。删除前可以先复制到 `archive/` 或确认 Git/备份状态。

| 文件/目录 | 建议 | 理由 |
| --- | --- | --- |
| `__pycache__/`、`src/__pycache__/`、`datasets/__pycache__/`、`scripts/__pycache__/` | 可删除 | Python 缓存。 |
| `.DS_Store` | 可删除 | macOS 系统文件。 |
| `*.Zone.Identifier` | 可删除 | Windows 下载标记，不影响实验。 |
| `micromamba.tar.bz2`、`micromamba.tar.bz2:Zone.Identifier` | 可删除或移到 tools/archive | 环境安装包，不应放在代码主目录。 |
| `mylab/` 嵌套副本 | 谨慎删除/归档 | `/home/leo494/mylab/mylab` 看起来是重复拷贝，删除前确认其中是否有唯一文件。 |
| `datasets/__init__.py` | 已修复 | 现在导出 `custom_data_processor_simplified.py` 中的 `SimpleData` 和 `VASPDataProcessor`，不再引用缺失旧模块。 |
| `docs/README.md` | 保留为上游说明 | 官方 EquiformerV2 README，可放到 `docs/upstream/`，不应作为本项目主 README。 |

## 当前代码中的明显问题

1. 路径硬编码：`src/train_equiformer.py` 使用 `D:/mylab/equiformer/result/lab01`，在 WSL/Linux 下不通用。
2. 目录拼写错误：`checkpionts/` 应为 `checkpoints/`。考虑迁移时保留旧路径兼容说明。
3. README 错位：根目录 `README.md` 讲的是 Catalysis-Hub 下载脚本，不是第 3 章 Eqv2-Lite 项目。
4. 依赖文件错位：根目录 `requirements.txt` 不是训练依赖，真实训练环境在 `env_equiformer_v2_project.yml` 和 `env/env_equiformer_v2.yml`。
5. `datasets/__init__.py` 失效：引用不存在模块，可能导致 `import datasets` 失败。
6. 融合数据切分风险：`scripts/extract_equiformer_predictions.py` 用预测文件顺序做 80/10/10 切分，建议不要作为严谨实验依据。
7. `experiments/model_comparison/model_comparison.csv` 有乱码迹象，应确认原始编码或重新生成。
8. 第 3 章速度实验已经限定为推理效率对比，相关说明集中在 `result/speed_benchmark/README.md`。
9. `SimpleData`、`HydrogenDataset`、`custom_collate_fn` 在 `standalone_equiformer_v2.py` 和 `enhanced_equiformer_v2.py` 中重复定义。后续可抽到统一数据模块，但必须保留旧 LMDB pickle 兼容。
10. `src/equiformer_v2/nets/tabular_branch.py` 和 `scripts/train_test_tabular_fusion.py` 存在融合模块重复实现，后续需要合并或明确主次。
11. 当前缺少 split manifest，无法直接从文档证明 train/val/test 的样本 ID、Excel 行号、VASP 文件和 LMDB index 一一对应。
12. 当前缺少 artifact manifest，无法区分哪些大文件应进入 Git、哪些应放 Git LFS 或外部下载。

## 必须补充的 manifest

为保证“规整不影响原实验结果”，建议在实际迁移前新增以下文档或数据清单：

### 1. 数据切分清单

建议路径：`docs/manifests/split_manifest.md` 和 `data/processed/split_manifest.csv`

至少记录：

| 字段 | 说明 |
| --- | --- |
| `split` | train / val / test |
| `lmdb_index` | LMDB 内部 index |
| `structure_id` | VASP 文件编号或结构编号 |
| `vasp_path` | 原始 VASP 路径 |
| `excel_row` | Excel 原始行号 |
| `target_delta_gh` | 目标 ΔGH |
| `feature_source` | 使用 10 features、25 features 还是 cleaned_data |

### 2. 资产清单

建议路径：`docs/artifacts.md`

至少记录：

| 资产 | 是否必须 | 是否可重建 | 建议存放 | 说明 |
| --- | --- | --- | --- | --- |
| `datasets/custom_hydrogen/*.lmdb` | 是 | 可由 raw 数据重建 | Git LFS 或外部 artifact | 主实验数据 |
| `checkpionts/*.pt` | 是 | 可重训但成本高 | Git LFS 或 release artifact | 论文结果复现权重 |
| `data/raw/the_atomic_structure_for_ML_model.zip` | 是 | 否/需原始来源 | 外部 artifact | 原始结构数据 |
| `data/raw/*.xlsx` | 是 | 否/需原始来源 | Git LFS 或 artifact | 原始特征表 |
| `experiments/*/metrics.*` | 是 | 可重跑但应保留 | Git 普通文件 | 结果证据 |
| `experiments/*/*.png` | 可选 | 可重画 | Git 或 artifact | 论文图/报告图 |

### 3. 结果清单

建议路径：`docs/results_manifest.md`

至少记录：

| 结果 | 脚本 | 配置 | 输入 | 输出 | 论文用途 |
| --- | --- | --- | --- | --- | --- |
| Eqv2-Lite 主实验 | `src/train_equiformer.py` | `result/lab01/config.json` | `datasets/custom_hydrogen/` | `result/lab01/` | 第 3 章主模型 |
| Enhanced 对照 | `src/train_enhanced_equiformer_v2.py` | checkpoint config 或 CLI 参数 | `datasets/custom_hydrogen/` | `experiments/2025-09-10_run1/` | 原版对照 |
| Gate 融合 | `scripts/train_test_tabular_fusion.py` | CLI 参数 + `tabular_norm.json` | `struct_preds_real` + `cleaned_data.csv` | `experiments/20251021_165820_*` | 融合基线 |
| 消融实验 | 对应 ablation 脚本 | 脚本内配置/CSV grid | `datasets/custom_hydrogen/` | `result/eq_lite_ablation/` | 核心消融 |
| 推理效率 | `result/speed_benchmark/infer_speed_test.py` | `eq_lite.json` / `eqv2_best.json` | `datasets/custom_hydrogen/val.lmdb` | `infer_speed_results.json` | 效率对比 |

## 建议规整结构

只做文档和目录规划时，建议最终整理为：

```text
equiformer/
  README.md                         # 重写为第 3 章项目说明
  docs/
    chapter3-code-inventory.md       # 本文档
    upstream/                        # 官方 EquiformerV2 说明和引用材料
  src/
    __init__.py
    models/
      eqv2_lite.py                   # 从 standalone_equiformer_v2.py 迁入或重命名
      eqv2_enhanced.py               # 从 enhanced_equiformer_v2.py 迁入或重命名
    data/
      hydrogen_dataset.py            # Dataset、collate、SimpleData
      preprocess_hydrogen.py         # 从 custom_data_processor_simplified.py 迁入
    train/
      train_eqv2_lite.py
      train_eqv2_enhanced.py
    eval/
      predict_eqv2_enhanced.py
    fusion/
      train_tabular_fusion.py
    compatibility/
      legacy_imports.md              # 记录旧入口和新入口映射
  scripts/
    train_eqv2_lite.py               # 推荐 CLI wrapper，调用 src 包
    train_eqv2_enhanced.py
    train_tabular_fusion.py
  experiments/
    chapter3/
      baseline/
      fusion/
      ablation/
      speed/
  result/
    chapter3/
      lab01/
      eq_lite_ablation/
      speed_benchmark/
  archive/
    old_oc20/
    cathub_downloader/
    demo_fusion/
    system_cache_excluded/
```

实际移动文件前，建议先提交/备份当前状态，再分两步做：

1. 先只创建新 README 和清单，不改路径。
2. 再迁移代码并加兼容入口，避免旧实验脚本一次性失效。

兼容入口示例：

```python
# src/standalone_equiformer_v2.py
from src.models.eqv2_lite import *  # noqa: F401,F403
```

该兼容层至少保留到所有历史脚本完成迁移和验证之后。

## 迁移前验收标准

每次实际移动/重命名后，至少完成以下 smoke checks：

1. `python -c "from src.standalone_equiformer_v2 import StandaloneEquiformerV2"`
2. `python -c "from src.enhanced_equiformer_v2 import EnhancedEquiformerV2"`
3. 读取 `datasets/custom_hydrogen/val.lmdb` 的 1 个样本。
4. 用 Eqv2-Lite 跑 1 个 batch forward。
5. 加载 `checkpionts/best_standalone_equiformer_v2_model.pt`。
6. 加载 `checkpionts/best_enhanced_equiformer_v2.pt`。
7. 读取 `experiments/struct_preds_real/test_preds.csv` 和 `data/processed/cleaned_data.csv` 并完成融合脚本参数解析。
8. 读取 `实验结果/实验结果摘要.csv` 并确认关键指标未被覆盖。

## 最小复现实验路径

如果只为了复现论文第 3 章，最小路径应是：

1. 数据准备：运行或检查 `datasets/custom_data_processor_simplified.py`，确认 `datasets/custom_hydrogen/*.lmdb` 和 `normalization_stats.json` 存在。
2. Eqv2-Lite 主实验：运行 `src/train_equiformer.py`，产出 `result/lab01/`。
3. Eqv2-Lite 评估：运行或整理 `src/evaluate_equiformer_v2.py`，确认可加载 Standalone checkpoint。
4. Enhanced 对照：运行 `src/train_enhanced_equiformer_v2.py` 或使用已有 `experiments/2025-09-10_run1/`。
5. 消融实验：运行 `scripts/run_eq_lite_ablation_depth.py`、`scripts/run_channels.py`、`scripts/run_radial_ablation.py`、`src/run_lmax_ablation.py`。
6. 融合实验：使用 `experiments/struct_preds_real/*.csv` 和 `data/processed/cleaned_data.csv` 运行 `scripts/train_test_tabular_fusion.py`。
7. 速度对比：运行 `result/speed_benchmark/infer_speed_test.py`，用 `eq_lite.json` 和 `eqv2_best.json` 对比推理速度。
8. 汇总结果：参考 `/home/leo494/mylab/实验结果/实验结果摘要.csv`。
