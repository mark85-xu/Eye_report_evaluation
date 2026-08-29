# Eye Report Evaluation

赛题3「基于多视角眼表图像的颈动脉超声诊断报告生成」的**代理评测系统（Proxy Evaluation）**。

> **重要声明**：本仓库实现的是 **Proxy Evaluation**，用于 Val 阶段不同模型输出的**相对比较**，**不是官方 `testOffLine.py`**。比赛最终排名以赛方官方评分为准；拿到官方脚本后，需进行 proxy ↔ official 对齐（见文末"待官方确认清单"）。

---

## 一、仓库代码结构

```
Eye_report_evaluation/
├── README.md                        # 本文件：结构 / 指标 / 部署
├── scripts/
│   ├── evaluate_reports.py          # 评测 CLI：对 Val 预测计算 BLEU/ROUGE_L/临床指标
│   └── make_submission.py           # 提交 CLI：从原始预测生成官方格式 res.csv
└── src/
    ├── data/
    │   └── report_parser.py         # 冻结版 v0.1.1 规则解析器（7 个 P0 概念）
    │                                #   —— 仅供 clinical_metric 复用，勿修改
    └── evaluation/
        ├── __init__.py              # 导出入口，__version__ = "0.1.0"
        ├── tokenizer.py             # 唯一分词入口：jieba + 医学字典 → 空格连接
        ├── text_metrics.py          # BLEU-1/2/3/4 + ROUGE_L + METEOR(可选)
        ├── formatter.py             # res.csv 生成（内部 id → official image_id，GBK）
        ├── validator.py             # P0 提交校验（编码/列/id/空报告/GBK 往返）
        ├── clinical_metric.py       # P1 临床一致性指标（复用 report_parser）
        └── README.md                # 模块级设计说明（代理决策与理由）
```

模块职责：

| 模块 | 职责 |
|---|---|
| `tokenizer.py` | 全项目唯一分词入口。去 `CDFI` → 去空白 → `jieba + medical_dict_final.txt` → 空格连接；保留全角标点 |
| `text_metrics.py` | 基于 `pycocoevalcap`：`BLEU-4` 主指标 + `BLEU-1/2/3` + `ROUGE_L`；`METEOR` 可选（无 Java 时自动跳过，不阻塞） |
| `formatter.py` | 将内部 `metadata.id` 映射为官方 `image_id`，产出 `image_id + predicted_report` 两列，GBK 保存，与官方提交路径一致 |
| `validator.py` | P0 提交自检：文件存在 / GBK 可读 / 列精确 / id 无缺失·多余·重复 / 报告非空 / 分词 sanity / GBK 往返 |
| `clinical_metric.py` | P1 分析指标（**非官方排名指标**）：每个 P0 概念 F1、main_status 准确率、macro/micro F1、错误标签（SIDE_ERROR / NEGATION_ERROR / WORDING_ONLY 等），`unknown` 不当 0 |

---

## 二、评价指标

### 官方评分口径（依据赛题规则）
- **只评 `report_2`**，`report_1` 不参与评分。
- 提交文件 `res.csv` 仅两列：`image_id`、`predicted_report`。
- 报告必须用 `jieba + medical_dict_final.txt` 分词后用空格连接。
- 保存方式：`pd.DataFrame({'image_id': ids, 'predicted_report': res}).to_csv("res.csv", index=False, header=True, encoding="gbk")`。
- 按 `image_id` 匹配。

### 本仓库支持的指标
| 指标 | 类型 | 说明 |
|---|---|---|
| **BLEU-4** | 主指标（决定排名） | pycocoevalcap，另附 BLEU-1/2/3 |
| ROUGE_L | 辅助 | pycocoevalcap |
| METEOR | 辅助（可选） | 需要 `java` 在 PATH 上（jar 与 paraphrase 数据已随 pycocoevalcap 打包）；缺失时 `meteor_status: skipped` |
| Clinical (P1) | 分析辅助 | 复用冻结 `report_parser`，衡量医疗语义一致性，**非官方排名指标** |

### 约束
- **不做同义词归一化**（`双侧`≠`两侧`、`未见`≠`未发现` 等一律不合并），BLEU 基于最终实际提交文本。
- **不做 `official_eval/`**；本仓库仅 `src/evaluation/`。
- BLEU-4 corpus 分数与 per-sample 均值的聚合方式不同，per-sample 仅用于错误分析，不取平均冒充 corpus 分。

---

## 三、部署方法

### 1. 环境与依赖
```bash
python -m pip install jieba pandas numpy pycocoevalcap
# 可选：若需要真正跑出 METEOR 分数，安装 Java 并保证 `java` 在 PATH 上
#   sudo apt-get install -y default-jre-headless
# 或使用便携版 JRE（免 root，详见 src/evaluation/README.md 第 11 节）
# 未安装时 METEOR 自动跳过，不影响 BLEU/ROUGE_L
```

### 2. 数据准备
本仓库**不包含**竞赛冻结数据资产，需要自行提供：
- `metadata.csv`：至少含 `id`（内部主键）、`official_id`、`report_2`、`split`。
- 医学字典 `medical_dict_final.txt`（jieba 用户词典）。
- 模型原始预测 CSV：内部 `id` + 原始报告文本列（未分词）。

### 3. 评测 Val 预测
```bash
python scripts/evaluate_reports.py \
    --pred <pred_csv> \
    --metadata <metadata.csv> \
    --split val \
    --medical-dict <medical_dict_final.txt> \
    --output-dir <out_dir>
```
输出：`metrics_summary.json`（corpus 分数）、`per_sample_metrics.csv`（逐样本 BLEU/ROUGE_L + 原始/分词文本）、`clinical_summary.json` / `clinical_per_sample.csv`（P1）。

### 4. 生成官方格式 res.csv（并自检）
```bash
python scripts/make_submission.py \
    --pred <pred_csv> \
    --metadata <metadata.csv> \
    --medical-dict <medical_dict_final.txt> \
    --output res.csv \
    --validate --expected-split test
```
生成 GBK 编码、两列（`image_id` / `predicted_report`）的 `res.csv`；`--validate` 会跑 P0 校验（通过则打印 `Submission : VALID`）。

---

## 四、待官方 `testOffLine.py` 确认的清单

1. 官方分词实现是否 = jieba + `medical_dict_final.txt` + 空格连接（当前从官方 gts 样本反推）。
2. 保留全角标点、移除 `CDFI` 这两个代理决策是否与官方一致。
3. METEOR 是否计入官方指标（已在本机装 Java 跑通，默认 `ok`；是否被官方采用仍待确认）。
4. 官方 BLEU 的实现与聚合方式。
5. 官方对 `image_id` 缺失/多余/重复的容忍度（validator 当前按最严口径）。
