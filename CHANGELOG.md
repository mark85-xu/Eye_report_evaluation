# Changelog

本仓库所有更新记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/)。
版本号沿用 `src/evaluation/__init__.py` 的 `__version__`。

## [0.1.1] - 2026-08-29

### 新增
- **METEOR 辅助指标评测链路打通**：
  - 捕获 `pycocoevalcap.Meteor.compute_score` 的逐样本分数，写入 per-sample 结果与
    `per_sample_metrics.csv`（新增 `METEOR` 列；此前只取 corpus 分，逐样本被丢弃）。
  - `scripts/evaluate_reports.py` 在结果汇总中打印 METEOR corpus 分值，与 BLEU/ROUGE 并列。
- **`src/data/report_parser.py`（冻结 v0.1.1）补入仓库**：`clinical_metric.py` 依赖它，
  但此前 `.gitignore` 未锚定的 `data/` 模式递归匹配了 `src/data/`，导致全新 clone 无法运行
  Clinical。现该文件已提交（与原文件字节级一致，sha256
  `d2cd6bf54569cfc2edbbf5e59641091ddbe84f50cedcd30e71f5f3f7ed1f1504`）。

### 修复
- `.gitignore`：目录模式 `data/`、`tests/`、`outputs/`、`赛题资料及数据集/` 全部锚定到仓库根
  （`/data/` 等），避免误伤 `src/data/` 这类嵌套目录。

### 文档
- 根 `README.md` 与 `src/evaluation/README.md` 更新 METEOR 说明：**仅需 `java` 在 PATH 上**
  （`meteor-1.5.jar` 与 `data/paraphrase-en.gz` 已随 pycocoevalcap 打包，无需单独安装
  Stanford Parser），并给出 apt（需 sudo）与便携版 JRE（免 root）两种安装方式。

### 验证（本机 Ubuntu 22.04 + Temurin JRE 17）
- Val 347 样本（report_1 基线）：**METEOR = 0.1447（status: ok）**；
  BLEU-4 = 0.0127、ROUGE_L = 0.294（与此前一致，未受影响）；Clinical 正常。
- 小样本判别力：完全一致 → METEOR 1.0；否定错误 → 0.437；同义改写 → 0.623。
- 单元测试 57/57 通过。

## [0.1.0] - 2026-08-28

### 新增（初始版本：代理评测系统 Proxy Evaluation）
- `src/evaluation/`：`tokenizer.py`（唯一分词入口）、`text_metrics.py`（BLEU-1/2/3/4 +
  ROUGE_L + METEOR）、`formatter.py`（GBK res.csv 生成）、`validator.py`（P0 提交校验）、
  `clinical_metric.py`（P1 临床一致性指标）、`README.md`。
- `scripts/evaluate_reports.py`、`scripts/make_submission.py`。
- 指标：**BLEU-4（主指标，决定排名）** + BLEU-1/2/3 + ROUGE_L；Clinical (P1) 分析辅助。
- METEOR 当时因本机无 Java，默认 `meteor_status: skipped`（不阻塞主链路）。

### 说明
- 本仓库为 **Proxy Evaluation**，非官方 `testOffLine.py`；拿到官方脚本后需 proxy ↔ official
  对齐（见根 README「待官方确认清单」）。
