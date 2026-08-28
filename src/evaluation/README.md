# Proxy Evaluation V0.1

`src/evaluation/` 是赛题3（基于多视角眼表图像的颈动脉超声诊断报告生成）的
**代理评测系统（Proxy Evaluation）**。

> ⚠️ **NOT OFFICIAL**：当前模块不是官方 `testOffLine.py`，也不声称“完全复现官方评测”。
> 它用于 **Val 相对比较**、模型选择、消融实验，以及生成/检查合法 `res.csv`。
> 最终比赛前三天拿到官方 `testOffLine.py` 后，再进行 proxy ↔ official 对齐。

---

## 1. 模块用途

| 文件 | 职责 | 优先级 |
|---|---|---|
| `tokenizer.py` | 唯一分词入口：jieba + `medical_dict_final.txt` | P0 |
| `text_metrics.py` | pycocoevalcap：BLEU-1/2/3/4 + ROUGE_L + METEOR(optional) | P0 |
| `formatter.py` | 从 raw prediction 生成 `image_id,predicted_report` 的 `res.csv` | P0 |
| `validator.py` | 提交前自动检查 missing/extra/duplicate/empty/列名/GBK | P0 |
| `clinical_metric.py` | 复用 `report_parser` 的医学一致性分析（非官方排名指标） | P1 |
| `README.md` | 本文档 | - |

CLI 入口在 `scripts/evaluate_reports.py` 与 `scripts/make_submission.py`。

## 2. 为什么叫 Proxy Evaluation

- 官方只确认了 **评分库为 `pycocoevalcap`**、**只评 `report_2`**、**`BLEU-4` 为主指标**、
  **`res.csv` 必须 `GBK` + 空格分词**。
- 但我们尚未拿到官方 `testOffLine.py`（分词细节、标点处理、gts/res 格式、聚合方式等
  官方实现细节未最终确认）。
- 因此本系统在“官方已知规则”基础上做**尽量贴近**的实现，明确标记为 proxy，
  避免团队误把本地数字当线上最终分。

## 3. 官方评分规则摘要

- 最终评分 **只基于 `report_2`**；`report_1` 不参与评分（仅可辅助训练）。
- 提交文件 `res.csv`，只有两列：`image_id`、`predicted_report`。
- `predicted_report` 必须为 **jieba + medical_dict_final.txt 分词后空格连接**的字符串。
- 官方保存要求：
  ```python
  df_res = pd.DataFrame({'image_id': ids, 'predicted_report': res})
  df_res.to_csv("res.csv", index=False, header=True, encoding="gbk")
  ```
- 官方按 `image_id` 匹配（不是行序），必须：全部存在、不多、不少、不重复、非空。
- 指标：**BLEU-4（主指标，决定排名）**、METEOR（辅助）、ROUGE_L（辅助）。

## 4. BLEU-4 为什么是主指标

官方明确：“在模型训练调优及测试打榜阶段，**BLEU-4 将作为主要评价指标**”。
BLEU-4 依赖连续 4-gram 匹配，本质更偏**文本表面相似度**而非医学语义等价。
因此本项目：
- **不做同义词归一化**（`双侧→两侧`、`未见→未发现`、`毛糙→欠光滑` 等一律禁止）。
- 不做否定词/左右侧/斑块有无的归一。
- BLEU 必须基于**最终实际提交文本**计算。
- 所有模型选择、检索权重比较、消融实验的第一文本指标 = **BLEU-4**。

## 5. tokenizer 规则

- 唯一入口：`src/evaluation/tokenizer.py` 的 `tokenize_report()` / `tokenize_reports()`。
- 流程：
  ```
  raw Chinese report
    -> 清理输入格式（None/NaN/空串/首尾空格/连续空格/换行/\r\n）
    -> jieba + medical_dict_final.txt（用户词典）
    -> " ".join(tokens)
  ```
- **不**做：同义词替换、医学语义纠错、左右侧归一、否定词替换、模板润色。
- 观察到官方 gts.csv 行为并复刻（详见 README §13 差异清单）：
  - 全角标点 `，` `：` `。` `；` **保留**为独立 token。
  - `CDFI` 字母被移除，其后的 `：` 保留（约 65% 报告含 CDFI）。
  - `内-中膜`、`颈总动脉`、`颈内动脉` 等医学词典词保持单 token。

## 6. medical_dict_final.txt （医学词典）使用方式

字典文件混合两种分隔符：
- 绝大多数行 `word\tfreq`，例如 `精神\t606946`
- 少量行 `word freq tag`，例如 `囊性 99999 n`

`jieba.load_userdict` 只按**单个空格**切分，会把 tab 行整体当作一个词，因此本模块
自己解析字典并调用 `jieba.add_word(word, freq)` 加载（`load_medical_dict()`），
保证两种格式都正确生效。加载幂等。

## 7. GT 与 Prediction 必须用同一 tokenizer

```
GT raw  --same tokenizer-->  GT tokens
Pred raw --same tokenizer-->  Pred tokens
```

禁止 GT 一种切法、Pred 另一种切法。同时：
- 默认输入为 **raw report**，统一调用 `tokenize_report`。
- 若调用方明确传入 `already_tokenized=True`（已分词文本），**不会再次 jieba**，
  只做空白规整，避免二次分词导致不可控改变。

## 8. 如何运行 Val Evaluation

```bash
python scripts/evaluate_reports.py \
    --pred outputs/evaluation/val_preds_raw.csv \
    --metadata data/metadata.csv \
    --split val \
    --medical-dict 赛题资料及数据集/医学字典/medical_dict_final.txt \
    --output-dir outputs/evaluation/val_baseline
```

`--pred` 为两列 CSV：`id`（内部 `metadata.id`）、`predicted_report`（raw 文本）。
GT 取自 `data/metadata.csv` 的 `report_2`，只限 `--split`（默认 `val`）。
输出：
- `metrics_summary.json`
- `per_sample_metrics.csv`
- `clinical_summary.json` / `clinical_per_sample.csv`（P1，`--no-clinical` 关闭）

CLI 打印：
```
N              : 347
BLEU_1         : ...
BLEU_2         : ...
BLEU_3         : ...
BLEU_4         : ... (PRIMARY)
ROUGE_L        : ...
METEOR status  : ok / skipped
```

## 9. 如何生成 res.csv

```bash
python scripts/make_submission.py \
    --pred outputs/evaluation/val_preds_raw.csv \
    --metadata data/metadata.csv \
    --medical-dict 赛题资料及数据集/医学字典/medical_dict_final.txt \
    --output outputs/evaluation/res_val.csv \
    --validate --expected-split val
```

内部 join 主键为 `metadata.id`；输出 `image_id` 取 `metadata.official_id`
（即官方 JSON `id`）。**不要**假设 `metadata.id == image_id`。若 id 映射无法唯一
确定，formatter 会**直接报错**而不猜测。

## 10. 如何运行 submission validator

```python
from src.evaluation.validator import validate_submission
report = validate_submission("res.csv", expected_ids=[...])
```

校验项：文件存在 / GBK 可读 / 列名严格为两列 / id 无 NaN 无空无重复 / 与
expected_ids 相比 missing=0 且 extra=0 / predicted_report 无空无纯空格 /
分词 sanity check / GBK round-trip。全部通过打印 `Submission : VALID`，
否则 `FAIL` 并列出具体问题。

## 11. METEOR 为什么可能被跳过

METEOR 依赖 Java（Stanford Parser jar）。当前 Linux 环境若未安装 Java 或
初始化失败，`text_metrics` 会捕获异常并输出：

```
METEOR: skipped
reason: <异常信息>
```

**不会**因为 METEOR 阻塞 BLEU + ROUGE_L 主链路。BLEU-4 始终可用。

## 12. Clinical Metric 与官方排名的关系

`clinical_metric.py` 直接复用 `src/data/report_parser.py`（v0.1.1，冻结），
比较：`main_status / roughness / imt_thickening / left_plaque / right_plaque /
stenosis / flow_normal`，输出概念级 F1、clinical macro/micro F1、exact match，
以及错误分类（SIDE_ERROR / NEGATION_ERROR / PLAQUE_FALSE_POSITIVE /
PLAQUE_FALSE_NEGATIVE / IMT_ERROR / ROUGHNESS_ERROR / STENOSIS_ERROR /
FLOW_ERROR / WORDING_ONLY）。

**仅用于内部医学一致性分析，不参与官方 BLEU-4 排名。**
`unknown` 按 parser 既有定义处理，**禁止**擅自把 unknown 当 0。

## 13. 当前与官方 testOffLine.py 仍可能存在的差异

以下事项必须等官方 `testOffLine.py` 才能最终确认：

1. 分词细节：`CDFI` 移除、标点是否归一、`内-中膜` 等词是否与官方一致。
2. gts/res 传给 pycocoevalcap 的确切结构。
3. BLEU 平滑 / 聚合法（corpus vs per-sentence averaging）。
4. METEOR 的运行环境与版本。
5. `image_id` 的最终来源字段。

## Ground Truth 泄漏约束

`report_2`（Val）只允许作为 Evaluation GT。禁止进入 Template Bank、检索候选、
Val text embedding retrieval 输入、rerank 候选、模型预测输入。Template Bank 始终
**Train-only**，Evaluation 与 Retrieval 逻辑隔离。
