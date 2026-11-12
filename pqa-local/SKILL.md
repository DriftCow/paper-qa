---
name: pqa-local-lit-screening
description: 本地文献库全库筛查技能。当用户要求"找出所有与某主题相关的文献"、"全部列出相关论文"、系统性文献综述、或需要可验证的页码级引用时使用。通过确定性混合检索（BM25+向量+RRF）保证查全率，Agent 仅负责查询扩展、批量判定与综合写作；引用四要素在判定当轮落盘，判定与写作分会话执行，杜绝压缩后重取原文。
---

# pqa-local 文献全库筛查技能

## 何时使用本技能

**使用**：用户要"找全/列出所有相关文献"、做系统性综述筛查、需要 Recall 优先的检索、要求引用必须带页码和原文。
**不使用**：普通快速问答（一句话答案即可，无需查全）；修改本仓库代码。

## 角色分工（核心原则）

你（DeepSeek Harness）是**大脑**，本地脚本是**手脚**：

- ✅ 你来干：生成查询计划 JSON、批量判定（核心度/立场）、综合写作 <!-- 改：逐篇→批量 -->
- ❌ 你干不了（必须交给脚本）：全库 3 万 chunks 的 BM25/向量混合检索、RRF 融合排序、证据片段定位（精确页码与原文，这是防幻觉引用的唯一保障）
- 🚫 禁止：凭记忆或推测输出文献清单、编造页码或原文——所有文献结论必须能追溯到 inspect_run.py 展示的 span 文本

## 成本模型与三条铁律（先读这里）<!-- 改：新增。先让执行者理解成本从哪里来，再讲流程 -->

长任务的真实成本 ≈ 各轮输出之和 + **轮次数 × 已累积上下文**。轮次越碎、历史越长，每轮越贵（历史被反复重发，成本随轮次近似平方增长）；一旦触发压缩（compaction）丢失原文，被迫"重取"是实测中最贵的单项浪费。因此本技能一切流程设计服从三条铁律：

1. **单次触碰原则**：每个证据 span 最多取一次。判定当轮就把引用四要素（dockey、chunk_name、页码、逐字摘录）写入 `dsh_verdicts.md`；此后任何阶段——包括上下文被压缩后、写作时——只读该文件，**绝不重取原文**。
2. **批量优先原则**：能一条命令取多篇，就不逐篇开轮。`--paper` 支持一次多篇（批量模式）；`--grep` 一条命令可覆盖多篇候选且命中行自带完整引用要素。轮次每合并 5 倍，历史重发成本近似降 5 倍以上。
3. **落盘即交接**：`dsh_verdicts.md` 是判定阶段与写作阶段之间的**唯一交接物**。判定结束时它必须"自给自足"——写作阶段只读它就能产出全部结论与引用，无需再调用任何查看命令。

## 前置条件

1. 已构建索引：`pqa-local\index\docs.pkl` 与 `pqa-local\index_agent\lit\` 存在。若不存在或语料已更新，先运行（见"索引重建"）。
2. `.env` 已配置（脚本需调用 embedding API 计算查询向量）：`SILICONFLOW_API_KEY` 等。
3. 工作目录：所有命令在 `d:\claudeHome\codeLearn\PaperQA\paper-qa\pqa-local` 下执行。

## 标准工作流（四步）

### 第 1 步：生成查询计划 JSON

针对用户问题，生成 6～12 条查询变体，写入 `reviews\<run-id>\query_plan.json`：

```json
{
  "queries": [
    {"query_id": "plan001", "text": "district heating decarbonization", "channel": "both", "language": "en", "family": "synonym"},
    {"query_id": "plan002", "text": "区域供热 脱碳", "channel": "both", "language": "zh", "family": "synonym"},
    {"query_id": "plan003", "text": "\"heat pump\"", "channel": "phrase", "language": "en", "family": "facet"}
  ]
}
```

**字段约定**（`channel`/`language`/`family` 可省略，默认 `"both"`/`"auto"`/`"expanded"`）：

| 字段 | 取值 | 说明 |
|---|---|---|
| `query_id` | 必填字符串 | 唯一 ID，如 `plan001`。原始问题会被脚本自动以 `q000` 追加，**无需**写入 |
| `text` | 必填字符串 | 查询文本。避免过长整句，保留关键术语 |
| `channel` | `both` / `keyword` / `phrase` / `vector` | `both`=BM25+向量双路；`phrase`=Tantivy 精确短语（专业术语、型号、缩写用它）；`keyword`=仅 BM25；`vector`=仅向量 |
| `language` | `zh` / `en` / `mixed` | 仅供溯源标注 |
| `family` | `synonym` / `facet` / `broader` / `narrower` / `user` | 变体类型标注 |

**生成原则**：中英文都要覆盖；包含同义表达、概念拆解、上位概念（如"能源转型"）、下位概念（如"热泵"、"余热回收"）；缩写与全称各一条；`phrase` 通道用于精确术语并给文本加引号。同时记住这些关键词，第 3 步的 grep 三角定位要从这里挑。

`run-id` 命名建议：`dsh_主题关键词_日期`（如 `dsh_dh_decarbon_20261108`）。始终显式指定 `--run-id`，不要让脚本用时间戳。

### 第 2 步：调用本地检索（cmd 命令模板）

```bat
cd /d d:\claudeHome\codeLearn\PaperQA\paper-qa\pqa-local
.venv\Scripts\python.exe find_papers.py "用户原始问题" --query-plan reviews\<run-id>\query_plan.json --retrieve-only --run-id <run-id>
```

**必须使用完整路径** `.venv\Scripts\python.exe`，不要用 `pqa` 或裸 `python`。

关键参数：

| 参数 | 用途 |
|---|---|
| `--query-plan` | 加载第 1 步生成的计划，冻结查询 |
| `--retrieve-only` | 只检索不调用脚本内置复核 LLM——判定由你来做 |
| `--run-id` | 固定输出目录名 |
| `--depth 80` | 每路每查询召回深度（默认 80；查全要求极高时可提高到 150） |
| `--strict-corpus` | 全部已索引论文强制进入候选（漏斗最保险，但候选量=全库，仅小语料或极高查全要求时用） |
| `--phrase "精确短语"` / `--query "额外查询"` | 可选，等效于追加到查询计划 |

成功标志：终端打印 `语料预检通过：N 篇 / M chunks` 和 `结果目录：...reviews\<run-id>`。

### 第 3 步：批量判定（必须用 inspect_run.py，禁止自写脚本）

**禁止事项（防上下文爆炸与重复取文，本条是硬规则）**：

- 🚫 禁止直接读取 `evidence.json` / `result.json`：它们是数 MB 的原始 PDF 文本，一轮就能撞爆上下文并触发 full compaction（实测代价：166 秒 + 1.2 万 token，且连续发生两次）
- 🚫 禁止自己写 Python/shell 去解析这些 JSON：自写脚本容易把整段命中原文 `print` 进上下文（如 `print(text[:6000])`），且试错每多一轮报错就多烧一轮 token
- 🚫 禁止对**已落盘判定**的论文用 `--span` / `--paper` 重取原文——压缩后重取引用原文是实测最大单项浪费（约占全程成本三分之一），摘录在判定当轮就已写入 `dsh_verdicts.md`，重取即违规 <!-- 改：新增 -->
- ✅ 只使用 `inspect_run.py`：它的所有输出都有硬上限（片段截断、命中数封顶、批量封顶），想灌爆上下文都做不到

**渐进式查看命令**（由粗到细；3、4 为日常主力）：

```bat
:: 1) 运行概览：候选数、counts、errors
.venv\Scripts\python.exe inspect_run.py reviews\<run-id> --meta

:: 2) 候选总表：rrf_score 降序，含题名/年份/页码（默认前 60 篇，上限 300）
.venv\Scripts\python.exe inspect_run.py reviews\<run-id> --list 60

:: 3) grep 三角定位：一条命令覆盖多篇候选，命中行自带 dockey/span_id/chunk_name/页码 + ±80 字符逐字窗口
.venv\Scripts\python.exe inspect_run.py reviews\<run-id> --grep net energy cliff

:: 4) 批量 --paper：一次 4~6 篇；批量模式每篇最多 4 片段、每片段默认 240 字符、全程最多 24 片段
.venv\Scripts\python.exe inspect_run.py reviews\<run-id> --paper <dockey1> <dockey2> <dockey3> <dockey4>

:: 5) 单篇完整视图（仅批量截断仍无法判定时）：元数据 + 每个证据片段截断文本（默认 400 字符/片段，--chars 上限 1200）
.venv\Scripts\python.exe inspect_run.py reviews\<run-id> --paper <dockey>

:: 6) 按需下钻：单 span 更长原文（默认 800，--chars 上限 2400）
.venv\Scripts\python.exe inspect_run.py reviews\<run-id> --span <span_id> --chars 800
```

**判定流程（grep 优先，按批推进）**：<!-- 改：原"对疑似 core 逐篇 --paper"改为 grep 优先 + 批处理 -->

1. `--meta` 确认运行健康（`errors` 为空再继续）
2. `--list 60` 拿全景，按 rrf_score 排名从高到低推进判定；需要看 61–150 名时再 `--list 150` 一次性拉全（约 2 万字符，属全景一次性投入，允许超出单轮预算，不许反复翻页）
3. **grep 三角定位**：从查询计划挑 3~6 个高区分度关键词（主题词 + 命题词），每个跑一次 `--grep`。命中行自带四要素与 ±80 字符逐字窗口：
   - 题名 + 窗口足以确认 core 与 stance 的论文 → **当轮直接落盘**（窗口连续片段即摘录，可截取加省略号），不再 `--paper`
   - 窗口不足以判定、或题名疑似相关但未命中的 → 记入待批清单
4. **批量 `--paper`**：每批 4~6 篇，判 centrality/stance 并从输出直接复制摘录落盘；批量片段 240 字符通常同时够判定与摘录，不够的转入第 5 步
5. 单篇 `--paper` / `--span` 仅用于批量截断后仍存疑者（`--chars` 不超过 1200），取完当轮落盘
6. **落盘节奏**：每批判完立即追加写盘（一批 = 一次落盘轮，落盘轮输出近零），不要攒 10 篇再写

**落盘格式（`reviews\<run-id>\dsh_verdicts.md`）**：<!-- 改：行格式从五字段扩为九字段，把写作所需的一切（含逐字摘录）在判定当轮落盘 -->

文件头：

```
# 判定进度 <run-id>
问题：<原始问题或命题>
格式：dockey | centrality | stance | 一句话理由 | span_id | chunk_name | p.页码 | "逐字摘录≤200字符" | 短题名
```

行规则（字段内禁用竖线符 `|`）：

- **core 行**：stance 与摘录必填。"一句话理由"就是写作时要下的结论；摘录必须在**判定当轮**从 inspect_run.py 输出逐字复制（`--paper`/`--span` 的截断文本、`--grep` 的窗口均可），禁止凭记忆、禁止改写、禁止拼接不相邻句子
- **background / irrelevant 行**：stance 填 `n/a`，摘录可省略，span_id 保留
- **uncertain 行**：同 core 格式，理由注明"需人工复核原文"
- 修正旧判定 = 追加新行覆盖（行首加 `> 更正`），不删旧行

**覆盖账本（判定结束时必须写入文件末尾）**：<!-- 改：新增。召回边界从"凭感觉声明"变成可核对的账本，也为写作阶段备好统计与边界文案 -->

```
## 覆盖账本
- 判定范围：--list 前 N 名已全部判定（core X / background Y / irrelevant Z / uncertain W）
- grep 兜底扫描：ranks N+1 至候选末尾，用关键词 <逐个列出> 扫描；主题命中且未在前 N 的论文：<无 / 列出并已逐篇判定>
- 召回边界：检索深度 <depth>、strict=<是/否>；未逐篇判定区间的遗漏风险声明
```

**`--retrieve-only` 占位符警告**：`result.json` 的 `verdicts` 全是占位符（`centrality: "uncertain"`、`reason: "retrieve-only：尚未进行论文复核"`），**不是真实判定**——判定是你的工作。

### 第 4 步：写作会话（与判定会话分离）<!-- 改：全新。写作不再背着百余轮判定历史，只读交接物 -->

**判定会话收尾**：确认覆盖账本已写入后，向用户报告判定完成，并建议开启**新会话**执行写作，给出续接指令模板：

```
按 pqa-local-lit-screening 技能第 4 步，读取 reviews\<run-id>\dsh_verdicts.md 完成写作
```

**写作会话只允许三类操作**：

1. 直接 Read `reviews\<run-id>\dsh_verdicts.md`——它是唯一允许直读的运行产物（Agent 自写、体积有界，典型 ≤4 万字符），Read 一次即可
2. 可选调用一次 `--meta` 核对候选总数
3. 写最终答案

- 🚫 禁止再调用 `--paper` / `--list` / `--grep` 逐篇查看。**唯一例外**：发现某 core 行摘录缺失或"理由-摘录"不匹配时，允许单次 `--span <span_id> --chars 1200` 定点补取，补完立即落盘
- 若无法开新会话：写作前停止一切查看命令，直接依据落盘文件写作（历史仍背在身上，但不再增长）

**输出要点**：

- 先给统计概览（core X 篇，其中支持/反对/中性各多少）——数字照抄覆盖账本
- core 论文必须全部列出，不得截断；background 论文可汇总为附录（行内短题名已备好）
- 召回边界声明照抄覆盖账本，不凭记忆改写

## 上下文预算纪律（防 compaction）<!-- 改：预算从"单轮 3000 字符"（实测名存实亡）改为"轮次 × 单轮"双预算 -->

- 单轮新增信息预算 ≤8000 字符（批量 `--paper` 一轮约 6~8k、`--grep` 满额约 7k）。超出时缩小单轮范围（减批量篇数、减 `--chars`），**不是**退回逐篇模式
- 判定阶段轮次预算：约 100 篇判定规模 ≤40 轮；每轮只做一件事（一次批量取证据，或一次落盘）
- 例外：`--list 150` 全景一次拉全（约 2 万字符），一次性投入，换取后续无需翻页
- 摘录只需 ≤200 字符的连续片段；判定当轮落盘；同一 span 最多取一次（单次触碰）
- 上下文一旦被压缩：Read `dsh_verdicts.md` 一轮即可恢复全部进度**与引用**，绝不重读或重取任何大文件和已判论文原文
- 绝不把 `evidence.json` 的 span 全文复制进回复或新建文件

## 判定分级标准

### 核心度 centrality（每篇必判）

| 等级 | 判据 |
|---|---|
| `core` | 论文实质性地研究或直接讨论了目标主题/命题；摘掉该篇，综述会缺一块 |
| `background` | 仅背景性提及、使用相关方法但主题不同、或仅相关领域的综述性引用 |
| `irrelevant` | 证据片段与主题无实质关联（纯词面命中） |
| `uncertain` | 证据片段不足以判断（可注明需人工复核原文） |

### 立场 stance（仅对 core 论文，且问题含可判断命题时）

| 等级 | 判据 |
|---|---|
| `support` | 论文自身的工作/结论直接支持命题 |
| `oppose` | 论文证据直接反驳或质疑命题 |
| `mixed` | 同一篇内正反证据并存 |
| `neutral` | 讨论了命题但未表态 |
| `uncertain` | 证据不足以判断立场 |
| `not_applicable` | 主题检索类问题（无命题），全部标此值 |

**防作弊**：立场只看该论文**自己的研究发现**，不是它转述别人观点；`support` 必须有 span 原文支撑，引用全文某句话断章取义不算。

## 引用规范（强制）

每条文献结论必须包含四要素，且可验证：

```
[序号] 题名（作者, 年份）｜dockey: <dockey>｜<chunk_name>（PDF第<pages>页）
    依据："…span 原文中逐字摘录的连续片段…"
```

规则：

- `dockey`、`chunk_name`、`pages`、摘录原文必须逐字来自 `inspect_run.py` 展示的对应 span（判定阶段）或 `dsh_verdicts.md` 已落盘行（写作阶段），**一字不差**；`题名/作者/年份/DOI` 来自 `--list` / `--paper` 的输出
- `--grep` 的 ±80 字符窗口、批量 `--paper` 的截断文本同样是 inspect_run.py 展示的 span 文本，可直接用作摘录 <!-- 改：明确 grep 窗口的摘录资格 -->
- 禁止改写原文后再加引号；改述性内容放在引号外并明确标注"概述"
- 一篇论文引用多个片段时，逐个标注页码
- 若 span 原文过长，可截取连续片段加省略号，不得拼接不相邻句子
- 写作阶段的摘录**只允许**来自 `dsh_verdicts.md` 已落盘行；缺要素 = 判定不完整，回到判定会话补齐，不得凭记忆补写 <!-- 改：新增 -->

## 索引重建（仅在语料变更时）

```bat
cd /d d:\claudeHome\codeLearn\PaperQA\paper-qa\pqa-local
.venv\Scripts\python.exe build_agent_index.py
```

完成后再跑一次检索预检确认：`.venv\Scripts\python.exe find_papers.py "预检" --preflight-only`。

## 排障

| 现象 | 处理 |
|---|---|
| `Tantivy index is incomplete` / `Docs cache and Tantivy index differ` | 索引缺失或不一致，运行 `build_agent_index.py` |
| 终端出现 `errors` 列表非空 | 某路查询失败（如 embedding API 限流），告知用户后用相同 `run-id` 直接重跑（脚本会覆盖输出目录）；不要与 `--resume` 混用 |
| 候选明显偏少 | 查询计划覆盖不足——增加同义/上下位变体，或提高 `--depth`，换新 `run-id` 重跑 |
| 输出乱码 | 正常（脚本已设 UTF-8 stdio）；坚持乱码时检查终端代码页 `chcp 65001` |
| Token 暴涨 / 触发上下文压缩（full compaction） | 你违反了读取纪律（直读大 JSON、打印原文、一轮连发多条 grep）。处置：Read `dsh_verdicts.md` 一轮恢复进度与引用，从最后一行继续；**已判论文禁止重取原文** <!-- 改：恢复成本从"35 篇逐个重取"降为"一次文件读取" --> |
| `--paper` 传多篇报 `unrecognized arguments` | 该环境 `inspect_run.py` 为旧版：退回单篇模式（一轮内可连续执行多条单篇命令），或同步新版脚本 <!-- 改：新增 --> |
