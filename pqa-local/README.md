# pqa-local：PaperQA 持久化文献检索

这个目录把「建库」和「提问」拆成两步：

1. `build_index.py`：一次性解析、切块、embedding，把结果存成 `index/docs.pkl`。
2. `ask_index.py`：每次提问只加载缓存、只 embedding 问题本身，不再解析 PDF、不再 embedding 全部 chunk。

原来的 `qa.py` 每次运行都会新建 `Docs()` 并对每个 PDF 调用 `aadd`，因此每次都会重新解析、切块、embedding，还会让 LLM 重新生成 citation（docname 也会跟着变）。这就是 token 消耗大、结果不稳定的主要原因。

## 快速开始

在 `pqa-local` 目录下：

```powershell
# 可选但推荐：支持加密 PDF 和 pypdf 失败后的备用解析
.\.venv\Scripts\python.exe -m pip install cryptography -e ../packages/paper-qa-pymupdf

# 1) 建库（默认 ../../PDF；以后只处理新增/修改的 PDF）
.\.venv\Scripts\python.exe build_index.py --allow-low-entropy-text

# 2) 提问（每次只加载缓存 + embedding 一个问题）
.\.venv\Scripts\python.exe ask_index.py "这份语料库主要涵盖哪些研究主题？"

# 3) 只看检索结果，不调用 LLM 生成答案（最省钱，适合调关键词）
.\.venv\Scripts\python.exe ask_index.py --retrieval-only -k 10 "research methods"
```

`--allow-low-entropy-text` 只影响 PaperQA 的「文本熵」合法性检查；它不能替代扫描件 OCR。

当前 `index/docs.pkl` 的实测结果：

| 指标 | 数值 |
| --- | --- |
| PDF 文件 | 1,228 篇 |
| 成功索引 | 1,212 篇（另有 10 篇内容重复） |
| chunk 数 | 17,446 |
| 缓存大小 | 244.8 MB |
| 向量矩阵 | 约 68.2 MB（17,446 × 1024 × 4B） |
| 待 OCR/修复 | 6 篇 |

常用参数：

```powershell
# 同时索引旧语料和 papers 两个目录
.\.venv\Scripts\python.exe build_index.py --corpus ..\..\文件 .\papers

# 只索引前 2 篇做小规模测试
.\.venv\Scripts\python.exe build_index.py --rebuild --limit 2

# 减少每次问答检索/摘要的 chunk 数（默认 10）
.\.venv\Scripts\python.exe ask_index.py --evidence-k 5 "your question"
```

## B：agent 多轮检索（持久化）

如果你要的是“agent 反复搜索、补证据、最后回答”，用这一组：

```powershell
# 第一次：从 index/docs.pkl 建 Tantivy 索引（不调用 embedding / LLM）
.\.venv\Scripts\python.exe build_agent_index.py

# 之后：每次加载索引，rebuild_index=False，agent 多轮检索
.\.venv\Scripts\python.exe agent_ask.py "这份语料库主要涵盖哪些研究主题？"

# 控制 agent 的搜索轮数和每轮证据数
.\.venv\Scripts\python.exe agent_ask.py --search-count 6 --max-timesteps 10 --evidence-k 6 "your question"
```

`build_agent_index.py` 直接从 `docs.pkl` 生成 Tantivy 全文索引和每篇论文的 chunk 缓存，**不会重新 embedding，也不会重新让 LLM 生成 citation**。当前全量同步实测约 85 秒，索引 166.8 MB。

`agent_ask.py` 的实测行为（中文问题）：

```text
Tools called: paper_search, paper_search, gather_evidence, gather_evidence,
              paper_search, gather_evidence, paper_search, gather_evidence,
              gather_evidence, gen_answer, complete
Contexts used: 24
Wall time: 23.2s
Prompt tokens: 57,721
Output tokens: 4,524
Status: success
```

也就是说，agent 确实会多轮搜索，而不是只做一次 top-k。

### 中文关键词问题

Tantivy 默认按空格/标点分词，中文长串会变成一个 token，直接 `paper_search("不确定性")` 往往召回为 0。`cjk_search.py` 在**建索引和查询时**都把中文切成重叠 bigram：

```text
不确定性 -> 不确 确定 定性
能源系统 -> 能源 源系 系统
```

这样中文问题也能命中中文论文，同时英文查询保持不变。

### 新增 PDF 后

```powershell
.\.venv\Scripts\python.exe build_index.py --allow-low-entropy-text   # 更新 Docs 缓存（增量）
.\.venv\Scripts\python.exe build_agent_index.py                      # 更新 Tantivy 索引（增量）
```

两者都只处理新增/修改的论文。若删除了论文或想彻底清理旧索引，运行 `build_agent_index.py --rebuild`。

## C：多轮对话 + 任务路由

`agent_qa.py` 是生产级对话入口：加载 `index_agent` 持久索引（`rebuild_index=False` + CJK 补丁），支持**会话内多轮上下文**、**标准/综述双模式**，以及**任务路由**——系统先识别问题类型，输出形式跟着任务走，而不是一律生成综述：

| 任务类型 | 触发示例 | 输出 |
| --- | --- | --- |
| 文献查找 | “找出所有提到 X 的文献” | 结构化文献证据清单（每篇一次：作者/年份/题名、相关表述、原文摘录、PDF 页码、支持类型），不承诺穷尽 |
| 反对意见检索 | “有没有文献质疑或反对 X？” | 按 直接批评/方法或边界批评/间接挑战/替代理论框架 四类报告；检索截断时只给阶段性结论，不写“文献中不存在” |
| 计数追问 | “你上面总共找到了几篇？” | 直接对上一轮保存的文献集合去重计数（DOI 优先，其次规范化题名/文档标识），逐篇编号 + 分级统计，**不重新检索** |
| 成员追问 | “《某篇论文的题名》上面有这篇吗？” | 只对照上一轮答案**实际引用**的文献集合回答“有/没有”；答“没有”时附“上一轮未列出，不代表该文献不相关或语料库中没有”，**不重新检索、不查其他论文的参考文献表** |
| 综述 | 明确要求“综述/研究现状/综合分析” | 分节综述（原 review 模板） |
| 其他 | 常规问题 | 常规问答 |

档位（standard/review）只控制检索广度（search_count/evidence_k/max_sources 等）；输出模板由任务意图决定。

**引用后处理**（`citation_fix.py`）：答案打印前会把 PaperQA 的 `(文件名 pages X-Y)` 内联引用改写成可读形式 `作者 年份(PDF第X–Y页)`；所有被正文引用的文献都通过 pypdf 只读提取前两页元数据（不可读文件名如 `6.pdf` 必要时再加一次小 LLM 调用），结果缓存到 `index/citation_manifest.json`。答案中 LLM 写下的「年份/题名/作者元数据缺失」诚实标记，若 manifest 已从该文献首页核验到对应数据、且该条目的引用唯一绑定这篇文献，则自动回填（跨文献歧义时保留标记，不猜测）。无法解析时保留完整文件名和路径并标注"元数据缺失"。页码为 PDF 顺序页码，能识别时附印刷页码。文末参考文献由正文实际引用重建，保证一一对应。

**一致性检查**（`answer_lint.py`）：统一同一统计的计数口径（同一数字前后不得矛盾）；对“文献中不存在”类全称否定强制附加“本轮检索”限定；已打印 PDF 页码时删除“页码未给出/缺失”类矛盾表述（印刷页缺失的诚实说明保留）。

**FIND 多路检索 + 检索审计**（`retrieval_audit.py` + `task_router.find_retrieval_plans`）：FIND 问题先把原问题的向量检索与主题多路查询（如某主题：8 条关键词 + 4 条带引号短语）的候选并集预置进语料池，再交 agent 取证；向量路径只调用一次问题 embedding，不调用答案模型。每次 `paper_search` 和预置检索的 query、候选、排名、dockey 都会被记录，运行后写入 `audit/audit_*.json`，可查每篇文献的旅程（候选 best_rank/hit_by → 是否进入证据 → 是否被正文引用 → 排除原因，未知原因如实标“未说明”）。审计不记录 API key，不保存整篇论文。

**按论文评测召回**（`retrieval_check.py`）：`--expect` 指定已知相关论文的 docname 片段（可重复），分别比较关键词、向量、可选短语和 RRF 混合排序的 `Recall@k`，并区分未入库与已入库但未召回。它只计算已标注论文的召回，不据此承诺全库穷尽；不生成答案。

**引文逐字核验**（`citation_fix.verify_quoted_excerpts`）：答案中带引号的英文摘录必须是证据文本的逐字子串（容忍换行、空白和排印字符差异，核验前剥离内嵌引用标注）；不匹配时去掉引号并把“原文摘录”降级为“释义”。统计区“引用能回读到证据片段”与“引号内容通过逐字核验”分列报告。

**回归测试**：`tests/`（离线，不调用 API）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -o addopts="" -p no:cacheprovider
```

**多轮上下文**：非首问（计数类和成员类除外）会先用 LLM 把追问改写成独立问题（打印「改写后：…」可检查），再走 agent 检索。历史只存于当次会话内存，`/new` 清空，进程退出即弃。

斜杠命令：`/review` `/standard` 切档、`/mode` 查看当前参数、`/new` 清历史、`/history` 看历史、`/exit` 退出、`/help` 帮助。

双模式参数对比（PaperQA 默认值 → 标准档 → 综述档）：

| 参数 | 默认 | 标准档 | 综述档 | 作用 |
| --- | --- | --- | --- | --- |
| `search_count` | 8 | 12 | 40 | 每次 paper_search 引入论文数 |
| `evidence_k` | 10 | 15 | 40 | 每次 gather_evidence 摘要 chunk 数 |
| `answer_max_sources` | 5 | 15 | 40 | 最终答案最多引用来源数 |
| `agent_evidence_n` | 1 | 5 | 8 | agent 每轮看到的证据数 |
| `max_timesteps` | 无 | 14 | 24 | agent 步数上限 |
| `answer_length` | ~200 词 | ~600 字 | 1500-2500 字分节 | 答案长度 |

综述档还会覆盖两个 prompt：`prompts.qa` 要求中文分节综述 + 逐句引用 + 文末参考文献清单；`agent.agent_prompt` 要求 agent 用 4-6 个不同角度 query 多次检索并翻页。

成本参考（DeepSeek input $0.27/M、output $1.10/M）：标准档约 **$0.02-0.03/次**，综述档约 **$0.10-0.15/次**（prompt caching 命中后更低）。

## 什么时候需要重建

`index/docs.pkl` 与以下配置绑定，任一项变化都需要 `--rebuild`：

- paper-qa 版本
- embedding 模型（当前 `openai/BAAI/bge-m3`）
- chunk 大小/overlap（当前 5000/250）
- PDF 解析器
- multimodal 设置

**新增/删除/修改 PDF 不需要 `--rebuild`**：直接重跑 `build_index.py`。它按文件 MD5 增量处理，并默认 `--prune` 删除语料里已经不存在的文档。

## 设计要点

- **向量库**：`NumpyVectorStore`。当前 17,446 个 chunk、1024 维，检索矩阵约 68.2 MB，精确余弦检索足够快；不需要 Qdrant。
- **缓存**：pickle 整个 `Docs`（官方文档明确支持用 pickle 缓存 embeddings）。写盘前清空 `texts_index`，因为它的 `texts_hashes` 依赖 Python 进程级字符串哈希；加载后 PaperQA 会用已保存的 embedding 在本地重建矩阵，不调用 API。
- **稳定的 docname/citation**：建库时不传 `citation=None`，而是用文件名 + 文件夹 ID，避免每次让 LLM 生成不同 citation。切块和 docname 固定，检索结果可复现。
- **增量/并发**：按排序后的路径顺序逐篇添加，保证 docname 去重结果稳定；定期保存，失败可续跑。

## 已知限制

- 当前只索引 PDF；`.docx/.xlsx` 需要 `unstructured` 依赖，`.doc` PaperQA 不支持。
- 仍有 6 个 PDF 未收录：1 个空文件，5 个无可用正文（备用解析仅得到 0 或 12 个字符），需要 OCR 或修复原文件。详见 `index/build_report.json` 的 `errors`。
- PaperQA 有一个「文本熵」合法性检查，会把字符种类特别多的中文论文误判为非文本（`This does not look like a text document`）。对这类有实际正文的 PDF，可加 `--allow-low-entropy-text` 重跑；扫描件仍需 OCR。
- 引用显示为「文件名 [文件夹ID]」，不是正式文献格式；如需作者/年份/DOI，可后续接 manifest 或改用 `DocDetails`。
- 综述档（`agent_qa.py --review`）为追求覆盖面把 `search_count`/`evidence_k`/`answer_max_sources` 都提到 40，单次问答成本约 $0.10-0.15，是标准档的 5 倍左右；写综述再用，日常问答用标准档即可。
- `.env` 里有真实 API key，不要提交到 Git 或发给别人。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `build_index.py` | 一次性/增量建库（Docs 缓存）；单篇 pypdf 失败可用 PyMuPDF 备用解析，正文过少则要求 OCR |
| `retrieval_check.py` | 不生成答案，按标注论文比较关键词/向量/短语/混合检索的 Recall@k |
| `ask_index.py` | 加载缓存并做一次性问答（A） |
| `build_agent_index.py` | 从 Docs 缓存建 Tantivy 索引（B，不重新 embedding） |
| `agent_ask.py` | agent 多轮检索问答（B，单次调试/基准工具） |
| `cjk_search.py` | 中文 bigram 索引/查询，修复 Tantivy 中文召回 |
| `task_router.py` | 任务意图识别（文献查找/反对意见/计数/综述/问答）与对应提示词 |
| `paper_ledger.py` | 每轮文献台账：保存上一轮命中文献，供计数追问去重复算 |
| `citation_fix.py` | 引用后处理：pqac 键替换、可读引用改写、元数据清单、参考文献重建 |
| `answer_lint.py` | 答案一致性检查（计数口径统一、否定结论限定） |
| `index_cache.py` | pickle 缓存读写与版本校验 |
| `scan_corpus.py` | 只读语料体检（页数、字符数、失败 PDF、重复文件） |
| `check_doc.py` | 单篇索引质量抽查（chunk 文本预览、向量/双索引收录检查，零 API） |
| `tests/` | 离线回归测试（路由/台账/引用/一致性，不调用 API） |
| `index/docs.pkl` | 生成的索引缓存（不要手动编辑） |
| `index/citation_manifest.json` | 数字/不透明文件名的元数据解析缓存（自动生成） |
| `index/build_report.json` | 建库报告：每篇 chunk 数、耗时、失败原因 |
| `qa.py` | 原始 demo + 公共 `build_settings()`（每次重新 embedding；仅作对照） |
| `agent_qa.py` | 多轮对话入口（任务路由 + 标准/综述双档，见上文「C」节） |
