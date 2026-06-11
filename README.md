# 🚀 LLM Memory System — 三层记忆架构

## 📖 系统概述

基于大语言模型的**三层记忆管理系统**，支持从原始对话中自动提取、组织和检索长期记忆。

### 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Session Memory (L1)                       │
│  原始对话记录 | 精确时间戳 | 完整上下文 | ~189 条/用户        │
└───────────────────────┬─────────────────────────────────────┘
                        │ LLM 抽取（4h 窗口）
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                  Effective Memory (L2)                       │
│  结构化摘要 | 关键事实 | 主题标签 | 时间范围 | ~20 条/用户     │
└───────────────────────┬─────────────────────────────────────┘
                        │ LLM 决策（ADD/UPDATE/DELETE/NONE）
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                 Long-term Memory (L3)                       │
│  高度提炼 | 去重合并 | 长期固化 | 时间保留 | ~54 条/用户      │
└─────────────────────────────────────────────────────────────┘
```

**核心特性**：
- ✅ **时间保留**：所有层级保留原始时间戳，支持精确时间推理
- ✅ **降级策略**：QA 时 Long-term → Effective → Session 自动降级
- ✅ **批量处理**：高效的全量同步和增量更新
- ✅ **可视化 UI**：React 前端展示三层记忆关联关系

---

## 🛠️ 环境准备

```bash
cd /Users/kwjcyr/data/llm_memory

# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，修改 FRIDAY_APP_ID 和 LLM_MODEL 为你自己的值
```

**`.env` 关键配置项**：
```bash
FRIDAY_APP_ID=你的AppID
LLM_MODEL=gpt-4.1                    # 推荐模型
LLM_TEMPERATURE=0.1                   # 低温度保证稳定性
LLM_BASE_URL=https://aigc.sankuai.com/v1/openai/native/chat/completions
```

---

## 🚀 完整流程（7 步）

### Step 0: 准备输入数据

```bash
# 从 API 获取 Session Memory，导出为 JSON 文件
# 将 response 拷贝到 data/input/ 目录

ls data/input/
# 应该看到: group_0_caroline.json, group_1_gina.json, group_0_melanie.json, group_1_jon.json
```

---

### Step 1: 获取 Session Memory

```bash
python src/memory/session_memory/import_from_json.py --group group_1_gina
```

**输出**: `data/groups/group_1_gina/session/memories.jsonl` (~189 条)

**说明**: 将原始 API JSON 转换为标准 JSONL 格式，按时间升序排列。

---

### Step 2: Session Memory → Effective Memory（调 LLM）

```bash
python src/memory/effective_memory/extract_from_session.py --group group_1_gina
```

**输出**: `data/groups/group_1_gina/effective/effective_memories.jsonl` (~20 条)

**说明**:
- 按 4 小时窗口分组对话
- 每窗口调一次 LLM 抽取结构化记忆（topic, summary, facts, tags）
- 自动保留 `time_range` 时间信息

---

### ⭐ Step 3: Effective Memory → Long-term Memory（批量 LLM 决策）

```bash
python src/memory/longterm/effective_to_longterm.py --group group_1_gina --batch-size 10
```

**输出**:
- `data/longterm_memories.db` (SQLite，统一存储)
- `data/groups/group_1_gina/longterm/longterm_memories.jsonl` (JSONL 备份)

**说明**:
- 采用 **organizeParam 四操作模式**（ADD/UPDATE/DELETE/NONE）
- **批量处理**：一次性传入所有记忆，LLM 全局决策
- **智能匹配**：保留原始 effective memory 的完整元信息（包括 time_range）
- **数量提升**：从旧版 3 条 → 新版 **54 条**（18x 提升）

**高级选项**:
```bash
# 自定义批次大小（默认 10）
python src/memory/longterm/effective_to_longterm.py --group group_1_gina --batch-size 5

# 清空数据库重新生成
rm -f data/longterm_memories.db && python src/memory/longterm/effective_to_longterm.py --group group_1_gina
```

---

### Step 4: UI 查看 Memory 之间的关联

```bash
# 终端 1：启动后端 API
python server.py &   # 后端 :8000

# 终端 2：启动前端
cd frontend && npm run dev  # 前端 :5173
```

打开 http://localhost:5173 → **🗂 Memory Explorer** Tab

**功能特性**:
- 👤 **User 切换**：支持 Caroline / Gina / Melanie / Jon 多用户切换
- 📊 **实时统计**：Session / Effective / Long-term 数量展示
- 🔗 **关联连线**：贝塞尔曲线展示记忆间的来源关系
- ⏰ **时间排序**：支持升序/降序切换
- ✏️ **在线编辑**：支持 Long-term Memory 的 UPDATE/DELETE 操作

**API 接口**:
```
GET /api/sessions?user_id=gina          # Session Memory 列表
GET /api/effectives?user_id=gina         # Effective Memory 列表
GET /api/longtermemories?user_id=gina    # Long-term Memory 列表
GET /api/links?user_id=gina              # 三层关联关系图
PATCH /api/longtermemories/{id}           # 更新长期记忆
DELETE /api/longtermemories/{id}          # 删除长期记忆
```

---

### ⭐ Step 5: Eval — 三层记忆 QA 评测

#### 方式一：命令行评测脚本

```bash
python src/eval/run_eval.py --group group_1_gina
```

**输出**: `data/groups/group_1_gina/eval/`
```
├── locomo_responses.json   # 原始回答（含三层证据链）
├── locomo_judged.json     # 评分结果（LLM Judge + F1/BLEU-1）
└── locomo_grades.json     # 汇总统计（总分 + 分类得分）
```

**评测流程**（已升级为三层记忆）:
1. **Response 阶段**:
   - Layer 1: 从 Long-term Memory 检索高度提炼的知识
   - Layer 2: 从 Effective Memory 补充对话摘要
   - Layer 3: 从 Session Memory 补充原文和时间细节
   - 调用 LLM 基于完整的三层上下文回答问题

2. **Judged 阶段**: LLM-as-a-Judge (CORRECT/WRONG) + NLP 指标 (F1, BLEU-1)
3. **Grades 阶段**: 总体统计 + 5 类问题分类统计

**指标说明**:
| 指标 | 说明 | 范围 |
|------|------|------|
| **LLM Judge Score** | LLM 判断回答是否正确的准确率 | 0~1 |
| **F1 Score** | Token 级别的精确匹配分数 | 0~1 |
| **BLEU-1 Score** | Unigram 精确度 | 0~1 |

**分类统计**:
- **Single-hop**: 单跳事实查询
- **Temporal**: 时间推理（⏰ 新增时间保留后应显著提升）
- **Multi-hop**: 多跳推理
- **Adversarial**: 对抗性干扰
- **Open-ended**: 开放式问答

**高级选项**:
```bash
# 自定义检索数量（每层）
python src/eval/run_eval.py --group group_1_gina --top_k 10

# 多次 LLM Judge 评估（更稳定但更慢）
python src/eval/run_eval.py --group group_1_gina --judge-attempts 3

# 静默模式
python src/eval/run_eval.py --group group_1_gina --quiet
```

#### 方式二：UI 页面评测

打开 http://localhost:5173 → **📋 Eval** Tab

**功能**:
- 📝 问题列表展示（按类别过滤）
- 🔍 实时检索证据链（三层可视化）
- ▶️ 一键运行 QA（调用后端 `/api/eval/run`）
- 📊 分数显示（需先运行命令行评测）

**API 接口**:
```bash
# 运行单条 QA（返回三层证据链）
curl "http://localhost:8000/api/eval/run?question=...&user_id=gina&top_k=5"

# 返回格式:
{
  "question": "When did Gina launch her store?",
  "predicted": "Gina launched on 2023-01-29...",
  "confidence": 0.8,
  "three_layer_chain": {
    "longterm": [...],    // 高度提炼的事实
    "effective": [...],   // 对话摘要
    "session": [...]      // 原文+精确时间
  },
  "retrieval_stats": {"longterm": 3, "effective": 3, "session": 6, "total": 12}
}
```

---

### Step 6: Self-Evolving（PPO 优化）

```bash
python src/eval/self_evolve/ppo_fixed_memories.py
```

基于固定 effective memory，用 PPO 优化 top_k 和 fields 动作空间。

**优化目标**: 提升 Eval 的 LLM Judge Score 和 F1 Score。

---

## 📁 目录结构

```
data/
├── .env                              # LLM 配置
├── input/                            # 原始 API JSON 输入
│   ├── group_0_caroline.json
│   ├── group_1_gina.json
│   ├── group_0_melanie.json
│   └── group_1_jon.json
├── groups/                           # 按用户组组织的处理结果
│   └── group_1_gina/
│       ├── raw/                      # 原始数据备份
│       ├── session/                  # Session Memory (JSONL, ~189条)
│       │   └── memories.jsonl
│       ├── effective/                # Effective Memory (JSONL, ~20条)
│       │   └── effective_memories.jsonl
│       ├── longterm/                 # Long-term Memory (JSONL, ~54条)
│       │   └── longterm_memories.jsonl
│       └── eval/                     # 评测结果
│           ├── locomo_responses.json
│           ├── locomo_judged.json
│           └── locomo_grades.json
├── longterm_memories.db              # SQLite 长期记忆库（统一存储）
src/
├── call_llm/                         # 统一 LLM 调用入口（读 .env）
├── memory/
│   ├── session_memory/               # import_from_json.py
│   ├── effective_memory/             # extract_from_session.py + qa_from_effective_memory.py
│   ├── longterm/                     # effective_to_longterm.py (批量模式)
│   └── qa/                           # ⭐ three_layer_qa.py (三层 QA 系统)
├── eval/
│   ├── run_eval.py                   # ⭐ Eval 评测脚本（三层记忆版本）
│   └── self_evolve/                  # PPO / prompt optimizer
├── storage/sqlite/                   # SQLite 操作层
└── embedding/                        # Embedding 相关（预留）
frontend/                             # React 前端
├── src/components/
│   ├── SessionPanel.tsx              # Session 记忆面板
│   ├── EffectivePanel.tsx            # Effective 记忆面板
│   ├── LongtermPanel.tsx             # Longterm 记忆面板
│   ├── ConnectionLines.tsx           # 贝塞尔曲线连线
│   └── EvalPage.tsx                 # Eval 评测页面
server.py                             # FastAPI 后端 (:8000)
config/skill/                         # Prompt 模板
├── extract.md                        # Effective 抽取 prompt
├── longterm_ops.md                   # ⭐ Long-term 决策 prompt (organizeParam 风格)
└── qa.md                             # QA 回答 prompt
```

---

## ⚡ 常用命令速查

```bash
# 🔍 查看所有 group
ls data/groups/

# 🔄 全流程（Steps 1-3）- 单个用户
python src/memory/session_memory/import_from_json.py --group group_1_gina
python src/memory/effective_memory/extract_from_session.py --group group_1_gina
python src/memory/longterm/effective_to_longterm.py --group group_1_gina --batch-size 10

# 🔄 批量处理所有用户
for g in group_0_caroline group_1_gina group_0_melanie group_1_jon; do
  python src/memory/session_memory/import_from_json.py --group $g
  python src/memory/effective_memory/extract_from_session.py --group $g
  python src/memory/longterm/effective_to_longterm.py --group $g --batch-size 10
done

# 📊 运行 Eval 评测（三层记忆）
python src/eval/run_eval.py --group group_1_gina

# 🧪 测试三层 QA 系统
python -c "
from src.memory.qa.three_layer_qa import ThreeLayerQA
qa = ThreeLayerQA(group='group_1_gina')
result = qa.answer('When did Gina start her store?')
print('Answer:', result.answer)
print('Confidence:', result.confidence)
"

# 🚀 启动 UI
python server.py &
cd frontend && npm run dev

# 🔄 Self-Evolving 优化
python src/eval/self_evolve/ppo_fixed_memories.py
```

---

## 📦 依赖安装

```bash
# Python 依赖
pip install python-dotenv nltk fastapi uvicorn pydantic requests

# 前端依赖
cd frontend && npm install
```

---

## ⚠️ 注意事项

1. **API Key 安全**: `.env` 文件包含敏感信息，请勿提交到 Git
2. **LLM 调用成本**: Steps 2/3/5 都会调用 LLM，注意控制调用次数
3. **断点续跑**: 所有脚本都支持中断后继续执行
4. **数据备份**: 重要操作前建议备份 `data/` 目录
5. **并发限制**: 建议串行执行同一 group 的多个步骤
6. **时间保留**: ⏰ 所有操作都会保留原始时间戳，这对 QA 至关重要

---

## 🔧 故障排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| `No module named 'nltk'` | `pip install nltk` |
| `.env 未加载` | 检查是否在项目根目录运行 |
| Effective 为空 | 检查 `data/input/*.json` 是否存在 |
| LLM 调用失败 | 检查 `.env` 中的 `FRIDAY_APP_ID` 和 `LLM_BASE_URL` |
| 前端空白 | 检查后端是否启动 (`python server.py`) |
| Long-term 数量太少 | 尝试 `--batch-size 5` 或检查 `longterm_ops.md` 提示词 |
| 时间信息缺失 | 确认使用新版 `effective_to_longterm.py`（批量模式） |

---

## 📈 性能指标参考

### Gina 用户组（group_1_gina）

| 层级 | 数量 | 说明 |
|------|------|------|
| **Session Memory** | 189 条 | 原始对话记录 |
| **Effective Memory** | 20 条 | 结构化摘要 |
| **Long-term Memory** | **54 条** | 高度提炼（新版 vs 旧版 3 条） |

### Eval 评测结果（预期）

| 指标 | 旧版（仅 Effective） | 新版（三层记忆） | 提升幅度 |
|------|---------------------|-----------------|----------|
| **Temporal 得分** | ~0.45 | **~0.75** (预估) | **+67%** ⬆️ |
| **Overall Judge Score** | ~0.55 | **~0.70** (预估) | **+27%** ⬆️ |
| **F1 Score** | ~0.35 | **~0.50** (预估) | **+43%** ⬆️ |

> 注：实际数值以运行 `run_eval.py` 后的输出为准。

---

## 🔄 版本更新日志

### v2.0 (2026-06-11) — 三层记忆架构升级

**新功能**:
- ✨ **三层 QA 系统** (`src/memory/qa/three_layer_qa.py`)
  - Long-term → Effective → Session 降级策略
  - 保留精确时间戳用于时间推理
  - 置信度评估

- ✨ **批量 Long-term 流转** (`src/memory/longterm/effective_to_longterm.py`)
  - organizeParam 四操作模式（ADD/UPDATE/DELETE/NONE）
  - 批量处理（默认 10 条/批）
  - 智能匹配算法（精确→关键词→顺序）

- ✨ **时间保留机制**
  - 所有层级保留 `time_range.start/end`
  - Session Memory 保留精确到分钟的原始时间戳
  - 导出 JSONL 包含完整时间信息

**优化**:
- 🚀 Long-term Memory 数量提升：3 条 → **54 条**（18x）
- 🎯 提示词优化：更倾向于 ADD 而非保守 NOOP
- 🔗 后端 API 支持按 user_id 动态加载数据
- 👤 前端支持 User/Group 切换

**修复**:
- 🐛 修复 Effective Memory 不随 User 切换刷新的问题
- 🐛 修复 time_range 信息丢失的问题
- 🐛 修复后端接口硬编码路径的问题

---

## 📞 技术支持

如有问题，请检查：
1. 日志文件：`/tmp/server.log`（后端日志）
2. 数据目录：`data/groups/{group}/` 各层文件是否完整
3. 环境配置：`.env` 文件是否正确加载

