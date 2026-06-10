# 🚀 Quick Start — LLM Memory System 快速上手指南

## 环境准备

```bash
cd /Users/kwjcyr/data/llm_memory

# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，修改 FRIDAY_APP_ID 和 LLM_MODEL 为你自己的值
```

**`.env` 关键配置项**：
```bash
FRIDAY_APP_ID=你的AppID
LLM_MODEL=模型名称（如 gpt-4.1）
LLM_TEMPERATURE=0.1
LLM_BASE_URL=https://aigc.sankuai.com/v1/openai/native/chat/completions
```

---

## 完整流程（7 步）

### Step 0: 准备输入数据

```bash
# 从 API 获取 Session Memory，导出为 JSON 文件
# 将 response 拷贝到 data/input/ 目录，如 group_0_caroline.json

ls data/input/
# 应该看到: group_0_caroline.json, group_1_gina.json, ...
```

---

### Step 1: 获取 Session Memory

```bash
python src/memory/session_memory/import_from_json.py --group group_0_caroline
```

**输出**: `data/groups/group_0_caroline/session/memories.jsonl` (约 214 条)

**说明**: 将原始 API JSON 转换为标准 JSONL 格式，按时间排序。

---

### Step 2: Session Memory → Effective Memory（调 LLM）

```bash
python src/memory/effective_memory/extract_from_session.py --group group_0_caroline
```

**输出**: `data/groups/group_0_caroline/effective/effective_memories.jsonl` (约 19 条)

**说明**: 按 4 小时窗口分组对话，每窗口调一次 LLM 抽取结构化记忆（topic, summary, facts, tags）。

---

### Step 3: Effective Memory → Long-term Memory（LLM 决策）

```bash
python src/memory/longterm/effective_to_longterm.py --user caroline
```

**输出**:
- `data/longterm_memories.db` (SQLite)
- `data/longterm_memories.jsonl` (JSON 备份)

**说明**: 对每条 effective memory，检索相似已有长期记忆，调 LLM 决定 INSERT / UPDATE / DELETE / NOOP。

---

### Step 4: UI 查看 Memory 之间的关联

```bash
# 终端 1：启动后端
python server.py &   # 后端 API :8000

# 终端 2：启动前端
cd frontend && npm run dev  # 前端 :5173
```

打开 http://localhost:5173 → **🗂 Memory Explorer** Tab 查看三层记忆及连线。

**功能**:
- Session 面板：原始对话时间线
- Effective 面板：结构化记忆卡片
- Longterm 面板：长期固化记忆
- 贝塞尔曲线连线展示记忆间的来源关系

---

### Step 5: Eval — LocoMo QA 评测

```bash
python src/eval/run_eval.py --group group_0_caroline
```

**输出**: `data/groups/group_0_caroline/eval/`
```
├── locomo_responses.json   # 原始回答（含检索到的有效记忆）
├── locomo_judged.json     # 评分结果（LLM Judge + F1/BLEU-1）
└── locomo_grades.json     # 汇总统计（总分 + 分类得分）
```

**评测流程**:
1. **Response 阶段**: 从 Effective Memory 检索相关记忆 + 调 LLM 回答
2. **Judged 阶段**: LLM-as-a-Judge (CORRECT/WRONG) + NLP 指标 (F1, BLEU-1)
3. **Grades 阶段**: 总体统计 + 5 类问题分类统计

**指标说明**:
- **LLM Judge Score**: LLM 判断回答是否正确的准确率（0~1）
- **F1 Score**: Token 级别的精确匹配分数（0~1）
- **BLEU-1 Score**: Unigram 精确度（0~1）
- **分类统计**: Multi hop / Temporal reasoning / Open domain / Single hop / Adversarial

**高级选项**:
```bash
# 自定义检索数量
python src/eval/run_eval.py --group group_0_caroline --top_k 10

# 多次 LLM Judge 评估（更稳定但更慢）
python src/eval/run_eval.py --group group_0_caroline --judge-attempts 3

# 静默模式
python src/eval/run_eval.py --group group_0_caroline --quiet
```

**断点续跑**: 脚本会自动保存进度，可随时中断后重新执行。

---

### Step 6: Self-Evolving（PPO 优化）

```bash
python src/eval/self_evolve/ppo_fixed_memories.py
```

基于固定 19 条 effective memory，用 PPO 优化 top_k 和 fields 动作空间。

**优化目标**: 提升 Eval 的 LLM Judge Score 和 F1 Score。

---

## 目录结构

```
data/
├── .env                              # LLM 配置（app_id, model, temperature）
├── input/                            # 原始 API JSON 输入
│   ├── group_0_caroline.json
│   └── group_1_gina.json
├── groups/                           # 按用户组组织的处理结果
│   └── group_0_caroline/
│       ├── raw/                      # 原始数据备份
│       ├── session/                  # Session Memory (JSONL)
│       │   └── memories.jsonl
│       ├── effective/                # Effective Memory (JSONL)
│       │   └── effective_memories.jsonl
│       ├── longterm/                 # Long-term Memory (预留)
│       └── eval/                     # 评测结果
│           ├── locomo_responses.json
│           ├── locomo_judged.json
│           └── locomo_grades.json
├── longterm_memories.db              # SQLite 长期记忆库
└── longterm_memories.jsonl           # JSONL 备份
src/
├── call_llm/                         # 统一 LLM 调用入口（读 .env）
├── memory/
│   ├── session_memory/               # import_from_json.py
│   ├── effective_memory/             # extract_from_session.py
│   └── longterm/                     # effective_to_longterm.py
├── eval/
│   ├── run_eval.py                   # ⭐ Eval 评测脚本（对齐 memory_evaluation）
│   └── self_evolve/                  # PPO / prompt optimizer / harness
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
├── longterm_ops.md                   # Long-term 决策 prompt
└── qa.md                             # QA 回答 prompt
```

---

## 常用命令速查

```bash
# 🔍 查看所有 group
ls data/groups/

# 🔄 切换 group 运行全流程（Steps 1-3）
for g in group_0_caroline group_1_gina; do
  python src/memory/session_memory/import_from_json.py --group $g
  python src/memory/effective_memory/extract_from_session.py --group $g
  python src/memory/longterm/effective_to_longterm.py --user ${g#*_}
done

# 📊 运行 Eval 评测
python src/eval/run_eval.py --group group_0_caroline

# 🚀 启动 UI
python server.py &
cd frontend && npm run dev

# 🔄 Self-Evolving 优化
python src/eval/self_evolve/ppo_fixed_memories.py
```

---

## 依赖安装

```bash
pip install python-dotenv nltk fastapi uvicorn pydantic
npm install -g vite  # 前端构建工具
cd frontend && npm install  # 前端依赖
```

---

## 注意事项

1. **API Key 安全**: `.env` 文件包含敏感信息，请勿提交到 Git
2. **LLM 调用成本**: Steps 2/3/5 都会调用 LLM，注意控制调用次数
3. **断点续跑**: 所有脚本都支持中断后继续执行
4. **数据备份**: 重要操作前建议备份 `data/` 目录
5. **并发限制**: 建议串行执行同一 group 的多个步骤

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| `No module named 'nltk'` | `pip install nltk` |
| `.env 未加载` | 检查是否在项目根目录运行 |
| Effective 为空 | 检查 `data/input/*.json` 是否存在 |
| LLM 调用失败 | 检查 `.env` 中的 `FRIDAY_APP_ID` 和 `LLM_BASE_URL` |
| 前端空白 | 检查后端是否启动 (`python server.py`) |

