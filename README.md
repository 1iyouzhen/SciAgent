# SciAgent - 可验证的科学文献问答系统

> **解决单一大模型的核心痛点**
>
> 当前主流的AI问答系统普遍依赖单一大语言模型，存在诸多难以回避的问题：**幻觉严重**——模型经常编造看似合理但实际错误的信息，尤其在专业科研领域更为突出；**无法溯源**——答案缺乏引用来源，用户无法验证信息的准确性和可靠性；**知识陈旧**——模型训练数据存在截止日期，无法获取最新研究成果；**推理黑箱**——用户只能看到最终答案，无法了解模型的推理过程和依据；**单点失效**——完全依赖一个模型，一旦该模型在某领域表现不佳则无法弥补。
>
> SciAgent 正是为解决这些痛点而生。它采用**多智能体协作架构**，通过 Planner（任务分解）→ Retriever（证据检索）→ Caption（图像理解）→ MultiLLM Reasoner（多模型协同推理）→ Reviewer（自我校验）的完整流水线，实现**可验证、可溯源、可解释**的科学文献问答。系统强制要求每个论点都标注引用来源，支持多个LLM模型集成投票以降低单模型偏差，并通过深度思考模式展示完整的推理链条，让用户真正理解答案是如何得出的。

---

## 核心特性

### 🔬 可验证的RAG架构
- **强制引用标注**：每个关键论点必须标注 `[来源X]`，可追溯到具体文档和页码
- **置信度自校准**：多维度评估答案可靠性（证据强度、来源可靠性、推理有效性、覆盖完整性）
- **自我校验机制**：Reviewer Agent 自动检测幻觉和逻辑错误，支持迭代优化

### 🤖 多智能体协作
- **Planner Agent**：智能任务分解，将复杂问题拆解为可独立检索的子任务
- **Retriever Agent**：Qwen3-Embedding + FAISS 向量检索，支持混合检索（向量+关键词）
- **Caption Agent**：Qwen-VL 图像理解，支持图表分析和OCR文字提取
- **MultiLLM Reasoner**：多模型协同推理，支持文本推理、数学推理、视觉理解
- **Reviewer Agent**：Rule-based + LLM Judge 双重校验，支持 Reviewer → Retriever 迭代循环
- **Reflector Agent**：对话质量分析，生成改进建议和后续问题推荐

### 🧠 多LLM集成推理（动态权重）
- **动态权重分配**：基于各模型自我校准的置信度动态计算权重，置信度越高的模型获得越大的话语权（使用置信度平方放大差异）
- **集成投票**：支持多个模型（Qwen、DeepSeek等）协同推理，自动选择动态权重最高的结果作为主答案
- **专业分工**：文本推理、数学推理、视觉理解可配置不同专长模型
- **自动回退**：主模型失败时自动切换备选模型，确保服务稳定性

### 💭 深度思考模式（类DeepSeek-R1）
- **完整推理链展示**：分步骤显示问题理解→证据分析→跨文档关联→推理链构建→置信度评估
- **流式输出**：实时展示思考过程，用户可观察AI的推理轨迹
- **多文档联合推理追踪**：清晰标注每条证据来自哪个文档的哪一页

### 📚 多模态文档处理
- **多格式支持**：PDF、TXT、DOCX、MD、TEX
- **MinerU深度解析**：支持文本、表格、图像提取和布局分析
- **URL直接导入**：支持从arXiv、OpenReview等学术网站直接下载PDF
- **增量索引**：新文档自动追加到现有索引，无需重建

### 🎨 AI图像生成
- **基于论文内容生成可视化**：根据文档内容自动生成科学示意图
- **硅基流动API集成**：支持FLUX.1-schnell等图像生成模型
- **智能提示词生成**：LLM自动将用户需求转换为英文图像生成提示词

### 📝 综述报告生成
- **自动生成结构化报告**：标题、摘要、分章节内容、结论
- **每句话标注引用**：支持点击引用跳转到原文位置
- **智能问题推荐**：根据上传文档自动生成3-5个有价值的研究问题

### 💾 记忆库系统
- **对话历史持久化**：保存所有对话记录，支持跨会话检索
- **相似问题检索**：自动查找历史中的相似问题和答案
- **自我反思**：分析对话质量，识别知识空白，生成改进建议

---

## 功能详情

### 1. 用户认证系统
- 用户注册/登录/登出
- 会话管理（7天有效期）
- 安全的密码哈希存储

### 2. 多文档上传与分析
- 支持最多50个文档同时上传
- 支持格式：PDF, TXT, DOCX, MD, TEX
- 支持从学术网站URL直接下载PDF（arXiv, OpenReview等）
- 多LLM协作分析多个资源

### 3. 深度思考模式 (类似DeepSeek-R1)
- 展示完整推理过程
- 分步骤显示思考链
- 支持流式输出

### 4. 智能推荐问题
- 根据上传文档自动生成3-5个推荐问题
- 点击即可快速提问

### 5. 综述报告生成
- 根据用户需求自动生成综述报告
- 每句话标注来源引用
- 支持引用定位到具体文档和页码

### 6. 记忆库系统
- 保存对话历史
- 相似问题检索
- 跨对话历史记忆
- 自我反思功能

### 7. 多文档推理追踪
- 展示文档来源
- 显示推理链
- 标注引用出处

### 8. 置信度自校准
- 每次回答输出置信率
- 多维度评估
- 可视化展示

### 9. 回答重新生成
- 支持重新生成
- 保留历史版本
- 可查看对比

### 10. 对话历史管理
- 查看所有历史对话
- 删除对话
- 跨对话记忆

### 11. AI图像生成
- 基于论文内容生成科学可视化图片
- 支持多种图片风格（流程图、示意图、信息图等）
- 自动生成英文提示词，调用FLUX等模型生成图片

### 12. 文档翻译
- 支持整篇文档或指定页码范围翻译
- 保留LaTeX公式格式
- 多语言支持

---

## 快速开始

### 1. 环境准备

**克隆项目**
```bash
git clone <repository-url>
cd sci_agent
```

**安装依赖**
```bash
pip install -r sci_agent/requirements.txt
```

**可选依赖**（增强功能）
```bash
# FAISS向量检索（推荐，提升检索性能）
pip install faiss-cpu

# MinerU PDF深度解析（支持图像提取）
pip install magic-pdf

# 本地模型支持
pip install transformers torch
```

### 2. 配置API密钥

编辑 `sci_agent/.env` 文件：

```bash
# 硅基流动 API（推荐，支持多模型）
SILICONFLOW_API_KEY=your-api-key

# 图像生成配置（可选）
IMAGE_API_PROVIDER=siliconflow
IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
IMAGE_SIZE=1024x1024

# HuggingFace镜像（国内用户推荐）
HF_ENDPOINT=https://hf-mirror.com
```

**支持的API提供商**：

| 提供商 | 环境变量 | 推荐模型                                     |
|--------|----------|------------------------------------------|
| 硅基流动 | `SILICONFLOW_API_KEY` | Qwen/Qwen2.5-7B-Instruct, DeepSeek-V2.5,Qwen/Qwen2.5-14B-Instruct |
| 阿里云DashScope | `DASHSCOPE_API_KEY` | qwen-plus, qwen-max                      |
| OpenAI | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini                      |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat                            |
| 智谱 | `ZHIPU_API_KEY` | glm-4-flash, glm-4v                      |
| Moonshot | `MOONSHOT_API_KEY` | moonshot-v1-8k                           |

### 3. 启动服务

**方式一：使用启动脚本**
```bash
python -m sci_agent.run_server
```

**方式二：使用uvicorn**
```bash
uvicorn sci_agent.api.app:app --host 0.0.0.0 --port 8080 --reload
```

### 4. 访问系统
- 前端界面: http://localhost:8080/
- API文档: http://localhost:8080/docs
- ReDoc文档: http://localhost:8080/redoc

---

## 使用指南

### 基本使用流程

1. **注册/登录账号** - 首次使用需注册，支持会话持久化
2. **创建新对话** - 每个对话独立管理文档和历史
3. **上传文档** - 支持拖拽上传或粘贴URL
4. **等待索引构建** - 系统自动解析文档并构建向量索引
5. **查看推荐问题** - 系统根据文档内容自动生成推荐问题
6. **提问** - 输入问题或点击推荐问题
7. **查看推理过程** - 深度思考模式展示完整推理链
8. **验证引用** - 点击引用标记跳转到原文位置
9. **生成报告**（可选）- 根据需求生成综述报告
10. **查看反思**（可选）- 获取对话质量分析和改进建议

### 命令行使用

**构建文档索引**
```bash
python -m sci_agent.main --build-index --pdf-dir ./your-pdfs
```

**单次问答**
```bash
python -m sci_agent.main --question "请总结这篇论文的主要贡献"
```

**交互模式**
```bash
python -m sci_agent.main --interactive
```

### 配置多LLM集成推理

编辑 `sci_agent/config.yaml`：

```yaml
agents:
  reasoning:
    enable_ensemble: true
    
    # 文本推理模型
    text_reasoner:
      provider: siliconflow
      model: Qwen/Qwen2.5-7B-Instruct
    
    # 数学推理模型
    math_reasoner:
      provider: siliconflow
      model: deepseek-ai/DeepSeek-V2.5
    
    # 集成模型列表（配置中的weight仅作为初始参考，实际运行时会根据各模型置信度动态调整）
    ensemble_models:
      - provider: siliconflow
        model: Qwen/Qwen2.5-7B-Instruct
        weight: 0.4  # 初始权重，运行时会被动态权重覆盖
      - provider: siliconflow
        model: deepseek-ai/DeepSeek-V2.5
        weight: 0.35
      - provider: siliconflow
        model: Qwen/Qwen2.5-14B-Instruct
        weight: 0.25
    
    strategy:
      mode: ensemble  # single | ensemble | cascade
      ensemble_method: weighted  # vote | weighted(动态加权) | best
```

**动态权重机制说明**：
- 系统会调用所有配置的集成模型进行推理
- 每个模型返回答案时会附带自我校准的置信度
- 动态权重 = 置信度² / Σ(所有模型置信度²)
- 置信度高的模型获得更大权重，最终选择动态权重最高的模型答案作为主答案
- 加权置信度 = Σ(各模型置信度 × 动态权重)

---

## API接口

### 认证
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 用户注册 |
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/auth/logout` | 用户登出 |
| GET | `/api/auth/me` | 获取当前用户信息 |

### 对话管理
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/conversations` | 创建新对话 |
| GET | `/api/conversations` | 获取对话列表 |
| GET | `/api/conversations/{id}` | 获取对话详情 |
| DELETE | `/api/conversations/{id}` | 删除对话 |

### 文档上传
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文件（支持多文件，最多50个） |
| POST | `/api/documents/upload-urls` | 从URL上传（支持arXiv等） |
| GET | `/api/documents/session/{conversation_id}` | 获取文档会话状态 |
| DELETE | `/api/documents/{session_id}/{doc_id}` | 删除指定文档 |

### 问答
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ask` | 提问（普通模式） |
| POST | `/api/ask/stream` | 提问（流式模式，展示思考过程） |
| POST | `/api/regenerate` | 重新生成回答 |
| GET | `/api/conversations/{id}/turns/{turn_id}/history` | 获取回答历史版本 |

### 报告与推荐
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/report/generate` | 生成综述报告 |
| GET | `/api/report/suggested-questions/{conversation_id}` | 获取推荐问题 |
| POST | `/api/report/mindmap` | 生成思维导图 |

### 图像生成
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/visualize/generate` | 基于文档生成可视化图片 |

### 翻译
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/translate/text` | 翻译文本 |
| POST | `/api/translate/document` | 翻译文档（支持指定页码范围） |

### 反思与记忆
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/conversations/{id}/reflection` | 获取对话反思 |
| POST | `/api/conversations/{id}/reflection/refresh` | 刷新对话反思 |
| GET | `/api/history/context` | 获取用户历史上下文 |
| GET | `/api/history/documents` | 获取用户文档缓存 |

---

## 项目结构

```
sci_agent/
├── agents/                      # 智能体模块
│   ├── base.py                  # Agent基类和上下文定义
│   ├── planner.py               # 任务分解Agent
│   ├── retriever.py             # 多模态混合检索Agent
│   ├── caption.py               # 图像理解Agent（Qwen-VL）
│   ├── multi_llm_reasoner.py    # 多LLM协同推理Agent
│   ├── deep_thinker.py          # 深度思考Agent（类DeepSeek-R1）
│   ├── reviewer.py              # 自我校验Agent
│   ├── reflector.py             # 对话反思Agent
│   ├── report_generator.py      # 综述报告生成Agent
│   └── image_generator.py       # AI图像生成Agent
│
├── api/                         # API模块
│   └── app.py                   # FastAPI应用（RESTful API）
│
├── models/                      # 数据模型
│   ├── user.py                  # 用户模型和会话管理
│   ├── memory.py                # 记忆库模型（对话、反思、推理追踪）
│   └── document.py              # 文档模型和会话管理
│
├── tools/                       # 工具模块
│   ├── llm_client.py            # 统一LLM客户端（支持8+平台）
│   ├── vector_db.py             # 向量数据库（Qwen3-Embedding + FAISS）
│   └── pdf_parser.py            # PDF解析器（支持MinerU）
│
├── web/                         # 前端界面
│   └── index.html               # 单页应用
│
├── data/                        # 数据目录
│   ├── pdfs/                    # PDF文档存放
│   ├── processed/               # 解析缓存
│   ├── documents/               # 用户文档会话
│   ├── memory/                  # 对话记忆
│   ├── users/                   # 用户数据
│   └── generated_images/        # 生成的图片
│
├── main.py                      # 主流水线入口
├── run_server.py                # 服务启动脚本
├── config.yaml                  # 配置文件
├── .env                         # 环境变量（API密钥）
└── requirements.txt             # 依赖列表
```

---

## 技术架构

### 工作流程
```
用户问题
    ↓
┌─────────────────────────────────────────────────────────┐
│  Planner Agent                                          │
│  - 分析问题类型（研究类/比较类/原理类）                    │
│  - 分解为2-8个可独立检索的子任务                          │
│  - 生成检索关键词                                        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Retriever Agent                                        │
│  - Qwen3-Embedding 向量编码                              │
│  - FAISS 高效检索                                        │
│  - 混合检索（向量 + 关键词）                              │
│  - 结果重排序                                            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Caption Agent                                          │
│  - Qwen-VL 图像理解                                      │
│  - 图表分析                                              │
│  - OCR文字提取                                           │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  MultiLLM Reasoner Agent                                │
│  - 多模型协同推理（Qwen + DeepSeek + ...）               │
│  - 动态权重分配（基于各模型置信度自动计算权重）            │
│  - 数学公式推理（支持LaTeX格式）                          │
│  - 深度思考链展示                                        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  Reviewer Agent                                         │
│  - Rule-based 规则检查                                   │
│  - LLM Judge 智能评估                                    │
│  - 置信度计算                                            │
│  - 迭代优化（Reviewer → Retriever 循环）                 │
└─────────────────────────────────────────────────────────┘
    ↓
最终答案（带引用、置信度、推理追踪）
```

### 支持的LLM平台
- **硅基流动** (SiliconFlow) - 推荐，支持多模型聚合
- **阿里云DashScope** - Qwen系列
- **OpenAI** - GPT系列
- **DeepSeek** - DeepSeek系列
- **智谱** - GLM系列
- **Moonshot** - Kimi
- **百川** - Baichuan系列
- **Ollama** - 本地部署

---

## 常见问题

**Q: 如何提升回答质量？**
- 上传更多相关文档
- 使用更具体的问题描述
- 启用多LLM集成推理模式

**Q: 图像理解功能不工作？**
- 确保安装了MinerU：`pip install magic-pdf`
- 重新构建索引：`python -m sci_agent.main --build-index`

**Q: API调用失败？**
- 检查 `.env` 文件中的API密钥是否正确
- 确认网络可以访问对应的API服务

**Q: 如何使用本地模型？**
- 安装Ollama并下载模型
- 配置 `api_provider: ollama`
- 设置对应的模型名称

---

## License

MIT License
