---
name: translate-langchain-doc
description: 翻译 LangChain 官方技术文档（Deep Agents 系列），产出高质量中英双语对照 Markdown 文件
author: AtomCode
version: 1.0
inputs:
  - name: url
    description: LangChain 文档页面的完整 URL（如 https://docs.langchain.com/oss/python/deepagents/backends）
    required: true
  - name: output_path
    description: 翻译文件输出路径（相对 llm/langchain/deepagents/，默认基于 URL 最后一段自动推断）
    required: false
---

# 翻译 LangChain 官方文档（Translate LangChain Docs）

本 Skill 定义了翻译 LangChain 官方技术文档的完整工作流、规则和输出格式。当你需要翻译新的 LangChain 文档页面时，请严格遵循以下流程。

---

## 工作流（Workflow）

### 第 1 步：抓取页面内容

使用 `web_fetch` 工具获取目标 URL 的渲染文本内容。如果内容被截断（`max_chars` 限制），分段获取页面不同部分的内容。

```python
# 初始获取（限制 30000 字符）
web_fetch(url="<目标URL>", max_chars=30000)

# 如果截断，逐段获取（先处理需要的部分）
web_fetch(url="<目标URL>", max_chars=30000, offset=30000)
```

### 第 2 步：解析页面结构

从抓取的文本中提取以下结构要素：

| 要素 | 提取方式 |
|------|---------|
| **页面标题** | 通常为首个 `# 标题` 或页面顶部粗体文本 |
| **章节标题** | `##` / `###` / `####` 层级标题 |
| **引文/概要** | 标题下方的简介段落（通常以 `>` 开头） |
| **表格** | 识别所有表格的行列结构，保留表头 |
| **代码块** | 识别所有 `python` / `bash` / 其他语言的代码块 |
| **列表** | 有序和无序列表 |
| **链接** | 所有 `[显示文本](URL)` 形式的链接 |
| **图片** | 所有 `![alt](URL)` 形式的图片引用 |

### 第 3 步：翻译内容

逐章节翻译，按以下顺序处理：

1. **标题** → 翻译，保留原始层级标记
2. **引文/概要** → 翻译，保留 `>` 引用格式
3. **正文段落** → 翻译，保留**粗体**、*斜体*、`行内代码` 等格式标记
4. **表格** → 翻译表头和表体内容，保留列对齐和行列结构
5. **列表** → 翻译列表项，保留嵌套层级
6. **代码块** → 不翻译代码内容（保留原始代码），仅翻译代码前的说明文字
7. **链接** → 翻译显示文本，保留原始 URL

### 第 4 步：保存输出文件

将翻译结果写入 `llm/langchain/deepagents/<文件名>.md`。

文件命名规则：
- URL 最后一段作为文件名（如 `backends` → `backends.md`）
- 如果同一页面已存在旧版翻译，添加序号前缀（如 `1.models.md`）

### 第 5 步：输出结构

输出文件包含以下结构：

```markdown
# 中文标题（英文原文标题）

> 概要/引文翻译

---

正文内容（翻译后）

---

> **原文**：<原始 URL>
>
> **许可**：本文档基于 LangChain 官方文档翻译，仅供学习参考。
```

---

## 翻译规则（Translation Rules）

### 通用原则

| 规则 | 说明 |
|------|------|
| **准确性优先** | 技术术语必须准确，宁可在括号中保留英文也不要意译错误 |
| **首次出现标注** | 重要技术术语首次出现时，在中文后加括号标注英文原文，如"后端（Backend）" |
| **保持一致性** | 同一术语在全文中使用统一译法 |
| **可读性** | 中文句子不要太长，长英文句拆分成短句 |
| **中文标点** | 正文使用中文标点（，。、：；「」——等），代码块内保持英文标点 |
| **代码不翻译** | Python 代码、标识符、函数名、变量名、字符串（除非是文档中的界面文字）保持原样 |
| **注释选择性翻译** | 代码中的英文注释可以翻译，但保持 `# ` 格式 |

### 术语翻译表

以下是在 Deep Agents 文档中使用的标准术语翻译：

| 英文 | 中文 |
|------|------|
| Agent | Agent（不译，保持英文） |
| Deep Agent | Deep Agent（不译） |
| Backend | 后端 |
| Model | 模型 |
| Provider | 提供商 |
| Tool | 工具 |
| Tool calling | 工具调用 |
| System prompt | 系统提示 |
| Context | 上下文 |
| Context engineering | 上下文工程 |
| Context compression | 上下文压缩 |
| Offloading | 卸载 |
| Summarization | 摘要 |
| Memory | 记忆 |
| Skill | 技能 |
| Subagent | 子 Agent |
| Runtime context | 运行时上下文 |
| Middleware | 中间件 |
| State | 状态/线程作用域 |
| Checkpoint | 检查点 |
| Thread | 线程 |
| Namespace | 命名空间 |
| Sandbox | 沙箱 |
| Virtual file system | 虚拟文件系统 |
| Virtual mode | 虚拟模式 |
| Root directory | 根目录 |
| Permissions | 权限 |
| Policy hooks | 策略钩子 |
| Composite | 复合/路由 |
| Migrate / Migration | 迁移 |
| Deprecated | 已弃用 |
| Protocol | 协议 |
| Harness | 操控 |
| Harness profile | 操控配置文件 |
| Prompt chaining | 提示链 |
| Parallelization | 并行化 |
| Orchestrator-workers | 编排器-工作者 |
| Evaluator-optimizer | 评估器-优化器 |
| Long-term memory | 长期记忆 |
| Context isolation | 上下文隔离 |
| Input context | 输入上下文 |
| LangGraph | LangGraph（不译） |
| LangSmith | LangSmith（不译） |
| Context Hub | Context Hub（不译） |
| Token | Token（不译） |
| SSRF | SSRF（不译） |

### 格式保留规则

| 元素 | 处理方式 |
|------|---------|
| `#` 标题标记 | 保留层级和数量 |
| `**粗体**` | 保留，中文加粗 |
| `*斜体*` | 保留 |
| `` `行内代码` `` | 保留，内容不翻译 |
| ` ```python ` 代码块 | 保留标记语言，不翻译代码内容 |
| `|` 表格分隔符 | 保留，翻译表头和表体 |
| `- ` 无序列表 | 保留，翻译列表项 |
| `1. ` 有序列表 | 保留序号，翻译列表项 |
| `[链接](URL)` | 翻译显示文本，保留 URL |
| `![图片](URL)` | 翻译 alt 文本，保留 URL |
| `---` 分隔线 | 保留 |
| `> ` 引用 | 保留，翻译引用内容 |
| 空行 | 保留，用于结构化分段 |

---

## 输出文件格式说明

### 文件头

- **标题**：`# 中文标题（英文原文标题）`
- **概要**：`> 翻译后的概要说明`
- **分隔线**：`---`

### 段落

- 翻译后的段落直接替换原文段落
- 技术术语首次出现时标注英文
- 保持原段落顺序

### 代码块

```
```python
# 这里写翻译后的注释（可选）
original_code_here  # 不翻译
````

### 表格

| 中文表头 A | 中文表头 B |
|-----------|-----------|
| 翻译后内容 | 翻译后内容 |

### 文件尾部

所有翻译文件以以下内容结尾：

```
---

> **原文**：<原文 URL>
>
> **许可**：本文档基于 LangChain 官方文档翻译，仅供学习参考。
```

---

## 已完成翻译参考

以下页面已完成翻译，可作为风格参考：

| 原文 URL | 翻译文件 |
|----------|---------|
| `https://docs.langchain.com/oss/python/deepagents/models` | `llm/langchain/deepagents/1.models.md` |
| `https://docs.langchain.com/oss/python/deepagents/context-engineering` | `llm/langchain/deepagents/2.context-engineering.md` |
| `https://docs.langchain.com/oss/python/deepagents/backends` | `llm/langchain/deepagents/backends.md` |

---

## 质量检查清单

翻译完成后检查以下项：

- [ ] 所有章节标题已翻译并保留层级
- [ ] 所有代码块内容保持原样
- [ ] 所有表格结构完整（列数一致，对齐正确）
- [ ] 所有链接显示文本已翻译，URL 保留
- [ ] 技术术语首次出现标注英文
- [ ] 文件尾部包含原文 URL 和许可声明
- [ ] 文件命名正确，存放在 `llm/langchain/deepagents/` 目录下
- [ ] 文件编码为 UTF-8（无 BOM）

---

## 注意事项

1. **图片处理**：如果页面包含内容图片（非导航/品牌 Logo），需要下载图片并保存到 `llm/langchain/deepagents/images/` 目录，在翻译文件中使用相对路径引用。
2. **内容截断**：如果源文档过长导致 `web_fetch` 截断，需分多次获取，确保完整捕获所有章节。
3. **版本标注**：如果文档有明显版本信息（如 API 版本号），在标题下方标注。目前 Deep Agents 文档无显式版本号。
4. **原文链接保留**：文档中的内部链接（如 `/oss/python/deepagents/skills`）保留原样，不转换为完整 URL，因为它们是 LangChain 站点的相对路径。
