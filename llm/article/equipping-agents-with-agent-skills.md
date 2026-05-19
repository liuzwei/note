# 用 Agent Skills 装备 Agent，为现实世界做准备

> 原文：[Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
> 作者：Barry Zhang, Keith Lazuka, Mahesh Murag
> 发布时间：2025 年 10 月 16 日

**核心观点**：Claude 很强大，但真正的工作需要**程序性知识和组织上下文**。本文介绍 Agent Skills——一种使用文件和文件夹构建专用 Agent 的新方法。

> 更新：我们已于 2025 年 12 月 18 日将 Agent Skills 发布为[跨平台可移植的开放标准](https://github.com/anthropics/claude-agent-skills)。

---

随着模型能力的提升，我们现在可以构建与完整计算环境交互的通用 Agent。例如，Claude Code 可以通过本地代码执行和文件系统跨领域完成复杂任务。但随着这些 Agent 变得更加强大，我们需要更可组合、可扩展和可移植的方式来为它们配备特定领域的专业知识。

这促使我们创建了 **Agent Skills**：由指令、脚本和资源组成的组织化文件夹，Agent 可以动态发现和加载这些内容，以在特定任务上表现更好。Skills 通过将你的专业知识打包成 Claude 的可组合资源，扩展 Claude 的能力，将通用 Agent 转化为适合你需求的专用 Agent。

**为 Agent 构建 Skill 就像为新员工准备入职指南一样。** 不必为每个用例构建碎片化的、定制设计的 Agent，任何人都可以通过捕获和分享他们的程序性知识，用可组合的能力来专门化他们的 Agent。在本文中，我们将解释什么是 Skills，展示它们如何工作，并分享构建你自己的 Skills 的最佳实践。

---

## Skill 的解剖结构

Skill 是一个包含 `SKILL.md` 文件的目录，该文件包含有组织的指令、脚本和资源文件夹，为 Agent 提供额外的能力。

> Skill 是一个包含 `SKILL.md` 文件的目录，该文件包含有组织的指令、脚本和资源文件夹，为 Agent 提供额外的能力。

让我们通过一个真实示例来看看 Skills 是如何运作的：一个为 Claude 最近发布的文档编辑能力提供支持的 Skill。Claude 已经对理解 PDF 了解很多，但直接操作 PDF（例如填写表单）的能力有限。这个 PDF Skill 让我们为 Claude 赋予了这些新能力。

最简单的形式是，**一个 Skill 是一个包含 `SKILL.md` 文件的目录**。这个文件必须以 YAML frontmatter 开头，包含一些必需的元数据：`name` 和 `description`。在启动时，Agent 会将每个已安装 Skill 的 `name` 和 `description` 预加载到系统提示中。

这些元数据是**渐进式披露**的第一级：它只提供足够的信息让 Claude 知道何时应该使用每个 Skill，而无需将所有内容加载到上下文中。这个文件的正文是第二级细节。如果 Claude 认为该 Skill 与当前任务相关，它就会通过读取完整的 `SKILL.md` 到上下文来加载该 Skill。

> `SKILL.md` 文件必须以 YAML Frontmatter 开头，包含文件名和描述，这些信息在启动时加载到系统提示中。

随着 Skills 复杂度的增长，它们可能包含太多上下文而无法全部塞进一个 `SKILL.md`，或者包含仅在特定场景下相关的上下文。在这种情况下，Skills 可以在 Skill 目录中捆绑额外的文件，并从 `SKILL.md` 中按名称引用它们。这些额外的链接文件是第三级（及更深层级）的细节，Claude 可以根据需要选择导航和发现它们。

在下面展示的 PDF Skill 中，`SKILL.md` 引用了两个额外文件（`reference.md` 和 `forms.md`），Skill 作者选择将它们与核心 `SKILL.md` 捆绑在一起。通过将表单填写指令移到一个单独的文件（`forms.md`）中，Skill 作者能够保持核心 Skill 的精简，相信 Claude 只会在填写表单时读取 `forms.md`。

> 你可以将更多上下文（通过额外的文件）纳入你的 Skill，然后 Claude 可以根据系统提示触发这些上下文。

**渐进式披露**是使 Agent Skills 灵活且可扩展的核心设计原则。就像一本组织良好的手册，从目录开始，然后是具体章节，最后是详细的附录，Skills 让 Claude 只在需要时加载信息：

> 拥有文件系统和代码执行工具的 Agent 在处理特定任务时，无需将整个 Skill 读入其上下文窗口。这意味着可以捆绑到 Skill 中的上下文量实际上是**不受限的**。

## Skills 与上下文窗口

下图展示了当用户消息触发 Skill 时，上下文窗口如何变化。

> Skills 通过系统提示在上下文窗口中被触发。

所示的操作顺序：

1. 开始时，上下文窗口包含核心系统提示、每个已安装 Skill 的元数据，以及用户的初始消息；
2. Claude 通过调用 Bash 工具读取 `pdf/SKILL.md` 的内容来触发 PDF Skill；
3. Claude 选择读取与 Skill 捆绑的 `forms.md` 文件；
4. 最后，Claude 现在已加载了 PDF Skill 的相关指令，继续处理用户的任务。

## Skills 与代码执行

Skills 还可以包含代码，供 Claude 自行决定作为工具执行。

大型语言模型在许多任务上表现出色，但某些操作更适合传统的代码执行。例如，通过 token 生成对列表进行排序远比简单运行排序算法要昂贵得多。除了效率问题外，许多应用还需要只有代码才能提供的确定性的可靠性。

在我们的示例中，PDF Skill 包含一个预写的 Python 脚本，用于读取 PDF 并提取所有表单字段。Claude 可以运行此脚本，而无需将脚本或 PDF 加载到上下文中。而且由于代码是确定性的，这个工作流程是一致且可重复的。

> Skills 还可以包含代码，供 Claude 根据任务性质自行决定作为工具执行。

## 开发和评估 Skills

以下是一些有助于入门编写和测试 Skills 的指南：

**从评估开始**：通过在代表性任务上运行你的 Agent 并观察它们在哪些方面遇到困难或需要额外上下文，来识别 Agent 能力中的具体差距。然后增量地构建 Skills 来解决这些不足。

**为扩展而构建**：当 `SKILL.md` 文件变得难以管理时，将其内容拆分到单独的文件中并引用它们。如果某些上下文互斥或很少一起使用，保持路径分离将减少 token 用量。最后，代码既可以作为可执行工具，也可以作为文档。应该明确 Claude 是应该直接运行脚本还是将其读入上下文作为参考。

**从 Claude 的角度思考**：观察 Claude 在真实场景中如何使用你的 Skill，并根据观察进行迭代：注意意外的轨迹或对某些上下文的过度依赖。特别关注你的 Skill 的 `name` 和 `description`。Claude 将使用这些信息来决定是否在响应当前任务时触发该 Skill。

**与 Claude 一起迭代**：当你与 Claude 一起处理任务时，请 Claude 将其成功的方法和常见错误捕获到 Skill 中的可复用上下文和代码中。如果它在使用 Skill 完成任务时偏离了方向，请它自我反思哪里出了问题。这个过程将帮助你发现 Claude 实际需要什么上下文，而不是试图预先猜测。

## 使用 Skills 时的安全考虑

Skills 通过指令和代码为 Claude 提供新能力。虽然这使它们强大，但也意味着恶意 Skill 可能会在使用它们的环境中引入漏洞，或引导 Claude 泄露数据和执行意外操作。

我们建议**只从可信来源安装 Skills**。当从不太可信的来源安装 Skill 时，请在使用前彻底审计它。首先阅读 Skill 中捆绑的文件内容，了解它的作用，特别注意代码依赖和捆绑的资源（如图像或脚本）。同样，注意 Skill 中指示 Claude 连接到潜在不可信的外部网络源的指令或代码。

## Skills 的未来

Agent Skills 目前已得到 Claude.ai、Claude Code、Claude Agent SDK 和 Claude Developer Platform 的支持。

在接下来的几周里，我们将继续添加支持创建、编辑、发现、分享和使用 Skills 的完整生命周期的功能。我们特别期待 Skills 帮助组织和个人与 Claude 分享他们的上下文和工作流程的机会。我们还将探索 Skills 如何通过教会 Agent 涉及外部工具和软件的更复杂工作流程，来补充模型上下文协议（MCP）服务器。

放眼未来，我们希望让 Agent 能够**自主创建、编辑和评估** Skills，让它们将自己的行为模式编码为可复用的能力。

Skills 是一个简单的概念，拥有相应简单的格式。这种简单性使组织、开发者和最终用户更容易构建定制化的 Agent 并赋予它们新能力。

我们很期待看到人们用 Skills 构建什么。立即开始，查看我们的 [Skills 文档](https://docs.anthropic.com/en/docs/build-with-claude/agent-skills) 和 [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agent-skills)。

## 致谢

本文由 Barry Zhang、Keith Lazuka 和 Mahesh Murag 撰写，他们都是文件夹的忠实爱好者。特别感谢 Anthropic 中许多倡导、支持和构建了 Skills 的同事们。
