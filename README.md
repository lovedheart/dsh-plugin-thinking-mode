# dsh-plugin-thinking-mode

DeepSeek Harness 插件：为 **Qwen3 系列模型**（Qwen3.8-27B 等，经 OpenAI 兼容端点提供，如 SGLang / vLLM / Qwen Cloud 兼容网关）**切换思考（thinking）模式**。

参考：

- 模型卡片 <https://huggingface.co/Qwen/Qwen3.8-27B> —— Qwen3.8 默认**开启**思考，可通过 Chat Completions API 的 `chat_template_kwargs: { enable_thinking: true|false }` 按请求开关。
- DSH 插件教程 <https://deepseek-harness.github.io/deepseek-harness/develop/basic/>

## 功能

| 组件 | 说明 |
| --- | --- |
| `thinking_mode` 工具 | 模型或用户可调用：`get` / `on` / `off` / `auto` / `toggle`，并可**可选带 `reasoningEffort` 参数**（`xhigh` / `medium` / `low`）在同一调用里改推理深度。输出当前模式、说明、是否变更、两套采样预设、当前 `reasoningEffort`，以及默认模型路由上拦截是否生效（`interceptActive`）。 |
| `thinking-mode` 设置节 | 持久化在 `~/.dsh/settings.yaml`，`applies: live` 即时生效，并在 Web 设置页中显示（设置页里也能直接切换，是第二个切换入口）。 |
| 系统提示段 | 每轮重新渲染，向模型报告当前模式及可用操作（模型知道如何响应"关闭思考/打开思考"这类指令）。 |
| `llm/stream` 拦截器 | 全局 waterfall 监听器。当模式为 `on`/`off` **且** 请求命中匹配的 provider/model 时，插件用自建的 OpenAI 兼容客户端直接发流式请求，与标准 pi-ai 请求相比的差异是请求体多了 `chat_template_kwargs.enable_thinking`、按模式的采样预设（见上表）与 `reasoning_effort`（见上节）；其余报文（消息转换、工具、usage、SSE 解析）逐字段对齐标准适配器。`auto` 模式为纯透传，零改动。 |

### 三种模式

- **auto**（默认）：完全透传，使用 provider 自身默认（Qwen3.8 默认开思考）。零风险，不影响现有行为。
- **on**：强制 `enable_thinking: true` —— 模型先推理再回答（产生 reasoning 内容）。
- **off**：强制 `enable_thinking: false` —— 直接回答，更快更省（实测 27B 上约 4 倍响应事件数差异，token 更少）。

模式是**实例级全局状态**（一个设置节，所有会话共享）——v1 的取舍，README 级别记录于此。

### 按模式强制官方推荐采样参数

拦截路由同时会按模式覆盖请求体里的采样参数（Qwen3.8 模型卡片推荐值，SGLang wire 已验证接受）：

| 参数 | on（thinking） | off（instruct） |
| --- | --- | --- |
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |

预设是**设置节字段**（`sampling`，Web 设置页可编辑），默认即上表；`applySampling: false` 可关闭覆盖、恢复原始采样透传。`auto` 模式始终不碰采样参数（纯透传）。

### reasoning_effort（推理深度档位）

Qwen3.8 官方支持 `reasoning_effort` 调节推理深度与成本：

| 档位 | 说明 |
| --- | --- |
| `xhigh`（默认） | 复杂任务、彻底分析，成本最高 |
| `medium` | 精度与速度平衡 |
| `low` | 高效推理，最省最快 |

- 设置节字段 `reasoningEffort`（三种改法：Web 设置页直接改；`thinking_mode` 工具带 `reasoningEffort` 参数在同一调用里改；或在对话里让我改）
- 发送策略：`on` 模式始终发送（顶层 `reasoning_effort` 字段）；`off` 模式仅在选了非默认档位时发送（思考关闭时 provider 侧本无效果）；`auto` 模式永不发送（纯透传）
- 端点合法值（实测 SGLang）：`none/minimal/low/medium/high/xhigh/max`；非法值返回 HTTP 400，所以插件只发送三个官方档位

## 工作原理（安全性说明）

1. 插件在 `llm/stream` waterfall 上注册 `global` 监听器（文档化的网关模式）。
2. 每次模型调用时按顺序判断：模式是否 `auto`？→ provider 设置节（`llm-pi-ai`，实时读取）里该 provider 的 `api` 是否为 `openai-completions`？→ 有无 `baseURL`？→ 模型 id 是否命中 `modelPatterns`（不区分大小写子串，默认 `["qwen"]`）？→ 用户消息含图片时附件服务是否可用？
3. 任何一项不满足 → `yield* await next()` 走标准适配器路径（**所有不确定性都退化为透传**）。
4. 命中时直接请求 `baseURL/chat/completions`，SSE 转成协议合法的 StreamChunk 序列，并合成 pi-ai `replayState`（`kind: 'pi-ai'`, `version: 1`, `api: 'openai-completions'`），使后续轮次的历史重建与标准路径完全一致（会话回放不受影响）。
5. 所有输出都经过 dsh-llm 全局流不变量校验（块配对、usage 唯一、finish 终止等）。

## 切换为什么需要等待（延迟构成）

切换动作**本身是瞬时的**（一次本地设置写入，毫秒级）。对话里"打开/关闭思考"感知到的等待来自**围绕它的完整模型推理**，共三层：

1. **一次对话切换 = 两次完整模型调用**。第 1 次：模型处理指令、决定调用 `thinking_mode` 工具（需预编码整个会话上下文）；工具执行（瞬时）；第 2 次：模型读取工具结果、生成确认回复（再过一遍上下文）。首 token 延迟（TTFT）随上下文大小近似线性增长，大上下文会话每次调用要等数秒到十几秒。
2. **新模式从下一次模型调用才生效**。当前正在生成的回复仍由旧模式完成，所以真正体验到新模式要再发一条消息、再等一次完整推理——感知上像"切了两次才见效"。
3. **on 模式正文更晚出现**。模型先输出 `reasoning_content` 再输出正文，推理 token 也算生成时间（`low` 档额外约几百毫秒到 1 秒，不是大头）。

实测 TTFT 参考（本机 SGLang + Qwen3.8-27B，2026-08-15）：

| 请求规模 | 思考关（off） | 思考开（on + low） |
| --- | --- | --- |
| ~2k token | — | 684ms |
| ~20k token | 7.9s | 7.2s |
| ~50k token | 17.2s | — |

**减少等待的办法：**

- **用 Web 设置页直接改**（最快）：设置 → `thinking-mode` 节，改 `mode` / `reasoningEffort` 保存——**零模型调用**，下次模型请求即生效。
- **新开会话**：上下文小，每次调用约 1 秒内出首 token。
- 对话里切换的等待省不掉——它是"让模型执行指令"的固有成本；切换频繁时建议走设置页。

## 安装

> 适用版本：DSH `0.1.0-rc.6`（peer 依赖即钉在此版本；升级 DSH 后请确认 peer 依赖仍满足）。

### 方式 A：npm 安装（推荐，已发布形态）

```sh
dsh plugin --profile web add dsh-plugin-thinking-mode
```

插件的 `package.json` 声明了 `dsh.bundle.patch`（`cordis.patch.yml`）与 peer 依赖（dsh-tools / dsh-system-prompt / dsh-llm / dsh-settings / cordis），`dsh plugin add` 会从 npm 拉取并注入 bundle 补丁。安装后 `thinking_mode` 工具、`thinking-mode` 设置节、系统提示段与 `llm/stream` 拦截器即自动注册。

### 方式 B：本地源码（开发 / 未发布时，HMR 热加载）

把本地源码目录的补丁行追加到 `~/.dsh/profiles/<profile>/cordis.patch.yml`（绝对路径免 pnpm 安装）：

```yaml
- insert:
    - id: thinking-mode
      name: '/path/to/your/local/dsh-plugin-thinking-mode/lib/index.js'
      config:
        defaultMode: auto
        modelPatterns:
          - qwen
```

把 `name` 换成你本地的实际路径。运行中的 `dsh web` 进程会由 HMR/launcher watch 热加载用户补丁层；未生效时重启 `dsh web` 即可。

## 配置

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `defaultMode` | `"auto" \| "on" \| "off"` | `"auto"` | 用户未显式选择前的模式 |
| `modelPatterns` | `string[]` | `["qwen"]` | 对模型 id 做不区分大小写的子串匹配；只有命中的模型才会被拦截 |
| `applySampling` | `boolean` | `true` | 拦截时是否按模式强制官方推荐采样参数 |
| `sampling` | 对象 | 见上表 | 两套预设 `thinking` / `instruct`，各含 6 个采样字段（Web 设置页可改） |
| `defaultReasoningEffort` | `"xhigh" \| "medium" \| "low"` | `"xhigh"` | 用户未显式选择前的推理深度档位 |

运行时状态（`thinking-mode` 设置节的 `mode`）优先于 `defaultMode`。

## 开发 / 测试

```sh
# 独立集成测试（mock SSE 服务器 + 独立 cordis Context，不依赖 dsh 运行实例）：
ln -s ~/.dsh/profiles/node_modules node_modules   # 开发用符号链接（已 gitignore）
npm test          # = node test/harness-test.mjs
```

测试覆盖：Config 默认值、工具五种动作与状态迁移、系统提示动态文本、`off`/`on` 直连路由（请求体逐字段断言：`chat_template_kwargs`、工具函数形态、消息 wire 转换、鉴权头）、chunk 流语法校验（不变量规则复现）、`replayState` 精确断言、`auto`/非匹配模型透传、HTTP 500 与空响应终止、图片请求在附件服务缺省时的退化与存在时的直连。

已验证（本机 SGLang + Qwen3.8-27B 实测）：

- `enable_thinking: false` 时响应无 `reasoning_content`，更快；`true` 时有推理内容。
- **off 模式 + 工具调用正常**（`finish_reason: tool_calls`，名称与参数正常流式返回）——agent 循环在拦截路径上完整可用。
- 沙箱 DSH_HOME 中完整 dsh headless 启动 + 真实模型跑通：工具切换、拦截激活（`interceptActive: true`）、状态跨重启持久化。

## 文件结构

```
├── package.json          # 插件清单（dsh.bundle.patch、peer 依赖、发布字段）
├── cordis.patch.yml      # bundle 层补丁（npm 安装时自动注入）
├── LICENSE               # MIT
├── lib/
│   ├── index.js          # 入口：工具 + 设置节 + 系统提示段 + 安装拦截器
│   ├── intercept.js      # llm/stream waterfall 拦截 + 路由判定
│   ├── messages.js       # Harness 消息 → OpenAI wire 转换（对齐 pi-ai 适配器）
│   └── stream.js         # OpenAI SSE → StreamChunk + replayState 合成
└── test/
    └── harness-test.mjs  # 独立集成测试（npm test）
```

## 限制

- 状态为实例级全局，不是每会话（v1 取舍）。
- 仅拦截 `openai-completions` 路由的匹配模型；其他 provider/路由一律透传。
- 采样预设按 Qwen3.8 模型卡片写死为默认值；`top_k`/`min_p`/`repetition_penalty` 在 SGLang 上已验证接受，其他 OpenAI 兼容端点若不接受未知键会自行忽略（SGLang 即忽略未知键，已实测）。
- `reasoning_effort` 当前提供三个官方档位（xhigh/medium/low）；端点还支持 none/minimal/high/max，可后续按需扩展。
- 图片请求要求附件服务存在（dsh 标准组合默认具备）；缺失时透传。
