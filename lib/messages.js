/**
 * Harness Message → OpenAI chat-completions wire conversion.
 *
 * The wire shape deliberately mirrors what the dsh-llm-pi-ai adapter sends
 * for the same route (system slot, in-history system messages as user
 * messages, tool results as `role: "tool"` messages, assistant content as a
 * plain string, empty assistant messages skipped). A session therefore sees
 * one consistent prompt whether the thinking-mode plugin routes a request
 * directly or the stock adapter handles it.
 */

/** Join the text blocks of one Harness message. */
function flattenText(message) {
  return message.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("");
}

/** Flatten text recursively inside one tool result. */
function toolResultText(blocks) {
  return blocks
    .map((block) =>
      block.type === "text"
        ? block.text
        : block.type === "tool-result"
          ? toolResultText(block.content)
          : ""
    )
    .join("");
}

/** Keep raw tool-argument JSON only when it parses to an object. */
function normalizeToolArguments(raw) {
  if (typeof raw === "string" && raw.length > 0) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) return raw;
    } catch {
      // malformed model output: fall through to "{}"
    }
  }
  return "{}";
}

/**
 * Convert Harness request messages to OpenAI wire messages.
 *
 * @param {object} options - the GenerateOptions request (system, messages).
 * @param {object} [attachments] - the attachment store, required only when a
 *   user message carries image blocks.
 * @returns {Promise<Array<object>>} the wire message list.
 */
export async function convertMessages(options, attachments) {
  const out = [];
  if (typeof options.system === "string" && options.system.length > 0) {
    out.push({ role: "system", content: options.system });
  }

  for (const message of options.messages) {
    if (message.role === "system") {
      // pi-ai parity: in-history system messages ride as user messages.
      const text = flattenText(message);
      if (text.length > 0) out.push({ role: "user", content: text });
      continue;
    }

    if (message.role === "assistant") {
      const text = message.content
        .filter((block) => block.type === "text" && block.text.trim().length > 0)
        .map((block) => block.text)
        .join("");
      const toolCalls = message.content.filter((block) => block.type === "tool-call");
      const assistantMsg = { role: "assistant", content: text.length > 0 ? text : null };
      if (toolCalls.length > 0) {
        assistantMsg.tool_calls = toolCalls.map((tc) => ({
          id: String(tc.id),
          type: "function",
          function: {
            name: tc.name,
            arguments: normalizeToolArguments(tc.arguments),
          },
        }));
      }
      // Providers require "either content or tool_calls, but not none".
      if (assistantMsg.content === null && toolCalls.length === 0) continue;
      out.push(assistantMsg);
      continue;
    }

    // user message
    const images = message.content.filter((block) => block.type === "image");
    const results = message.content.filter((block) => block.type === "tool-result");
    if (images.length > 0) {
      if (attachments === undefined) {
        throw new Error("thinking-mode intercept: image content requires the attachment service");
      }
      const content = [];
      for (const block of message.content) {
        if (block.type === "text" && block.text.length > 0) {
          content.push({ type: "text", text: block.text });
        } else if (block.type === "image") {
          const stored = await attachments.readImage(block.attachment);
          content.push({
            type: "image_url",
            image_url: {
              url: `data:${stored.ref.mediaType};base64,${Buffer.from(stored.data).toString("base64")}`,
            },
          });
        }
      }
      if (content.length > 0) out.push({ role: "user", content });
    } else {
      const text = flattenText(message);
      if (text.length > 0 || results.length === 0) out.push({ role: "user", content: text });
    }
    for (const result of results) {
      out.push({
        role: "tool",
        content: toolResultText(result.content) || "(no tool output)",
        tool_call_id: String(result.toolCallId),
      });
    }
  }

  return out;
}

/**
 * Qwen3.8 recommended sampling parameters (model card). Both presets share
 * top_k=20, min_p=0.0, repetition_penalty=1.0; they differ in
 * temperature / top_p / presence_penalty, which are applied per mode.
 * (top_k / min_p / repetition_penalty were verified accepted by the local
 * SGLang endpoint; SGLang silently ignores unknown keys.)
 */
export const SAMPLING = {
  thinking: {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 0.0,
    repetition_penalty: 1.0,
  },
  instruct: {
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 1.5,
    repetition_penalty: 1.0,
  },
};

/** Full per-mode preset shape: the three differing fields plus the shared three. */
const PRESET_KEYS = [
  ["temperature", "number"],
  ["top_p", "number"],
  ["top_k", "number"],
  ["min_p", "number"],
  ["presence_penalty", "number"],
  ["repetition_penalty", "number"],
];

/** JSON-schema object for one preset (used by the tool output schema). */
export function presetJsonSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: Object.fromEntries(PRESET_KEYS.map(([key, type]) => [key, { type, required: true }])),
  };
}

/** Schemastery builder for one preset (used by the plugin config/state). */
export function presetSchemastery(z) {
  return z.object(Object.fromEntries(PRESET_KEYS.map(([key, type]) => [key, z.number()])));
}

/**
 * Official Qwen3.8 reasoning_effort levels (model card). The OpenAI
 * compatible endpoint accepts none/minimal/low/medium/high/xhigh/max
 * (verified against the local SGLang: the invalid value is rejected with
 * HTTP 400, so we only ever send official levels).
 */
export const REASONING_EFFORTS = {
  /** Provider default — thorough analysis, highest cost. */
  xhigh: "xhigh",
  /** Balance of accuracy and speed. */
  medium: "medium",
  /** Efficient reasoning optimized for speed and cost. */
  low: "low",
};
export const REASONING_EFFORT_KEYS = Object.keys(REASONING_EFFORTS);
export const DEFAULT_REASONING_EFFORT = "xhigh";

/**
 * Build the chat-completions request body.
 *
 * @param {object} options - the GenerateOptions request.
 * @param {boolean} thinkingOn - the enable_thinking value to force.
 * @param {Array<object>} wireMessages - messages from convertMessages.
 * @param {object} [p] - sampling: {thinking, instruct} preset tables applied
 *   per mode (undefined = stock sampling passthrough, no overrides);
 *   effort: the active reasoning_effort (sent in 'on' mode; in 'off' mode
 *   only when it is not the provider default);
 *   preserveThinking: the user's preserve_thinking choice — sent as
 *   chat_template_kwargs.preserve_thinking while thinking is on, and
 *   forced to false while thinking is off (per the user's convention).
 * @returns {object} the JSON request body.
 */
export function buildBody(options, thinkingOn, wireMessages, { sampling, effort, preserveThinking } = {}) {
  const body = {
    model: options.model,
    messages: wireMessages,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (options.maxTokens !== undefined) body.max_tokens = options.maxTokens;
  if (Array.isArray(options.stop) && options.stop.length > 0) body.stop = options.stop;
  if (Array.isArray(options.tools) && options.tools.length > 0) {
    body.tools = options.tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    }));
  }
  // The per-request thinking switch (Qwen3 model card, Chat Completions API).
  // preserve_thinking: while thinking is on, whether prior turns' reasoning
  // content stays in the context (the user's choice, persisted in the
  // settings section); while thinking is off it is always false, sent
  // explicitly for determinism.
  body.chat_template_kwargs = {
    enable_thinking: thinkingOn,
    preserve_thinking: thinkingOn && preserveThinking === true,
  };
  // Mode-matched sampling: when presets are configured, the plugin forces the
  // model card's recommended sampling for the active mode (temperature,
  // top_p, presence_penalty differ between modes; top_k / min_p /
  // repetition_penalty are shared). Without presets, stock sampling passes
  // through untouched (temperature and friends ride on `options`).
  const preset = sampling?.[thinkingOn ? "thinking" : "instruct"];
  if (preset !== undefined && preset !== null) {
    for (const [key, value] of Object.entries(preset)) {
      if (typeof value === "number") body[key] = value;
    }
  } else if (options.temperature !== undefined) {
    body.temperature = options.temperature;
  }
  // reasoning_effort (Qwen3 model card): sent top-level in 'on' mode; in
  // 'off' mode only when the user explicitly chose a non-default level
  // (with thinking off, effort has no effect on the provider anyway).
  if (
    typeof effort === "string" &&
    Object.values(REASONING_EFFORTS).includes(effort) &&
    (thinkingOn || effort !== DEFAULT_REASONING_EFFORT)
  ) {
    body.reasoning_effort = effort;
  }
  return body;
}
