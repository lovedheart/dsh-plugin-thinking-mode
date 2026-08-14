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
 * Build the chat-completions request body.
 *
 * @param {object} options - the GenerateOptions request.
 * @param {boolean} thinkingOn - the enable_thinking value to force.
 * @param {Array<object>} wireMessages - messages from convertMessages.
 * @returns {object} the JSON request body.
 */
export function buildBody(options, thinkingOn, wireMessages) {
  const body = {
    model: options.model,
    messages: wireMessages,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (options.temperature !== undefined) body.temperature = options.temperature;
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
  body.chat_template_kwargs = { enable_thinking: thinkingOn };
  return body;
}
