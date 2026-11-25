/**
 * Harness Message → OpenAI chat-completions wire conversion.
 *
 * The wire shape mirrors the stock route end to end: the harness→pi-ai
 * conversion of dsh-llm-pi-ai (leading-system folding, assistant history
 * rebuilt through the pi-ai replay state, tool-role results with a tool-name
 * map, request-image normalization handles) followed by pi-ai's
 * convertMessages/convertTools. A session therefore sees one consistent
 * prompt whether the thinking-mode plugin routes a request directly or the
 * stock adapter handles it — the precondition for route flips mid-session not
 * re-prefilling the server-side prefix/session cache.
 *
 * Constructs the mirror does not reproduce raise UnsupportedHistory and the
 * intercept passes the request through to the stock route instead.
 */
import {
  UnsupportedHistory,
  REASONING_FIELDS,
  normalizeToolCallId,
  offloadedImageText,
  parseArguments,
  parseLegacyEncryptedReasoningDetail,
  parseOpenAIReasoningDetails,
  readPiAiReplayState,
  requestImageHandleText,
  requestImageTarget,
  sanitizeSurrogates,
} from "./parity.js";

/** Join the text blocks of one Harness message. */
function flattenText(message) {
  return message.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("");
}

/** Flatten text recursively inside one legacy tool-result content block. */
function toolResultText(blocks) {
  return blocks
    .map((block) =>
      block.type === "text"
        ? block.text
        : block.type === "tool-result"
          ? toolResultText(block.content)
          : "",
    )
    .join("");
}

/**
 * Replay a durable assistant message into neutral blocks, mirroring
 * dsh-llm-pi-ai's toPiAssistant plus pi-ai's transformMessages pass.
 *
 * A valid, matching pi-ai replay state reconstructs the native message
 * (thinking blocks keep their signatures → reasoning_content is replayed);
 * anything unusable degrades the whole message to provider-neutral history,
 * where foreign thinking is dropped and signatures are stripped.
 */
export function assistantHistoryView(message, { provider, model }) {
  const source = message.source ?? {};
  let blocks;
  let sameModel = false;
  if (source.replayState !== undefined) {
    try {
      const state = readPiAiReplayState(source.replayState);
      if (state.response.provider !== source.provider) throw new Error("provider does not match assistant source");
      if (state.response.model !== source.model) throw new Error("model does not match assistant source");
      if (state.blocks.length !== message.content.length) throw new Error("block count does not match assistant content");
      blocks = message.content.map((block, index) => {
        const replay = state.blocks[index];
        if (replay.type !== block.type) throw new Error(`block ${index} does not match assistant content`);
        switch (block.type) {
          case "text":
            return { kind: "text", text: block.text };
          case "reasoning":
            return {
              kind: "thinking",
              thinking: block.text,
              ...(replay.thinkingSignature === undefined ? {} : { thinkingSignature: replay.thinkingSignature }),
              ...(replay.redacted === undefined ? {} : { redacted: replay.redacted }),
            };
          case "tool-call":
            return {
              kind: "toolCall",
              id: String(block.id),
              name: block.name,
              arguments: parseArguments(block.arguments),
              ...(replay.thoughtSignature === undefined ? {} : { thoughtSignature: replay.thoughtSignature }),
            };
          default:
            throw new Error(`block ${index} has an unsupported Harness type`);
        }
      });
      sameModel =
        state.response.api === "openai-completions" &&
        state.response.provider === provider &&
        state.response.model === model;
    } catch {
      blocks = undefined; // degrade to provider-neutral, like toPiAssistant does
    }
  }
  if (blocks === undefined) {
    // foreignAssistant + transformMessages(!isSameModel): thinking dropped,
    // signatures stripped, text and tool calls kept.
    blocks = [];
    for (const block of message.content) {
      if (block.type === "text") blocks.push({ kind: "text", text: block.text });
      else if (block.type === "tool-call")
        blocks.push({
          kind: "toolCall",
          id: String(block.id),
          name: block.name,
          arguments: parseArguments(block.arguments),
        });
    }
    return { blocks, sameModel: false };
  }
  if (sameModel) {
    // transformMessages(isSameModel): keep redacted and signed thinking; an
    // unsigned empty thinking block contributes nothing.
    blocks = blocks.filter(
      (block) =>
        block.kind !== "thinking" ||
        block.redacted === true ||
        block.thinkingSignature !== undefined ||
        (block.thinking !== undefined && block.thinking.trim() !== ""),
    );
  } else {
    blocks = blocks
      .filter((block) => block.kind !== "thinking")
      .map((block) => (block.kind === "toolCall" ? { ...block, thoughtSignature: undefined } : block));
  }
  return { blocks, sameModel };
}

/** Prepare every request image once (dsh-llm-pi-ai prepareRequestImages). */
async function prepareRequestImages(messages, attachments, policy, signal) {
  const refs = new Map();
  for (const message of messages)
    for (const block of message.content)
      if (block.type === "image" && block.offloaded !== true && !refs.has(block.attachment.attachmentId))
        refs.set(block.attachment.attachmentId, block.attachment);
  const orderedRefs = [...refs.values()];
  const prepared = await Promise.all(
    orderedRefs.map(async (ref) =>
      attachments.readImageRequest(ref, await requestImageTarget(ref, policy), signal),
    ),
  );
  const versions = new Map();
  orderedRefs.forEach((ref, index) => versions.set(ref.attachmentId, prepared[index]));
  return versions;
}

/** Oldest-first image omissions required by the route budgets (proxy). */
function requiredImageOffload(occurrences, { maxBytes, maxImages }) {
  // Same accounting basis as stock: base64 length of the request bytes; fall
  // back to the payload when the service does not report `bytes`.
  const sizeOf = (version) =>
    version?.bytes ?? (version?.data === undefined ? 0 : Math.ceil(Buffer.byteLength(version.data) / 3) * 4);
  if (maxImages !== undefined && occurrences.length > maxImages) return occurrences.length - maxImages;
  if (maxBytes !== undefined) {
    let total = 0;
    for (const version of occurrences) total += sizeOf(version);
    if (total > maxBytes) {
      let omitted = 0;
      for (const version of occurrences) {
        if (total <= maxBytes) break;
        total -= sizeOf(version);
        omitted += 1;
      }
      return omitted;
    }
  }
  return 0;
}

/**
 * Convert Harness request messages to OpenAI wire messages with stock-route
 * parity.
 *
 * @param {object} options - the GenerateOptions request (system, messages).
 * @param {object} [attachments] - the attachment service, required only when
 *   a request carries image blocks.
 * @param {object} ctx - {compat, modelInput, requestImagePolicy,
 *   maxRequestImageBytes, maxRequestImages} for the resolved route.
 * @returns {Promise<Array<object>>} the wire message list.
 */
export async function convertMessages(options, attachments, ctx = {}) {
  const compat = ctx.compat ?? {};
  const instructionRole = compat.supportsDeveloperRole ? "developer" : "system";
  const modelInput = ctx.modelInput ?? [];
  // model.reasoning in pi-ai's terms: the route's model declares reasoning.

  // Anything the mirror does not reproduce goes back to the stock route
  // (dsh-llm-pi-ai's assertSupportedHistory plus the tool-update path).
  const CONTENT_TYPES = {
    system: ["text"],
    developer: ["text", "tool-addition", "tool-removal"],
    user: ["text", "image", "tool-result"],
    tool: ["text", "image"],
    assistant: ["text", "reasoning", "tool-call"],
  };
  for (const message of options.messages) {
    if (!["system", "developer", "user", "assistant", "tool"].includes(message.role))
      throw new UnsupportedHistory(`unknown message role ${message.role}`);
    for (const block of message.content) {
      if (!CONTENT_TYPES[message.role].includes(block.type))
        throw new UnsupportedHistory(`unsupported ${message.role} content block type ${block.type}`);
      if (block.type === "image" && (block.attachment?.attachmentId === undefined || block.attachment?.attachmentId === null))
        throw new UnsupportedHistory("image block has no attachment reference");
      if (
        (block.type === "tool-addition" || block.type === "tool-removal") &&
        message.role !== "developer"
      )
        throw new UnsupportedHistory(`tool-change block requires the developer role, not ${message.role}`);
    }
  }
  // Only non-offloaded images reach the model: they need the attachment
  // service, the route budget, and a model that declares image input
  // (offloaded ones are already stable text handles, like stock keeps them).
  const hasRequestImages = options.messages.some((message) =>
    message.content.some((block) => block.type === "image" && block.offloaded !== true),
  );
  if (hasRequestImages) {
    if (!modelInput.includes("image")) throw new UnsupportedHistory("model route does not declare image input");
    if (attachments === undefined) throw new UnsupportedHistory("image input requires the attachment service");
  }

  // Leading system messages fold into the initial prompt, "text-only" style.
  const headParts = [];
  if (typeof options.system === "string" && options.system.length > 0) headParts.push(options.system);
  let cursor = 0;
  while (cursor < options.messages.length && options.messages[cursor].role === "system") {
    const text = flattenText(options.messages[cursor]);
    if (text.length > 0) headParts.push(text);
    cursor += 1;
  }
  const systemPrompt = headParts.join("\n\n") || undefined;

  let versions;
  if (hasRequestImages) {
    versions = await prepareRequestImages(options.messages, attachments, ctx.requestImagePolicy ?? {}, options.signal);
    const occurrences = [];
    for (const message of options.messages)
      for (const block of message.content)
        if (block.type === "image" && block.offloaded !== true) occurrences.push(versions.get(block.attachment.attachmentId));
    const required = requiredImageOffload(occurrences, {
      maxBytes: ctx.maxRequestImageBytes,
      maxImages: ctx.maxRequestImages,
    });
    if (required > 0) throw new UnsupportedHistory(`${required} request image(s) need offloading for the route budget`);
  }

  const out = [];
  if (systemPrompt !== undefined) out.push({ role: instructionRole, content: sanitizeSurrogates(systemPrompt) });

  const normalizeId = (id) => normalizeToolCallId(String(id), options.provider);
  // tool-call ids → names, for routes that require the name in tool results.
  const toolNames = new Map();
  let lastRole = null;
  const bridgeAssistant = () => ({ role: "assistant", content: "I have processed the tool results." });

  /**
   * Render harness content blocks the way userContent/toolResultOf do.
   * Returns a plain string when every part is text, wire parts otherwise;
   * `images` collects the image_url parts produced (tool-result wrapper) and
   * `toolResults` collects wire messages for legacy tool-result blocks, so
   * the caller can emit them in block order instead of mid-render.
   */
  const renderBlocks = (blocks, images, toolResults) => {
    const parts = [];
    for (const block of blocks) {
      switch (block.type) {
        case "text":
          if (block.text.length > 0) parts.push({ type: "text", text: block.text });
          break;
        case "image": {
          if (block.offloaded === true) {
            parts.push({ type: "text", text: offloadedImageText(block.attachment, undefined) });
            break;
          }
          const version = versions.get(block.attachment.attachmentId);
          if (version === undefined || version.data === undefined || version.mediaType === undefined)
            throw new UnsupportedHistory(
              `request image ${block.attachment.attachmentId} has no prepared request version`,
            );
          parts.push({ type: "text", text: requestImageHandleText(block.attachment, version, undefined) });
          const url = `data:${version.mediaType};base64,${Buffer.from(version.data).toString("base64")}`;
          if (images !== undefined) images.push(url);
          parts.push({ type: "image", url });
          break;
        }
        case "tool-result": {
          // Legacy shape: a result riding inside a user message.
          toolResults.push({
            role: "tool",
            content: sanitizeSurrogates(toolResultText(block.content) || "(no tool output)"),
            tool_call_id: normalizeId(block.toolCallId),
            ...(compat.requiresToolResultName ? { name: toolNames.get(String(block.toolCallId)) ?? "unknown" } : {}),
          });
          break;
        }
        default:
          // Unreachable while the pre-flight whitelist holds; kept so a future
          // block type can never be silently dropped from the wire prompt.
          throw new UnsupportedHistory(`unsupported content block type ${block.type}`);
      }
    }
    if (parts.every((part) => part.type === "text")) return parts.map((part) => part.text).join("");
    return parts;
  };

  const pushUser = (rendered) => {
    if (typeof rendered === "string") {
      out.push({ role: "user", content: sanitizeSurrogates(rendered) });
      return;
    }
    const content = rendered
      .filter((part) => part.type !== "text" || part.text.length > 0)
      .map((part) =>
        part.type === "text"
          ? { type: "text", text: sanitizeSurrogates(part.text) }
          : { type: "image_url", image_url: { url: part.url } },
      );
    if (content.length === 0) return;
    out.push({ role: "user", content });
  };

  const messages = options.messages;
  for (let i = 0; i < messages.length; i += 1) {
    const message = messages[i];
    if (i < cursor) continue; // leading system messages already folded

    if (
      compat.requiresAssistantAfterToolResult &&
      lastRole === "toolResult" &&
      (message.role === "user" || message.role === "tool")
    )
      out.push(bridgeAssistant());

    if (message.role === "system" || message.role === "developer") {
      // Mid-conversation instruction: plain instruction role when the route
      // keeps system messages in history, otherwise a <system-reminder>-
      // wrapped user message (dsh-llm-pi-ai appendSharedHistory framing).
      const text = flattenText(message);
      if (text.length > 0) {
        if (compat.supportsMidConvoSystemMessages) out.push({ role: instructionRole, content: sanitizeSurrogates(text) });
        else {
          const reminder = text.replaceAll("</system-reminder>", "<\\/system-reminder>");
          out.push({ role: "user", content: sanitizeSurrogates(`<system-reminder>\n${reminder}\n</system-reminder>`) });
        }
      }
      lastRole = "system";
      continue;
    }

    if (message.role === "assistant") {
      const { blocks } = assistantHistoryView(message, {
        provider: options.provider,
        model: options.model,
      });
      const assistantMsg = {
        role: "assistant",
        content: compat.requiresAssistantAfterToolResult || compat.requiresNonNullAssistantContent ? "" : null,
      };
      const textParts = blocks
        .filter((block) => block.kind === "text" && block.text.trim().length > 0)
        .map((block) => ({ type: "text", text: sanitizeSurrogates(block.text) }));
      const assistantText = textParts.map((part) => part.text).join("");
      const thinking = blocks.filter(
        (block) => block.kind === "thinking" && block.thinking !== undefined && block.thinking.trim().length > 0,
      );
      const toolCalls = blocks.filter((block) => block.kind === "toolCall");
      for (const toolCall of toolCalls) if (!toolNames.has(toolCall.id)) toolNames.set(toolCall.id, toolCall.name);
      const signedDetails = thinking
        .map((block) => parseOpenAIReasoningDetails(block.thinkingSignature))
        .find((details) => details !== undefined);
      const legacyDetails = toolCalls
        .map((toolCall) => parseLegacyEncryptedReasoningDetail(toolCall.thoughtSignature))
        .filter((detail) => detail !== undefined);
      const preservedDetails = signedDetails ?? (legacyDetails.length > 0 ? legacyDetails : undefined);
      if (thinking.length > 0) {
        if (compat.requiresThinkingAsText) {
          const thinkingText = thinking.map((block) => sanitizeSurrogates(block.thinking)).join("\n\n");
          assistantMsg.content = [{ type: "text", text: thinkingText }, ...textParts];
        } else {
          if (assistantText.length > 0) assistantMsg.content = assistantText;
          if (preservedDetails === undefined) {
            let signature = thinking[0].thinkingSignature;
            if (options.provider === "opencode-go" && signature === "reasoning") signature = "reasoning_content";
            if (signature !== undefined && REASONING_FIELDS.includes(signature))
              assistantMsg[signature] = thinking.map((block) => block.thinking).join("\n");
          }
        }
      } else if (assistantText.length > 0) {
        assistantMsg.content = assistantText;
      }
      if (toolCalls.length > 0) {
        assistantMsg.tool_calls = toolCalls.map((toolCall) => ({
          id: normalizeId(toolCall.id),
          type: "function",
          function: { name: toolCall.name, arguments: JSON.stringify(toolCall.arguments) },
        }));
      }
      if (preservedDetails !== undefined) assistantMsg.reasoning_details = preservedDetails;
      // pi-ai only seeds for routes that both demand the field and run a
      // reasoning-declared model.
      if (
        compat.requiresReasoningContentOnAssistantMessages &&
        ctx.modelReasoning === true &&
        assistantMsg.reasoning_content === undefined
      )
        assistantMsg.reasoning_content = "";
      const content = assistantMsg.content;
      const hasContent =
        content !== null && content !== undefined && (typeof content === "string" ? content.length > 0 : content.length > 0);
      if (!hasContent && assistantMsg.tool_calls === undefined) continue;
      out.push(assistantMsg);
      lastRole = "assistant";
      continue;
    }

    if (message.role === "tool") {
      // Consecutive tool-role messages form one result run (pi-ai parity);
      // their images ride after the run in one captioned user message.
      const imageUrls = [];
      let j = i;
      for (; j < messages.length && messages[j].role === "tool"; j += 1) {
        const toolMsg = messages[j];
        const toolImages = [];
        const nestedResults = [];
        const rendered = renderBlocks(toolMsg.content, toolImages, nestedResults);
        const textResult =
          typeof rendered === "string" ? rendered : rendered.filter((part) => part.type === "text").map((part) => part.text).join("\n");
        const hasText = textResult.trim().length > 0;
        // Placeholder parity: dsh-llm-pi-ai's toolResultOf seeds an empty
        // all-text result with "(no output)"; "(see attached image)" is
        // pi-ai's image-only arm (unreachable while images carry handle
        // text, kept because the stock ternary has it).
        out.push({
          role: "tool",
          content: sanitizeSurrogates(hasText ? textResult : toolImages.length > 0 ? "(see attached image)" : "(no output)"),
          tool_call_id: normalizeId(toolMsg.toolCallId),
          ...(compat.requiresToolResultName ? { name: toolNames.get(String(toolMsg.toolCallId)) ?? "unknown" } : {}),
        });
        out.push(...nestedResults);
        imageUrls.push(...toolImages);
      }
      i = j - 1;
      if (imageUrls.length > 0) {
        if (compat.requiresAssistantAfterToolResult) out.push(bridgeAssistant());
        out.push({
          role: "user",
          content: [
            { type: "text", text: "Attached image(s) from tool result:" },
            ...imageUrls.map((url) => ({ type: "image_url", image_url: { url } })),
          ],
        });
        lastRole = "user";
      } else {
        lastRole = "toolResult";
      }
      continue;
    }

    // user message (also the legacy shape carrying tool-result blocks)
    const legacyTools = [];
    const rendered = renderBlocks(message.content, undefined, legacyTools);
    const legacyResultCount = message.content.filter((block) => block.type === "tool-result").length;
    if (typeof rendered === "string" && rendered.length === 0 && legacyResultCount > 0) {
      out.push(...legacyTools);
      lastRole = "toolResult";
    } else {
      if (typeof rendered !== "string" || rendered.length > 0 || legacyResultCount === 0) pushUser(rendered);
      // Text keeps its position ahead of the results riding in the same turn.
      out.push(...legacyTools);
      lastRole = legacyResultCount > 0 ? "toolResult" : "user";
    }
  }
  return out;
}

/** Wire tools for one request (pi-ai convertTools parity). */
export function convertTools(tools, compat) {
  const includeStrict = compat.supportsStrictMode !== false;
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
      ...(includeStrict ? { strict: false } : {}),
    },
  }));
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
 * @param {boolean} thinkingOn - the enable_thinking value to force. In
 *   adaptive mode the intercept resolves it per request before calling this.
 * @param {Array<object>} wireMessages - messages from convertMessages.
 * @param {object} [p] - sampling: {thinking, instruct} preset tables applied
 *   per mode (undefined = stock sampling passthrough, no overrides);
 *   effort: the active reasoning_effort (sent in 'on' mode; in 'off' mode
 *   only when it is not the provider default);
 *   preserveThinking: the user's preserve_thinking choice — sent as
 *   chat_template_kwargs.preserve_thinking while thinking is on, and
 *   forced to false while thinking is off (per the user's convention);
 *   sessionAffinity: send GenerateOptions.sessionId as body.session_id
 *   (default true; false for strict gateways that reject unknown fields);
 *   compat: resolved compat flags for stock body parity (supportsStore,
 *   maxTokensField, requiresToolsWithToolHistory).
 * @returns {object} the JSON request body.
 */
export function buildBody(options, thinkingOn, wireMessages, { sampling, effort, preserveThinking, sessionAffinity = true, compat = {} } = {}) {
  const body = {
    model: options.model,
    messages: wireMessages,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (compat.supportsStore === true) body.store = false;
  // options.stop is deliberately not forwarded: pi-ai's openai-completions
  // route never sends a `stop` field, and wire parity outranks the old
  // plugin's dead passthrough of it.
  if (options.maxTokens !== undefined)
    body[compat.maxTokensField === "max_tokens" ? "max_tokens" : "max_completion_tokens"] = options.maxTokens;
  // Session affinity: dsh threads the session id into GenerateOptions.sessionId,
  // and the local sglang fork's session radix cache (--enable-session-radix-cache)
  // reads `session_id` from the chat-completions BODY — the same field the
  // [PATCH session-id-body] sets on the pi-ai route. Without it here, every
  // intercepted request falls off the session cache and re-prefills the full
  // conversation server-side. Only send a usable id: a non-empty string or a
  // finite number. String(null)/String(false)/String({}) would put a shared
  // bogus key in the body and collide unrelated conversations on one SGLang
  // session — the opposite of the affinity this buys. Strict gateways that
  // 400 on unknown body fields opt out via sessionAffinity: false.
  const sessionId = options.sessionId;
  const usableSessionId =
    (typeof sessionId === "string" && sessionId.length > 0) ||
    (typeof sessionId === "number" && Number.isInteger(sessionId) && sessionId > 0);
  if (sessionAffinity && usableSessionId) body.session_id = String(sessionId);
  if (Array.isArray(options.tools) && options.tools.length > 0) {
    body.tools = convertTools(options.tools, compat);
  } else if (compat.requiresToolsWithToolHistory && wireMessages.some((message) => message.role === "tool")) {
    body.tools = [];
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
