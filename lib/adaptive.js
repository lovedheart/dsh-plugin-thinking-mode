/**
 * Adaptive thinking-mode helpers.
 *
 * In mode 'adaptive' the plugin decides `enable_thinking` per request:
 *   1. explicit per-request signal — the model passes a `thinking` argument on
 *      its own tool calls (thinking / fast); when any tool call in the
 *      conversation asks for thinking, the NEXT generation is forced on;
 *   2. heuristics over the request itself (last user message, tool-call run
 *      length, context size, image content);
 *   3. the adaptive default when nothing fires.
 *
 * The decision is deterministic and logged, and rides to the model as a
 * runtime-context line so it can override it with the `thinking_mode` tool.
 */

/** Per-turn marker values for the `thinking` tool-call argument. */
export const THINKING_HINT_KEYS = ["thinking", "fast"];

/** JSON-schema fragment for the per-turn hint parameter (tool registration). */
export const THINKING_HINT_SCHEMA = {
  type: "string",
  enum: THINKING_HINT_KEYS,
  description:
    "Per-turn thinking override for your NEXT response: 'thinking' forces reasoning on, 'fast' forces it off. Only meaningful while the thinking mode is 'adaptive'; ignored in on/off/auto.",
};

/**
 * Scan the request messages for per-turn `thinking` hints the model set on its
 * own tool calls. The most recent assistant message wins; a hint counts only
 * when no later user turn follows it (the hint applies to the immediately next
 * generation).
 *
 * @param {Array<object>} messages - Harness request messages.
 * @returns {"thinking" | "fast" | undefined}
 */
export function lastThinkingHint(messages) {
  let hint;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role === "user") {
      // A later user turn consumes the hint: stop at the first one below the
      // most recent assistant message we have already inspected.
      if (hint !== undefined) break;
      continue;
    }
    if (message.role !== "assistant") continue;
    for (const block of message.content) {
      if (block.type !== "tool-call" || block.name !== "thinking_mode") continue;
      const parsed = parseArguments(block.arguments);
      if (parsed === undefined) continue;
      if (THINKING_HINT_KEYS.includes(parsed.thinking)) {
        hint = parsed.thinking;
      }
    }
    if (hint !== undefined) break;
  }
  return hint;
}

/** Parse raw tool-call arguments, tolerating malformed model output. */
function parseArguments(raw) {
  if (typeof raw !== "string" || raw.length === 0) return undefined;
  try {
    const parsed = JSON.parse(raw);
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Heuristic decision for one request (no explicit hint present).
 *
 * @param {object} options - GenerateOptions-shaped request.
 * @param {object} rules - resolved adaptive rule thresholds.
 * @param {number} [estimate] - estimated prompt tokens of the request.
 * @returns {{on: boolean, reason: string}}
 */
export function decideByHeuristics(options, rules, estimate) {
  const lastUser = lastUserText(options);
  const text = lastUser.text;
  const trimmed = text.trim();

  if (trimmed.length === 0) return { on: true, reason: "empty-or-nonuser-last-turn" };
  if (/[/\\]/.test(trimmed) || trimmed.startsWith("@")) {
    return { on: false, reason: "path-reference" };
  }
  if (isAcknowledgement(trimmed)) return { on: false, reason: "acknowledgement" };
  if (rules.toolRunThinking && trailingToolResults(options) >= rules.toolRunThinking) {
    return { on: true, reason: "long-tool-run" };
  }
  if (hasImages(options)) return { on: false, reason: "image-content" };
  if (estimate !== undefined && estimate >= rules.largeContextTokens) {
    return { on: false, reason: `large-context(${estimate}>=${rules.largeContextTokens})` };
  }
  if (text.length <= rules.shortMessageChars && !looksComplex(text)) {
    return { on: false, reason: "short-simple-message" };
  }
  if (looksComplex(text)) return { on: true, reason: "complex-keywords" };
  return { on: rules.defaultOn, reason: "adaptive-default" };
}

/** The text of the last user message that is not a pure tool result. */
function lastUserText(options) {
  for (let i = options.messages.length - 1; i >= 0; i -= 1) {
    const message = options.messages[i];
    if (message.role !== "user") continue;
    const results = message.content.filter((block) => block.type === "tool-result");
    const text = message.content
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("");
    if (results.length > 0 && text.trim().length === 0) continue;
    return { text };
  }
  // No plain user turn at all (e.g. a continuation after tool results): use
  // the newest user-ish text available.
  for (let i = options.messages.length - 1; i >= 0; i -= 1) {
    const message = options.messages[i];
    if (message.role !== "user") continue;
    const text = message.content
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("");
    if (text.length > 0) return { text };
  }
  return { text: "" };
}

const ACKNOWLEDGEMENTS = new Set([
  "ok",
  "okay",
  "ok thanks",
  "thanks",
  "thank you",
  "ty",
  "thx",
  "yes",
  "no",
  "yep",
  "nope",
  "sure",
  "go",
  "go on",
  "continue",
  "keep going",
  "done",
  "good",
  "great",
  "nice",
  "cool",
  "hmm",
  "?",
  "??",
  "嗯",
  "嗯嗯",
  "好",
  "好的",
  "好滴",
  "行",
  "行吧",
  "可以",
  "对",
  "对的",
  "是",
  "是的",
  "不",
  "不是",
  "继续",
  "继续吧",
  "谢谢",
  "多谢",
  "收到",
  "明白",
  "懂了",
  "知道了",
  "辛苦了",
]);

function isAcknowledgement(text) {
  const normalized = text.toLowerCase().replace(/[!！。.!~？?、,，\s]+$/g, "").trim();
  return ACKNOWLEDGEMENTS.has(normalized) || ACKNOWLEDGEMENTS.has(text.toLowerCase().trim());
}

/** Count consecutive tool-result-only user turns at the tail of the history. */
function trailingToolResults(options) {
  let count = 0;
  for (let i = options.messages.length - 1; i >= 0; i -= 1) {
    const message = options.messages[i];
    if (message.role !== "user") continue;
    const hasResult = message.content.some((block) => block.type === "tool-result");
    const hasText = message.content
      .filter((block) => block.type === "text")
      .some((block) => block.text.trim().length > 0);
    if (hasResult && !hasText) count += 1;
    else break;
  }
  return count;
}

function hasImages(options) {
  return options.messages.some((message) =>
    message.content.some((block) => block.type === "image"),
  );
}

const COMPLEX_KEYWORDS =
  /(why|how does|how do|debug|fix|error|traceback|exception|refactor|design|architect|optimi[sz]e|prove|derive|analy[sz]e|compare|tradeoff|root cause|implement|算法|证明|推导|分析|设计|重构|排查|定位|根因|为什么|对比|评估|优化)/i;

function looksComplex(text) {
  return COMPLEX_KEYWORDS.test(text);
}

/**
 * Very rough prompt-token estimate (~3.5 characters per token, CJK-heavy
 * prompts overestimate slightly, which is safe for a cutoff). Falls back to a
 * character count of the serialized request when no better source exists.
 */
export function estimatePromptTokens(options) {
  let chars = 0;
  if (typeof options.system === "string") chars += options.system.length;
  for (const message of options.messages) {
    for (const block of message.content) {
      if (typeof block.text === "string") chars += block.text.length;
      else if (block.type === "image") chars += 1000; // rough cost of one image
      else if (typeof block.arguments === "string") chars += block.arguments.length;
    }
  }
  if (Array.isArray(options.tools)) {
    for (const tool of options.tools) {
      chars += (tool.name?.length ?? 0) + (tool.description?.length ?? 0);
      try {
        chars += JSON.stringify(tool.parameters ?? {}).length;
      } catch {
        /* ignore */
      }
    }
  }
  return Math.ceil(chars / 3.5);
}
