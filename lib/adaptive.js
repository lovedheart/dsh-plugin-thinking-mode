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

/**
 * Scaffold marker prefixes (host-owned: dsh-agent-loop's
 * RuntimeContextProjection snapshot and <system-reminder> wrappers). The
 * harness appends these as role=user turns after every step, and a snapshot
 * can also ride inside the user's own turn after their text.
 */
const SCAFFOLD_PREFIXES = ["<system-reminder>", "Current runtime context."];

/**
 * Index where a scaffold marker starts inside a text block (only at a line
 * start — a marker mid-sentence is the user quoting it), or -1 for none.
 */
function scaffoldStart(text) {
  let start = -1;
  for (const prefix of SCAFFOLD_PREFIXES) {
    let idx = text.indexOf(prefix);
    while (idx !== -1) {
      if (idx === 0 || text[idx - 1] === "\n") {
        if (start === -1 || idx < start) start = idx;
        break;
      }
      idx = text.indexOf(prefix, idx + 1);
    }
  }
  return start;
}

/**
 * The user-authored text of one message: its text blocks joined, with every
 * scaffold region (from a line-start marker to the end of its block) cut.
 * Pure tool-result and scaffold-only messages yield "".
 */
function userTextOf(message) {
  let text = "";
  for (const block of message.content) {
    if (block.type !== "text" || typeof block.text !== "string") continue;
    const cut = scaffoldStart(block.text);
    text += cut === -1 ? block.text : block.text.slice(0, cut);
  }
  return text;
}

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
      // A later *real* user turn consumes the hint: stop here. Scaffold and
      // pure tool-result turns (which the harness appends after every step)
      // do not.
      if (userTextOf(message).trim().length > 0) break;
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
  // Backwards walk for the last *real* user prompt, skipping pure
  // tool-result and scaffold-only turns. userTextOf has already cut any
  // trailing scaffold region inside a mixed turn, so a snapshot appended
  // after the user's text (or after their workspace path) can no longer
  // hijack the heuristics, and a prompt that merely starts with a marker
  // only drops that block, not the whole turn.
  for (let i = options.messages.length - 1; i >= 0; i -= 1) {
    const message = options.messages[i];
    if (message.role !== "user") continue;
    const text = userTextOf(message);
    if (text.trim().length === 0) continue;
    return { text };
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
    const hasText = userTextOf(message).trim().length > 0;
    // Scaffold-only or empty turns (the snapshot rides in as a user text
    // turn after every step) are transparent to the count; without this,
    // the trailing snapshot breaks the run at 0 and the long-tool-run rule
    // can never fire.
    if (!hasResult && !hasText) continue;
    if (hasResult && !hasText) {
      count += 1;
      continue;
    }
    break;
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
