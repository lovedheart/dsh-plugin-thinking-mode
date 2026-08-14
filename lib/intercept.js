/**
 * The `llm/stream` waterfall intercept.
 *
 * For every streaming model call the plugin decides whether it routes the
 * request through its own OpenAI-compatible client (mode 'on'/'off' and a
 * matching provider/model) or passes straight through to the resolved
 * adapter (mode 'auto', or anything it does not own). The direct route adds
 * `chat_template_kwargs.enable_thinking` to the request body — the
 * per-request thinking switch documented by the Qwen3.8 model card.
 *
 * The intercept is a documented waterfall short-circuit: the listener
 * yields its own chunks instead of calling next(). Everything it emits is
 * validated by the dsh-llm stream invariant that wraps every stream.
 */
import { attributionHeaders } from "@deepseek-ai/dsh-llm";
import { settingsNamespace } from "@deepseek-ai/dsh-settings";
import { buildBody, convertMessages } from "./messages.js";
import { streamOpenAiSse } from "./stream.js";

/** Read the llm-pi-ai settings section (live), or undefined. */
function readProviderProfiles(ctx) {
  try {
    const section = ctx.settings.get(settingsNamespace("llm-pi-ai"));
    return section?.providers;
  } catch {
    return undefined;
  }
}

/** Case-insensitive substring match of the model id against the patterns. */
function matchesAny(modelId, patterns) {
  if (!Array.isArray(patterns) || patterns.length === 0) return false;
  const id = String(modelId).toLowerCase();
  return patterns.some((pattern) => typeof pattern === "string" && pattern.length > 0 && id.includes(pattern.toLowerCase()));
}

/** Join a profile baseURL with the chat-completions path. */
function chatCompletionsUrl(baseURL) {
  return baseURL.replace(/\/+$/, "") + "/chat/completions";
}

/** Classify an HTTP error status into a stable LlmError code. */
function classifyHttpStatus(status) {
  if (status === 401 || status === 403) return "AUTH";
  if (status === 429) return "RATE_LIMIT";
  if (status === 400) return "INVALID_REQUEST";
  if (status >= 500 && status < 600) return "SERVER";
  return "PROVIDER_HTTP_ERROR";
}

/**
 * Decide the direct-route target for one provider/model, or undefined to
 * pass through to the normal adapter. Exported so the tool can report
 * whether the intercept is active for the current default route.
 */
export function resolveRouteTarget(ctx, config, provider, model, mode) {
  if (mode === "auto") return undefined;
  const profiles = readProviderProfiles(ctx);
  if (profiles === undefined || profiles === null) return undefined;
  const profile = profiles[provider];
  if (profile === undefined || profile === null) return undefined;
  if (profile.api !== "openai-completions") return undefined;
  if (typeof profile.baseURL !== "string" || profile.baseURL.length === 0) return undefined;
  if (!matchesAny(model, config.modelPatterns)) return undefined;
  const rawKey = profile.apiKeyEnv ? process.env[profile.apiKeyEnv] : undefined;
  const apiKey = typeof rawKey === "string" && rawKey.length > 0 ? rawKey : undefined;
  return { baseURL: profile.baseURL, apiKey };
}

/**
 * Install the thinking-mode intercept on the context.
 *
 * @param {import("@deepseek-ai/cordis").Context} ctx - the plugin context.
 * @param {object} p - getMode(), the row config, and the named logger.
 */
export function installThinkingModeIntercept(ctx, { getMode, config, log }) {
  // Lazy attachment-service lookup: the service is only consulted when a
  // user message carries image blocks, and some compositions may not provide
  // it at all. Re-read on each need (cheap, and a service registered later
  // must be honored).
  const getAttachments = () => {
    try {
      const value = ctx.attachments;
      return value === undefined ? undefined : value;
    } catch {
      return undefined;
    }
  };

  const hasImages = (options) =>
    options.messages.some((message) => message.content.some((block) => block.type === "image"));

  const resolveTarget = (provider, model, mode) =>
    resolveRouteTarget(ctx, config, provider, model, mode);

  /** Route one request through the direct OpenAI-compatible client. */
  async function* directStream(options, target, mode, attachments) {
    const thinkingOn = mode === "on";
    let response;
    try {
      const wireMessages = await convertMessages(options, attachments);
      const body = buildBody(options, thinkingOn, wireMessages);
      const headers = { "content-type": "application/json", ...attributionHeaders() };
      if (target.apiKey !== undefined) headers.authorization = `Bearer ${target.apiKey}`;
      response = await fetch(chatCompletionsUrl(target.baseURL), {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        ...(options.signal !== undefined ? { signal: options.signal } : {}),
      });
    } catch (error) {
      const aborted = options.signal?.aborted === true || error?.name === "AbortError";
      yield { type: "usage", usage: { inputTokens: 0, outputTokens: 0 } };
      yield {
        type: "finish",
        reason: aborted
          ? { kind: "aborted", failure: { message: "thinking-mode intercept request aborted", code: "ABORTED" } }
          : {
              kind: "error",
              failure: { message: String(error?.message ?? error), code: "TRANSPORT" },
            },
      };
      return;
    }

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      yield { type: "usage", usage: { inputTokens: 0, outputTokens: 0 } };
      yield {
        type: "finish",
        reason: {
          kind: "error",
          failure: {
            message: `thinking-mode intercept: provider API error (${response.status}): ${text.slice(0, 500) || response.statusText}`,
            code: classifyHttpStatus(response.status),
          },
        },
      };
      return;
    }

    yield* streamOpenAiSse(response, {
      provider: options.provider,
      model: options.model,
      signal: options.signal,
    });
  }

  ctx.on(
    "llm/stream",
    async function* thinkingModeIntercept(options, next) {
      let target;
      try {
        target = resolveTarget(options.provider, options.model, getMode());
      } catch (error) {
        log.warn(`intercept check failed, passing through: ${error?.message ?? error}`);
        target = undefined;
      }
      const needsImages = hasImages(options);
      if (target === undefined || (needsImages && getAttachments() === undefined)) {
        // Default, unmatched, or unhandleable request: stock adapter path.
        yield* await next();
        return;
      }
      try {
        log.info(
          `thinking-mode: routing ${options.provider}/${options.model} directly (mode=${getMode()})`
        );
      } catch {
        // logging must never break the request path
      }
      yield* directStream(options, target, getMode(), getAttachments());
    },
    { global: true },
  );
}
