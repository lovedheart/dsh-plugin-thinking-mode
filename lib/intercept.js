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

import { decideByHeuristics, estimatePromptTokens, lastThinkingHint } from "./adaptive.js";
import { buildBody, convertMessages } from "./messages.js";
import {
  DEFAULT_REQUEST_IMAGE_MAX_BYTES,
  DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET,
  UnsupportedHistory,
  resolveCompletionsCompat,
} from "./parity.js";
import { streamOpenAiSse } from "./stream.js";

/**
 * Read a Config value that the plugin declared `volatile()`. Such fields are
 * delivered as live references (so a profile edit is visible without a
 * remount) rather than plain values; a composition that hands the plugin a
 * plain object still works.
 */
export function live(value) {
  return value !== null && typeof value === "object" && typeof value.get === "function" ? value.get() : value;
}

/**
 * Read another profile entry's live Config, or undefined when this
 * configuration has no editor or no such entry. Configuration is no longer a
 * readable settings registry: each plugin's live configuration is its entry's
 * Config, addressed through the editor.
 */
function entryConfig(ctx, id) {
  try {
    const entry = ctx
      .get("configEditor", false)
      ?.entries()
      .find((row) => row.options?.id === id);
    const config = entry?.fiber?.config;
    return config !== null && typeof config === "object" ? config : undefined;
  } catch {
    return undefined;
  }
}

/** The configured provider routes (the llm-pi-ai entry's Config), or undefined. */
function readProviderProfiles(ctx) {
  const profiles = live(entryConfig(ctx, "llm-pi-ai")?.providers);
  return profiles !== null && typeof profiles === "object" ? profiles : undefined;
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
  const modelEntry = Array.isArray(profile.models) ? profile.models.find((entry) => entry?.id === model) : undefined;
  return {
    baseURL: profile.baseURL,
    apiKey,
    // Resolved the way the stock route resolves it: pi-ai detection from
    // provider/URL, then the profile's explicit compat overrides.
    compat: resolveCompletionsCompat(profile, provider, profile.baseURL),
    modelInput: Array.isArray(modelEntry?.input) ? modelEntry.input : undefined,
    modelReasoning: modelEntry?.reasoning === true,
    requestImagePolicy: {
      // per-image normalization target passed to readImageRequest (stock
      // profile fields requestImagePixelBudget / requestImageMaxBytes)…
      maxPixels: profile.requestImagePixelBudget ?? DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET,
      maxBytes: profile.requestImageMaxBytes ?? DEFAULT_REQUEST_IMAGE_MAX_BYTES,
    },
    // …vs the request-WIDE budget checked by requiredImageOffload (stock
    // defaults: 20 MiB / 50 images). Our proxy is conservative: exceeding it
    // hands the request to the stock route (which offloads oldest-first)
    // instead of offloading here.
    maxRequestImageBytes: profile.maxRequestImageBytes ?? 20971520,
    maxRequestImages: profile.maxRequestImages ?? 50,
  };
}

/**
 * Install the thinking-mode intercept on the context.
 *
 * @param {import("@deepseek-ai/cordis").Context} ctx - the plugin context.
 * @param {object} p - getMode(), getSampling() (the live per-mode presets,
 *   undefined when sampling pass-through is requested), getEffort() (the
 *   live reasoning_effort), the row config, and the named logger.
 */
export function installThinkingModeIntercept(ctx, { getMode, getSampling, getEffort, getPreserveThinking, getSessionAffinity, getAdaptiveRules, config, log }) {
  // The most recent adaptive decision, for the runtime-context line. Written
  // only by the intercept, read by the tool/prompt; best-effort telemetry.
  let lastAdaptiveDecision;

  /** Resolve the adaptive rules with defaults applied (live each request). */
  const adaptiveRules = () => {
    const rules = getAdaptiveRules?.() ?? {};
    return {
      defaultOn: rules.defaultOn !== false,
      shortMessageChars: Number.isFinite(rules.shortMessageChars) ? rules.shortMessageChars : 40,
      toolRunThinking: Number.isFinite(rules.toolRunThinking) ? rules.toolRunThinking : 6,
      largeContextTokens: Number.isFinite(rules.largeContextTokens) ? rules.largeContextTokens : 24000,
    };
  };

  /** Decide enable_thinking for one request in adaptive mode. */
  const decideAdaptive = (options) => {
    const hint = lastThinkingHint(options.messages);
    if (hint === "thinking") return { on: true, reason: "model-hint:thinking" };
    if (hint === "fast") return { on: false, reason: "model-hint:fast" };
    return decideByHeuristics(options, adaptiveRules(), estimatePromptTokens(options));
  };

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

  // Same gate as the converter's: only non-offloaded images need the
  // attachment service (offloaded ones serialize to stable text handles).
  const hasImages = (options) =>
    options.messages.some((message) =>
      message.content.some((block) => block.type === "image" && block.offloaded !== true),
    );

  const resolveTarget = (provider, model, mode) =>
    resolveRouteTarget(ctx, config, provider, model, mode);

  /**
   * Read one volatile config value. A config getter must never break the
   * request path, but a persistently throwing getter should not be invisible
   * either: warn once per (field, error) instead of degrading silently every
   * request.
   */
  const warnedGetters = new Set();
  const readConfig = (read, fallback, what) => {
    try {
      const value = read?.();
      return value === undefined ? fallback : value;
    } catch (error) {
      const key = `${what}:${error?.message ?? error}`;
      if (!warnedGetters.has(key)) {
        warnedGetters.add(key);
        log?.warn?.(`thinking-mode: ${what} config read failed (${error?.message ?? error}); using ${JSON.stringify(fallback)}`);
      }
      return fallback;
    }
  };

  /**
   * Convert and assemble one direct-route request. Throws UnsupportedHistory
   * when the history needs the stock path; anything else (attachment reads,
   * serialization) is surfaced to the caller as an error to report.
   */
  async function prepareDirect(options, target, thinkingOn, attachments) {
    const wireMessages = await convertMessages(options, attachments, {
      compat: target.compat,
      modelInput: target.modelInput,
      modelReasoning: target.modelReasoning,
      requestImagePolicy: target.requestImagePolicy,
      maxRequestImageBytes: target.maxRequestImageBytes,
      maxRequestImages: target.maxRequestImages,
    });
    const sampling = readConfig(getSampling, undefined, "sampling");
    const effort = readConfig(getEffort, undefined, "reasoningEffort");
    const preserveThinking = readConfig(getPreserveThinking, false, "preserveThinking");
    const sessionAffinity = readConfig(getSessionAffinity, true, "sessionAffinity");
    const body = buildBody(options, thinkingOn, wireMessages, {
      sampling,
      effort,
      preserveThinking,
      sessionAffinity,
      compat: target.compat,
    });
    const headers = { "content-type": "application/json", ...attributionHeaders() };
    if (target.apiKey !== undefined) headers.authorization = `Bearer ${target.apiKey}`;
    return { url: chatCompletionsUrl(target.baseURL), headers, body, thinkingOn };
  }

  /** Send one prepared direct request and stream the SSE response. */
  async function* runDirect(prepared, options) {
    let response;
    try {
      response = await fetch(prepared.url, {
        method: "POST",
        headers: prepared.headers,
        body: JSON.stringify(prepared.body),
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
      const mode = getMode();
      let target;
      try {
        target = resolveTarget(options.provider, options.model, mode);
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
      let decision;
      if (mode === "adaptive") {
        try {
          decision = decideAdaptive(options);
        } catch (error) {
          log.warn(`adaptive decision failed, using default: ${error?.message ?? error}`);
          decision = { on: true, reason: "decision-error" };
        }
        lastAdaptiveDecision = decision;
      }
      const thinkingOn = mode === "adaptive" ? decision.on : mode === "on";
      let prepared;
      try {
        prepared = await prepareDirect(options, target, thinkingOn, getAttachments());
      } catch (error) {
        if (error instanceof UnsupportedHistory) {
          // The stock adapter owns this history shape (extra tool definitions,
          // over-budget images, pass-through content, …): hand the call back.
          // warn, not info: the host has no console sink for info-level
          // ctx.logger output, so an info here is invisible in the journal and
          // a silently-degraded session looks identical to a dead intercept.
          log.warn(`thinking-mode: passing through to stock adapter (${error?.message ?? error})`);
          yield* await next();
          return;
        }
        log.warn(`thinking-mode: direct request preparation failed: ${error?.message ?? error}`);
        const aborted = options.signal?.aborted === true || error?.name === "AbortError";
        yield { type: "usage", usage: { inputTokens: 0, outputTokens: 0 } };
        yield {
          type: "finish",
          reason: {
            kind: aborted ? "aborted" : "error",
            failure: {
              message: String(error?.message ?? error),
              // No HTTP request was attempted: this is a conversion/attachment
              // failure, not a transport one.
              code: aborted ? "ABORTED" : "INVALID_REQUEST",
            },
          },
        };
        return;
      }
      try {
        log.info(
          `thinking-mode: routing ${options.provider}/${options.model} directly (mode=${mode}, enable_thinking=${prepared.thinkingOn}${
            decision === undefined ? "" : `, reason=${decision.reason}`
          })`,
        );
      } catch {
        // logging must never break the request path
      }
      yield* runDirect(prepared, options);
    },
    { global: true },
  );

  // Return the latest adaptive decision so the tool and the runtime-context
  // line can report what the plugin decided on the last request.
  return () => lastAdaptiveDecision;
}
