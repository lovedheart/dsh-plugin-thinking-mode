/**
 * dsh-plugin-thinking-mode
 *
 * Plugin entry: registers the `thinking_mode` tool, the `thinking-mode`
 * settings section, a system-prompt section, and the `llm/stream`
 * waterfall intercept that flips `chat_template_kwargs.enable_thinking`
 * on Qwen-family models for OpenAI-compatible endpoints.
 *
 * Reference: Qwen3.8-27B model card (huggingface.co/Qwen/Qwen3.8-27B) —
 * thinking mode is on by default and can be disabled per request via the
 * Chat Completions API (`chat_template_kwargs: { enable_thinking }`).
 */
import z from "@deepseek-ai/schemastery";
import { defineTool } from "@deepseek-ai/dsh-tools";

import { installThinkingModeIntercept, resolveRouteTarget } from "./intercept.js";
import {
  SAMPLING,
  REASONING_EFFORT_KEYS,
  DEFAULT_REASONING_EFFORT,
  presetJsonSchema,
  presetSchemastery,
} from "./messages.js";

/** Tool-output JSON schema for one mode's sampling preset. */
const PRESET_SCHEMA = presetJsonSchema();

export { apply, name, inject, Config };

/** Cordis plugin name used by loader diagnostics. */
const name = "thinking-mode";

/** Services required before `apply` runs. */
const inject = ["tools", "systemPrompt", "settings"];

/** The three thinking modes. */
const MODES = ["auto", "on", "off"];

/** Human-readable explanation per mode, used by the tool and the prompt. */
const MODE_DESCRIPTIONS = {
  auto: "provider default (thinking is ON by default for Qwen3.8)",
  on: "thinking forced — the model reasons before answering",
  off: "thinking disabled — direct answers, faster and cheaper",
};

/** Plugin configuration schema (defaults live on the fields). */
const Config = z.object({
  /** Mode used when the user has not stored an explicit choice. */
  defaultMode: z.union(MODES).default("auto"),
  /** Case-insensitive substrings matched against the model id. */
  modelPatterns: z.array(z.string()).default(["qwen"]),
  /**
   * Whether the intercept force-applies the model card's recommended
   * sampling per mode (on → thinking preset, off → instruct preset).
   * When false, sampling parameters pass through from the request as-is.
   */
  applySampling: z.boolean().default(true),
  /**
   * Per-mode sampling presets — the model card's recommended sampling
   * (temperature / top_p / presence_penalty differ between modes; top_k,
   * min_p, repetition_penalty are shared). Force-applied per mode by the
   * intercept when applySampling is true.
   */
  sampling: z
    .object({
      thinking: presetSchemastery(z),
      instruct: presetSchemastery(z),
    })
    .default(SAMPLING),
  /**
   * Default reasoning_effort (official Qwen3 levels: xhigh / medium /
   * low). Used when the user has not stored an explicit choice.
   */
  defaultReasoningEffort: z.union(REASONING_EFFORT_KEYS).default(DEFAULT_REASONING_EFFORT),
});

/**
 * Apply the thinking-mode plugin to the given context.
 *
 * @param {import("@deepseek-ai/cordis").Context} ctx - the plugin context.
 * @param {object} config - the validated row configuration.
 */
function apply(ctx, config) {
  const log = ctx.logger("thinking-mode");

  // ── Persisted, live state (settings section) ────────────────────────
  // The mode lives in the `thinking-mode` settings section so it survives
  // restarts, is editable from the Web UI settings page, and applies live
  // (every llm/stream dispatch re-reads it).
  const stateSchema = z.object({
    mode: z.union(MODES).default(config.defaultMode),
    // Same shape as the row config's `sampling` preset; the state is what
    // the intercept reads live, so it is also what the Web UI edits.
    applySampling: z.boolean().default(config.applySampling),
    sampling: z
      .object({
        thinking: presetSchemastery(z),
        instruct: presetSchemastery(z),
      })
      .default(config.sampling),
    /** Active reasoning_effort (xhigh / medium / low). */
    reasoningEffort: z.union(REASONING_EFFORT_KEYS).default(config.defaultReasoningEffort),
    /**
     * Whether prior turns' reasoning content is kept in the context
     * (chat_template_kwargs.preserve_thinking). Only takes effect while
     * thinking is on; while thinking is off it is always sent as false.
     */
    preserveThinking: z.boolean().default(false),
  });
  const scope = ctx.settings.register("thinking-mode", stateSchema, {
    applies: "live",
  });
  const getMode = () => scope.get().mode;
  /** The active per-mode sampling presets (undefined = stock passthrough). */
  const getSampling = () => {
    const s = scope.get();
    return s.applySampling === false ? undefined : { thinking: s.sampling.thinking, instruct: s.sampling.instruct };
  };
  /** The active reasoning_effort. */
  const getEffort = () => scope.get().reasoningEffort;
  /** The active preserve_thinking flag (only meaningful while thinking is on). */
  const getPreserveThinking = () => scope.get().preserveThinking;
  const snapshot = () => {
    const s = scope.get();
    return {
      mode: s.mode,
      description: MODE_DESCRIPTIONS[s.mode] ?? s.mode,
      sampling: {
        applySampling: s.applySampling,
        thinking: s.sampling.thinking,
        instruct: s.sampling.instruct,
      },
      reasoningEffort: s.reasoningEffort,
      preserveThinking: s.preserveThinking,
      interceptActive: defaultRouteInterceptActive(),
    };
  };

  /**
   * Whether the intercept would route the DEFAULT model's requests directly
   * at the current mode (true/false), or undefined when the default route
   * cannot be determined.
   */
  const defaultRouteInterceptActive = () => {
    try {
      const defaultModel = ctx.settings.get("agent-default-model");
      if (!defaultModel || typeof defaultModel.provider !== "string") return undefined;
      return resolveRouteTarget(ctx, config, defaultModel.provider, defaultModel.model, getMode()) !== undefined;
    } catch {
      return undefined;
    }
  };

  // ── thinking_mode tool ──────────────────────────────────────────────
  ctx.tools.register(
    defineTool({
      name: "thinking_mode",
      description:
        "Get or switch the model's thinking (reasoning) mode for Qwen-family models on OpenAI-compatible endpoints; " +
        "changes apply from the next model call. Current settings and conventions are in the thinking_mode section " +
        "of the system prompt. Triggers: 'turn off thinking' / '关闭思考' → off; 'think more' / '打开思考' → on; " +
        "adjust depth via reasoningEffort.",
      parameters: {
        action: {
          type: "string",
          required: true,
          enum: ["get", "on", "off", "auto", "toggle"],
          description: "'get' reports state; on/off/auto set explicitly; toggle flips (from 'auto' → 'off').",
        },
        reasoningEffort: {
          type: "string",
          enum: REASONING_EFFORT_KEYS,
          description:
            "Optional: set reasoning depth in the same call (xhigh thorough / medium balance / low fastest); " +
            "applies from the next call. Omit to leave unchanged.",
        },
        preserveThinking: {
          type: "boolean",
          description:
            "Optional: keep prior turns' reasoning content in context (chat_template_kwargs.preserve_thinking; " +
            "effective only while thinking is on). When enabling thinking, ask the user via ask_user_question first.",
        },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            mode: { type: "string", required: true },
            description: { type: "string", required: true },
            changed: { type: "boolean", required: true },
            sampling: {
              type: "object",
              additionalProperties: false,
              properties: {
                applySampling: { type: "boolean", required: true },
                thinking: PRESET_SCHEMA,
                instruct: PRESET_SCHEMA,
              },
            },
            reasoningEffort: {
              type: "string",
              required: true,
              enum: REASONING_EFFORT_KEYS,
            },
            preserveThinking: {
              type: "boolean",
              required: true,
              description:
                "The active preserve_thinking flag. Only meaningful while thinking is on; while off it is always sent as false.",
            },
            interceptActive: {
              type: "boolean",
              description:
                "Whether the per-request intercept is in effect for the default model route at this mode.",
            },
          },
        },
        render: (_args, value) => {
          const parts = [
            `Thinking mode is now "${value.mode}" — ${value.description}.`,
            value.changed ? " (changed by this call)" : " (unchanged)",
          ];
          if (value.interceptActive === true) {
            const presetName = value.mode === "on" ? "thinking" : "instruct";
            const preset = value.sampling?.[presetName];
            parts.push(
              " The intercept is ACTIVE for the default model route (requests are routed with the matching enable_thinking setting" +
                (preset
                  ? ` and the mode's recommended sampling: temperature=${preset.temperature}, top_p=${preset.top_p}, presence_penalty=${preset.presence_penalty}`
                  : "") +
                (value.mode === "on"
                  ? `; reasoning_effort=${value.reasoningEffort ?? "xhigh"}; preserve_thinking=${value.preserveThinking === true}`
                  : value.reasoningEffort && value.reasoningEffort !== "xhigh"
                    ? `; reasoning_effort=${value.reasoningEffort} (no effect while thinking is off)`
                    : "") +
                (value.mode === "off" ? "; preserve_thinking=false (always, while thinking is off)" : "") +
                ").",
            );
          } else if (value.interceptActive === false) {
            parts.push(
              " The intercept is NOT active for the default model route (provider default applies; check the model/provider against this plugin's modelPatterns).",
            );
          }
          return [{ type: "text", text: parts.join("") }];
        },
      },
      timeoutMs: 15000,
      isConcurrencySafe: () => false,
      async execute({ action, reasoningEffort, preserveThinking }) {
        const before = getMode();
        const beforeEffort = getEffort();
        const beforePreserve = getPreserveThinking();
        let after = before;
        if (action === "on" || action === "off" || action === "auto") {
          if (action !== before) {
            await scope.update({ mode: action });
            after = getMode();
          }
        } else if (action === "toggle") {
          after = before === "off" ? "on" : "off";
          await scope.update({ mode: after });
        }
        // Optionally change the reasoning depth in the same call. `scope.update`
        // merges, so a partial `{ reasoningEffort }` patch is safe.
        if (reasoningEffort !== undefined && reasoningEffort !== beforeEffort) {
          await scope.update({ reasoningEffort });
        }
        // Optionally set preserve_thinking in the same call (typically after
        // asking the user when enabling thinking; no-op while thinking is off).
        if (preserveThinking !== undefined && preserveThinking !== beforePreserve) {
          await scope.update({ preserveThinking });
        }
        const snap = snapshot();
        return {
          mode: snap.mode,
          description: snap.description,
          changed:
            snap.mode !== before || snap.reasoningEffort !== beforeEffort || snap.preserveThinking !== beforePreserve,
          sampling: snap.sampling,
          reasoningEffort: snap.reasoningEffort,
          preserveThinking: snap.preserveThinking,
          interceptActive: snap.interceptActive,
        };
      },
      presentCall: (args) => ({
        card: "generic",
        title: `thinking_mode: ${args.action}`,
        kind: "status",
      }),
    }),
  );

  // ── Runtime-context state (re-rendered on every assembly) ──────────
  // Registered as a dynamic context, not a system-prompt section: the live
  // state changes whenever the user flips the mode, and a changed system
  // prompt rewrites the cached head of every conversation. Contexts flow
  // through the append-only runtime-context snapshot instead (same pattern
  // as approval:policy), so a flip costs a tail message, not a re-prefill.
  ctx.systemPrompt.context({
    name: "thinking-mode:state",
    order: 125,
    text: () => {
      const mode = getMode();
      const effort = getEffort();
      const preserve = getPreserveThinking();
      return (
        `The \`thinking_mode\` tool switches the underlying model's thinking (reasoning) mode for Qwen-family ` +
        `models on OpenAI-compatible endpoints; changes apply from the next model call onward. ` +
        `Current mode: ${mode} — ${MODE_DESCRIPTIONS[mode]}. ` +
        `Current reasoning_effort: ${effort} (xhigh thorough / medium balance / low fastest). ` +
        `Current preserve_thinking: ${preserve} (keeps prior turns' reasoning content in context; effective only ` +
        `while thinking is on). ` +
        `When the user wants to ENABLE thinking, first ask via ask_user_question whether to preserve the reasoning ` +
        `content, then pass their answer as preserveThinking in the same call; when disabling, preserveThinking is ` +
        `automatically false — no need to ask.`
      );
    },
  });

  // ── llm/stream intercept ────────────────────────────────────────────
  installThinkingModeIntercept(ctx, {
    getMode,
    getSampling,
    getEffort,
    getPreserveThinking,
    config,
    log,
  });
}
