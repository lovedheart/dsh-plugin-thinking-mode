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
import { settingsNamespace } from "@deepseek-ai/dsh-settings";
import { installThinkingModeIntercept, resolveRouteTarget } from "./intercept.js";

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
  });
  const scope = ctx.settings.register(settingsNamespace("thinking-mode"), stateSchema, {
    applies: "live",
  });
  const getMode = () => scope.get().mode;

  /**
   * Whether the intercept would route the DEFAULT model's requests directly
   * at the current mode (true/false), or undefined when the default route
   * cannot be determined.
   */
  const defaultRouteInterceptActive = () => {
    try {
      const defaultModel = ctx.settings.get(settingsNamespace("agent-default-model"));
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
        "Get or switch the model's thinking (reasoning) mode for Qwen-family models on OpenAI-compatible endpoints. " +
        "Modes: 'on' (the model thinks before answering, producing reasoning content), " +
        "'off' (direct answers without thinking — faster and cheaper), " +
        "'auto' (provider default — thinking is ON by default for Qwen3.8). " +
        "Use when the user asks to enable or disable thinking, e.g. '关闭思考模式' / 'turn off thinking' → 'off', " +
        "'打开深度思考' / 'think more' → 'on'.",
      parameters: {
        action: {
          type: "string",
          required: true,
          enum: ["get", "on", "off", "auto", "toggle"],
          description:
            "'get' reports the current mode; 'on'/'off'/'auto' set it explicitly; " +
            "'toggle' flips between on and off (from 'auto' it goes to 'off').",
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
            interceptActive: {
              type: "boolean",
              description:
                "Whether the per-request intercept is in effect for the default model route at this mode.",
            },
          },
        },
        render: (_args, value) => [
          {
            type: "text",
            text:
              `Thinking mode is now "${value.mode}" — ${value.description}.` +
              (value.changed ? " (changed by this call)" : " (unchanged)") +
              (value.interceptActive === true
                ? " The intercept is ACTIVE for the default model route (requests are routed with the matching enable_thinking setting)."
                : value.interceptActive === false
                  ? " The intercept is NOT active for the default model route (provider default applies; check the model/provider against this plugin's modelPatterns)."
                  : ""),
          },
        ],
      },
      timeoutMs: 15000,
      isConcurrencySafe: () => false,
      async execute({ action }) {
        const before = getMode();
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
        return {
          mode: after,
          description: MODE_DESCRIPTIONS[after] ?? after,
          changed: after !== before,
          interceptActive: defaultRouteInterceptActive(),
        };
      },
      presentCall: (args) => ({
        card: "generic",
        title: `thinking_mode: ${args.action}`,
        kind: "status",
      }),
    }),
  );

  // ── System-prompt guidance (re-rendered on every assembly) ─────────
  ctx.systemPrompt.section({
    name: "tool:thinking_mode",
    order: 130,
    text: () => {
      const mode = getMode();
      return (
        `The \`thinking_mode\` tool switches the underlying model's thinking (reasoning) mode for Qwen-family ` +
        `models on OpenAI-compatible endpoints; the change applies to model calls from the next turn onward. ` +
        `Current mode: ${mode} — ${MODE_DESCRIPTIONS[mode]}. ` +
        `Actions: "get" (report state), "on" (force thinking), "off" (direct answers without thinking), ` +
        `"auto" (provider default), "toggle" (flip on/off, from auto goes to off). ` +
        `Call it when the user asks to switch thinking mode, for example "关闭思考" / "stop thinking" → "off", ` +
        `"打开思考" / "think before answering" → "on".`
      );
    },
  });

  // ── llm/stream intercept ────────────────────────────────────────────
  installThinkingModeIntercept(ctx, {
    getMode,
    config,
    log,
  });
}
