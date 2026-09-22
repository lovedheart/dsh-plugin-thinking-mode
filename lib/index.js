/**
 * dsh-plugin-thinking-mode
 *
 * Plugin entry: registers the `thinking_mode` tool, a dynamic system-prompt
 * context, and the `llm/stream` waterfall intercept that flips
 * `chat_template_kwargs.enable_thinking` on Qwen-family models for
 * OpenAI-compatible endpoints.
 *
 * The switch state *is* this plugin's own profile configuration: the live
 * fields are declared `volatile()` below, so a change made by the tool (or by
 * the Web settings page) is written back into the profile patch, committed
 * into the running instance's config references, and takes effect on the next
 * model call — no restart, no separate state store.
 *
 * Reference: Qwen3.8-27B model card (huggingface.co/Qwen/Qwen3.8-27B) —
 * thinking mode is on by default and can be disabled per request via the
 * Chat Completions API (`chat_template_kwargs: { enable_thinking }`).
 */
import z from "@deepseek-ai/schemastery";
import { defineTool } from "@deepseek-ai/dsh-tools";

import { installThinkingModeIntercept, live, resolveRouteTarget } from "./intercept.js";
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

/**
 * Plugin configuration. The fields carrying the switch state are declared
 * `volatile()`: the Loader treats a change to them as a live update — it
 * commits the new values into this instance's config references instead of
 * remounting the plugin — and the Settings service exposes them as the form
 * the Web UI edits. `persist()` below is the write-back path.
 */
const Config = z.object({
  /** Mode used when the user has not stored an explicit choice. */
  defaultMode: z.union(MODES).default("auto"),
  /**
   * Stored mode choice (the live state). Absent means "no explicit choice",
   * which `defaultMode` answers.
   */
  mode: z.union(MODES).volatile(),
  /** Case-insensitive substrings matched against the model id. */
  modelPatterns: z.array(z.string()).default(["qwen"]),
  /**
   * Whether the intercept force-applies the model card's recommended
   * sampling per mode (on → thinking preset, off → instruct preset).
   * When false, sampling parameters pass through from the request as-is.
   */
  applySampling: z.boolean().volatile().default(true),
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
    .volatile()
    .default(SAMPLING),
  /**
   * reasoning_effort used when the user has not stored an explicit choice
   * (official Qwen3 levels: xhigh / medium / low).
   */
  defaultReasoningEffort: z.union(REASONING_EFFORT_KEYS).default(DEFAULT_REASONING_EFFORT),
  /** Stored reasoning depth (the live state). */
  reasoningEffort: z.union(REASONING_EFFORT_KEYS).volatile(),
  /**
   * Whether prior turns' reasoning content is kept in the context
   * (chat_template_kwargs.preserve_thinking; the live state). Only takes
   * effect while thinking is on; while thinking is off it is always sent
   * as false.
   */
  preserveThinking: z.boolean().volatile().default(false),
});

/**
 * Apply the thinking-mode plugin to the given context.
 *
 * @param {import("@deepseek-ai/cordis").Context} ctx - the plugin context.
 * @param {object} config - the validated row configuration.
 */
function apply(ctx, config) {
  const log = ctx.logger("thinking-mode");

  // ── Live state: this plugin's own volatile Config fields ────────────
  // The switch state is configuration, not plugin-local memory: it survives
  // restarts, is edited by the Web UI settings page, and every llm/stream
  // dispatch re-reads it. Volatile fields arrive as live references, so an
  // update — from the tool, the settings page, or an edited profile — is
  // visible here without a remount.
  // A write that the profile refuses (e.g. this entry is mounted in a way the
  // configuration editor cannot round-trip) must not leave the session stuck on
  // the old values: the fields it rejected are held here for the rest of the
  // process, so the switch still takes effect immediately, and any write that
  // does land clears them. Nothing else reads or writes this map — the profile
  // stays the single source of truth whenever it can be.
  const pending = {};
  /** The last write failure, for the message `reject()` reports. */
  let persistError;
  /** The effective value of a state field: session override, else the Config. */
  const stateValue = (field) => (field in pending ? pending[field] : live(config[field]));

  const getMode = () => stateValue("mode") ?? config.defaultMode;
  /** The active per-mode sampling presets (undefined = stock passthrough). */
  const getSampling = () => {
    if (stateValue("applySampling") === false) return undefined;
    const presets = stateValue("sampling") ?? SAMPLING;
    return { thinking: presets.thinking, instruct: presets.instruct };
  };
  /** The active reasoning_effort. */
  const getEffort = () => stateValue("reasoningEffort") ?? config.defaultReasoningEffort;
  /** The active preserve_thinking flag (only meaningful while thinking is on). */
  const getPreserveThinking = () => stateValue("preserveThinking") === true;
  const snapshot = () => {
    const presets = stateValue("sampling") ?? SAMPLING;
    const mode = getMode();
    return {
      mode,
      description: MODE_DESCRIPTIONS[mode] ?? mode,
      sampling: {
        applySampling: stateValue("applySampling") !== false,
        thinking: presets.thinking,
        instruct: presets.instruct,
      },
      reasoningEffort: getEffort(),
      preserveThinking: getPreserveThinking(),
      // Whether everything in effect is backed by the profile (no session-only
      // override is in play).
      persisted: Object.keys(pending).length === 0,
      interceptActive: defaultRouteInterceptActive(),
    };
  };

  // ── Write-back path ─────────────────────────────────────────────────
  // `ns` is the profile entry id this instance's Config is stored under:
  // settings writes are addressed by it. `describe()` first, so the write
  // carries the revision it was derived from and silently clobbering a
  // concurrent edit is refused (`SETTINGS_CONFLICT`).
  const ns = ctx.fiber?.entry?.options?.id ?? name;

  /**
   * Merge a patch into this entry's configuration and let the Loader commit
   * it live. An ordinary (non-volatile) field is never in `patch`, so the
   * write is a volatile-only update: no dispose, no re-`apply`.
   *
   * @returns whether the new values are the ones now in effect.
   */
  async function persist(patch) {
    if (await tryWrite(patch)) {
      for (const field of Object.keys(patch)) delete pending[field];
      await flushPending();
      return true;
    }
    return reject(patch);
  }

  /** One `settings.update`, asserting the revision we read unless retrying. */
  async function tryWrite(patch, unconditionally = false) {
    let revision;
    if (!unconditionally) {
      try {
        revision = ctx.settings.describe().find((row) => row.ns === ns)?.revision;
      } catch (error) {
        // Without a descriptor there is no revision to assert; write anyway.
        log.debug(`settings describe failed for ${ns}: ${error?.message ?? error}`);
      }
    }
    try {
      await ctx.settings.update(ns, patch, revision);
      return true;
    } catch (error) {
      if (error?.code !== "SETTINGS_CONFLICT") {
        persistError = error;
        return false;
      }
      if (unconditionally) {
        persistError = error;
        return false;
      }
      // Someone else moved the revision along: retry once without the assert.
      return tryWrite(patch, true);
    }
  }

  /** Re-offer whatever an earlier rejected write left behind, once. */
  async function flushPending() {
    const leftovers = { ...pending };
    if (Object.keys(leftovers).length === 0) return;
    if (await tryWrite(leftovers, true)) {
      for (const field of Object.keys(leftovers)) delete pending[field];
    }
  }

  /**
   * A write the profile rejected: keep the values for this process so the
   * switch is still live, and say so with the reason.
   */
  function reject(patch, error) {
    Object.assign(pending, patch);
    log.warn(
      `could not persist ${JSON.stringify(patch)} to "${ns}": ${(error ?? persistError)?.message ?? "write rejected"}` +
        ` — held for this session only, a restart will revert it`,
    );
    return false;
  }

  /**
   * Whether the intercept would route the DEFAULT model's requests directly
   * at the current mode (true/false), or undefined when the default route
   * cannot be determined.
   */
  const defaultRouteInterceptActive = () => {
    try {
      const selection = ctx.get("agentDefaultModel", false)?.currentSelection?.();
      if (!selection || typeof selection.provider !== "string") return undefined;
      return resolveRouteTarget(ctx, config, selection.provider, selection.model, getMode()) !== undefined;
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
            persisted: {
              type: "boolean",
              required: true,
              description:
                "Whether the state is backed by the profile; false means a rejected write is held for this session only.",
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
          if (value.persisted === false) {
            parts.push(" NOTE: this profile rejected the write, so the change is held for this session only and a restart reverts it.");
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
          after = action;
        } else if (action === "toggle") {
          after = before === "off" ? "on" : "off";
        }
        // One merged write for everything this call changes; `update` merges,
        // so a partial patch leaves the other fields where they are.
        const patch = {};
        if (after !== before) patch.mode = after;
        if (reasoningEffort !== undefined && reasoningEffort !== beforeEffort) patch.reasoningEffort = reasoningEffort;
        if (preserveThinking !== undefined && preserveThinking !== beforePreserve) {
          patch.preserveThinking = preserveThinking;
        }
        if (Object.keys(patch).length > 0) await persist(patch);
        const snap = snapshot();
        return {
          mode: snap.mode,
          description: snap.description,
          changed:
            snap.mode !== before || snap.reasoningEffort !== beforeEffort || snap.preserveThinking !== beforePreserve,
          sampling: snap.sampling,
          reasoningEffort: snap.reasoningEffort,
          preserveThinking: snap.preserveThinking,
          persisted: snap.persisted,
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
