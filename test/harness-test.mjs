/**
 * Standalone integration test for dsh-plugin-thinking-mode.
 *
 * Boots a minimal cordis Context with stub tools/systemPrompt/settings/
 * configEditor services, applies the plugin, and exercises:
 *   - the Config schema defaults (live fields as volatile references)
 *   - the thinking_mode tool transitions (get/on/off/auto/toggle), whose
 *     changes are written back through the settings service
 *   - the dynamic system-prompt context
 *   - the llm/stream waterfall: direct route (on/off) against a mock
 *     OpenAI-compatible SSE server, passthrough (auto, unmatched model,
 *     image without attachment service), HTTP error and empty-response
 *     terminal finishes, and stream-grammar validity (invariant rules).
 *
 * Run: node test/harness-test.mjs
 */
import assert from "node:assert/strict";
import http from "node:http";
import * as plugin from "../lib/index.js";
import { buildReplayState } from "../lib/stream.js";
import {
  normalizeToolCallId,
  offloadedImageText,
  readPiAiReplayState,
  requestImageHandleText,
  resolveCompletionsCompat,
} from "../lib/parity.js";
import { Context } from "@deepseek-ai/cordis";
import { updateVolatile, volatileEntries } from "@deepseek-ai/cosmokit";

// ── canned SSE payloads ────────────────────────────────────────────────
const SSE_TOOL = [
  'data: {"id":"x1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me "}}]}',
  'data: {"id":"x1","choices":[{"index":0,"delta":{"reasoning_content":"think."}}]}',
  'data: {"id":"x1","choices":[{"index":0,"delta":{"content":"Sure"}}]}',
  'data: {"id":"x1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}}]}',
  'data: {"id":"x1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ci"}}]}}]}',
  'data: {"id":"x1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\":\\"Berlin\\"}"}}]}}]}',
  'data: {"id":"x1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
  'data: {"id":"x1","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"completion_tokens_details":{"reasoning_tokens":8},"prompt_tokens_details":{"cached_tokens":60}}}',
  "data: [DONE]",
].join("\n\n") + "\n\n";

const SSE_TEXT = [
  'data: {"id":"x2","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Hmm."}}]}',
  'data: {"id":"x2","choices":[{"index":0,"delta":{"content":"OK"}}]}',
  'data: {"id":"x2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  'data: {"id":"x2","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4}}',
  "data: [DONE]",
].join("\n\n") + "\n\n";

const SSE_EMPTY = [
  'data: {"id":"x3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  'data: {"id":"x3","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":0}}',
  "data: [DONE]",
].join("\n\n") + "\n\n";

// ── mock OpenAI-compatible server ──────────────────────────────────────
const serverState = { sse: SSE_TOOL, status: 200 };
const requests = [];
const server = http.createServer((req, res) => {
  let raw = "";
  req.on("data", (chunk) => (raw += chunk));
  req.on("end", () => {
    requests.push({
      url: req.url,
      method: req.method,
      headers: req.headers,
      body: JSON.parse(raw),
    });
    if (serverState.status !== 200) {
      res.writeHead(serverState.status, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "boom", type: "server_error" } }));
      return;
    }
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end(serverState.sse);
  });
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const baseURL = `http://127.0.0.1:${server.address().port}/v1`;

// ── the profile entry under test ───────────────────────────────────────
// The plugin's state IS its entry configuration: raw values (what a profile
// patch would store) plus the live volatile references the Loader hands the
// running instance. A settings write re-validates the raw values and commits
// them into those same references — cordis-plugin-loader's volatile-only path,
// reproduced here so the plugin under test is exercised exactly as the host
// would exercise it.
const ENTRY_ID = "thinking-mode";
const entryRaw = { defaultMode: "adaptive", modelPatterns: ["qwen"] };
const entryConfig = plugin.Config["~standard"].validate(entryRaw).value;

let revision = 0;
let failWrites = false;
const writes = [];
const violations = [];

function commitVolatile() {
  const candidate = plugin.Config["~standard"].validate(entryRaw).value;
  for (const { path, ref } of volatileEntries(entryConfig)) {
    updateVolatile(ref, path.reduce((value, key) => Reflect.get(value, key), candidate));
  }
}

// ── stub services on a root cordis context ─────────────────────────────
const ctx = new Context();

const tools = [];
ctx.provide("tools", { register: (tool) => tools.push(tool) });

const sections = [];
const contexts = [];
ctx.provide("systemPrompt", {
  section: (section) => sections.push(section),
  context: (context) => contexts.push(context),
});

// The provider routes live in the llm-pi-ai entry's Config.
const piConfig = {
  providers: {
    sglang: {
      displayName: "SGLang",
      api: "openai-completions",
      baseURL,
      apiKeyEnv: "TEST_THINKING_KEY",
      models: [{ id: "Qwen3.8-27B", input: ["text", "image"] }],
    },
    // Same local server, but profile compat overrides on every axis the
    // body assertions read (case 12 pins these non-default branches).
    overrides: {
      displayName: "Compat overrides",
      api: "openai-completions",
      baseURL,
      apiKeyEnv: "TEST_THINKING_KEY",
      compat: { supportsStore: false, maxTokensField: "max_tokens", supportsStrictMode: true },
      models: [{ id: "Qwen3.8-27B", input: ["text"] }],
    },
    // Flags the wire conversion reads, all on at once (case 14).
    fussy: {
      displayName: "Fussy route",
      api: "openai-completions",
      baseURL,
      apiKeyEnv: "TEST_THINKING_KEY",
      compat: {
        supportsDeveloperRole: true,
        requiresToolResultName: true,
        requiresAssistantAfterToolResult: true,
        requiresNonNullAssistantContent: true,
        requiresReasoningContentOnAssistantMessages: true,
        requiresThinkingAsText: true,
        requiresToolsWithToolHistory: true,
      },
      models: [{ id: "Qwen3.8-27B", input: ["text"], reasoning: true }],
    },
    // 1024-pixel budget forces the 64×64 test image down to a 32×32 request
    // preview, pinning the requestImagePixelBudget → readImageRequest target
    // scaling path end to end.
    budgeted: {
      displayName: "Pixel budget",
      api: "openai-completions",
      baseURL,
      apiKeyEnv: "TEST_THINKING_KEY",
      requestImagePixelBudget: 1024,
      models: [{ id: "Qwen3.8-27B", input: ["text", "image"] }],
    },
  },
};
ctx.provide("configEditor", {
  entries: () => [{ options: { id: "llm-pi-ai" }, fiber: { config: piConfig } }],
});

// The default model route, as the plugin reads it through the service.
ctx.provide("agentDefaultModel", {
  currentSelection: () => ({ provider: "sglang", model: "Qwen3.8-27B" }),
});

ctx.provide("settings", {
  describe: () => [{ ns: ENTRY_ID, revision, applies: "live" }],
  update: async (ns, patch, expectedRevision) => {
    if (failWrites) throw new Error("profile patch is not writable");
    writes.push({ ns, patch, expectedRevision, revision });
    for (const key of Object.keys(patch)) {
      if (plugin.Config.dict[key]?.meta?.volatile !== true) violations.push(`non-volatile field written: ${key}`);
    }
    Object.assign(entryRaw, patch);
    commitVolatile();
    revision += 1;
  },
});

process.env.TEST_THINKING_KEY = "secret-key-123";

/** Write through the settings service the way the host's UI does. */
async function setConfig(patch) {
  await ctx.settings.update(ENTRY_ID, patch, revision);
}

// ── apply the plugin ───────────────────────────────────────────────────
assert.equal(plugin.name, "thinking-mode");
assert.deepEqual(plugin.inject, ["tools", "systemPrompt", "settings"]);

// Config schema (Standard Schema v1 interface: `~standard.validate`). The
// state fields are volatile references; the rest are ordinary configuration.
const parsed = plugin.Config["~standard"].validate({}).value;
assert.equal(parsed.defaultMode, "auto");
assert.deepEqual(parsed.adaptive.get(), {
  defaultOn: true,
  shortMessageChars: 40,
  toolRunThinking: 6,
  largeContextTokens: 24000,
});
assert.equal(parsed.mode.get(), undefined, "no stored mode until the user chooses one");
assert.deepEqual(parsed.modelPatterns, ["qwen"]);
assert.equal(parsed.applySampling.get(), true);
assert.deepEqual(parsed.sampling.get().thinking, {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 0.0,
  repetition_penalty: 1.0,
});
assert.deepEqual(parsed.sampling.get().instruct, {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 1.5,
  repetition_penalty: 1.0,
});
assert.equal(parsed.defaultReasoningEffort, "xhigh");
assert.equal(parsed.reasoningEffort.get(), undefined, "no stored effort until the user chooses one");
assert.equal(parsed.preserveThinking.get(), false);
assert.equal(parsed.sessionAffinity.get(), true, "session_id rides by default");
assert.equal(parsed.stableAdaptiveContext.get(), true, "adaptive context is byte-stable by default");

plugin.apply(ctx, entryConfig);

assert.equal(tools.length, 1, "one tool registered");
assert.equal(tools[0].name, "thinking_mode");
assert.equal(sections.length, 0, "the live state is not a system-prompt section");
assert.equal(contexts.length, 1, "one dynamic context registered");
assert.equal(contexts[0].name, "thinking-mode:state");

// ── tool transitions ───────────────────────────────────────────────────
const tool = tools[0];

let out = await tool.execute({ action: "get" });
assert.equal(out.mode, "adaptive", "entry defaultMode drives the initial mode");
assert.equal(out.changed, false);
assert.equal(out.interceptActive, true, "intercept active for the adaptive default route");
assert.equal(out.adaptiveDecision, undefined, "no decision key before the first model call");
assert.equal(out.sampling.applySampling, true);
assert.equal(out.sampling.thinking.temperature, 1.0);
assert.equal(out.sampling.instruct.presence_penalty, 1.5);
assert.equal(out.reasoningEffort, "xhigh");
assert.deepEqual(writes, [], "a read-only call writes nothing");

// reasoning_effort is a persisted config field: the write goes through the
// settings service and is visible to every later reader.
await setConfig({ reasoningEffort: "low" });
out = await tool.execute({ action: "get" });
assert.equal(out.reasoningEffort, "low");

// reasoning_effort can be changed via the tool's `reasoningEffort` param.
out = await tool.execute({ action: "get", reasoningEffort: "medium" });
assert.equal(out.reasoningEffort, "medium", "effort changed via tool param");
assert.equal(out.changed, true, "changed reflects the effort change");
assert.deepEqual(writes.at(-1).patch, { reasoningEffort: "medium" }, "one merged write");
assert.equal(entryRaw.reasoningEffort, "medium", "effort persisted to the entry config");
out = await tool.execute({ action: "get", reasoningEffort: "medium" });
assert.equal(out.reasoningEffort, "medium");
assert.equal(out.changed, false, "no-op when effort already matches");

// preserveThinking defaults to false (schema default) and is reported by the tool.
out = await tool.execute({ action: "get" });
assert.equal(out.preserveThinking, false, "preserveThinking defaults to false");

// preserveThinking can be changed via the tool's `preserveThinking` param and persists.
out = await tool.execute({ action: "get", preserveThinking: true });
assert.equal(out.preserveThinking, true, "preserveThinking changed via tool param");
assert.equal(out.changed, true, "changed reflects the preserveThinking change");
assert.equal(entryRaw.preserveThinking, true, "preserveThinking persisted to the entry config");
out = await tool.execute({ action: "get", preserveThinking: true });
assert.equal(out.preserveThinking, true);
assert.equal(out.changed, false, "no-op when preserveThinking already matches");
// A single call can change mode AND preserveThinking together (one merged write).
out = await tool.execute({ action: "on", preserveThinking: false });
assert.equal(out.mode, "on");
assert.equal(out.preserveThinking, false);
assert.equal(out.changed, true);
assert.deepEqual(writes.at(-1).patch, { mode: "on", preserveThinking: false }, "one merged write");
assert.equal(entryRaw.preserveThinking, false);

// A single call can change mode AND effort together (one merged write).
await setConfig({ mode: "off" });
out = await tool.execute({ action: "on", reasoningEffort: "xhigh" });
assert.equal(out.mode, "on");
assert.equal(out.reasoningEffort, "xhigh");
assert.equal(out.changed, true);
assert.deepEqual(writes.at(-1).patch, { mode: "on", reasoningEffort: "xhigh" }, "one merged write");
assert.equal(entryRaw.mode, "on");
assert.equal(entryRaw.reasoningEffort, "xhigh");

// Restore the low effort the remainder of the suite asserts on.
out = await tool.execute({ action: "get", reasoningEffort: "low" });
assert.equal(out.reasoningEffort, "low");

out = await tool.execute({ action: "off" });
assert.equal(out.mode, "off");
assert.equal(out.changed, true);
assert.equal(out.interceptActive, true, "intercept active for default route in off mode");
assert.equal(entryRaw.mode, "off");

assert.match(contexts[0].text(), /Current mode: off/);
assert.match(contexts[0].text(), /Current reasoning_effort: low/);

out = await tool.execute({ action: "toggle" });
assert.equal(out.mode, "on");
assert.equal(entryRaw.mode, "on");
assert.equal(out.interceptActive, true, "intercept active for default route in on mode");
assert.match(contexts[0].text(), /Current mode: on/);
assert.match(contexts[0].text(), /Current reasoning_effort: low/);

out = await tool.execute({ action: "toggle" });
assert.equal(out.mode, "adaptive");

// The adaptive context must be byte-stable across calls (see the check after
// the waterfall helpers below): the host re-renders the dynamic runtime-context
// snapshot on every step and appends a new tail message whenever its text
// changes, so a per-call decision echo costs an extra prompt block per call.
assert.doesNotMatch(contexts[0].text(), /Last decision:/, "no per-call decision echo by default");

out = await tool.execute({ action: "auto" });
assert.equal(out.mode, "auto");
assert.equal(out.changed, true);
assert.equal(out.interceptActive, false, "no intercept in auto mode");
await setConfig({ reasoningEffort: "xhigh" });

// Every write so far addressed this entry at the revision it had just read.
assert.deepEqual(
  writes.map((row) => [row.ns, row.expectedRevision === row.revision]),
  writes.map(() => [ENTRY_ID, true]),
  "writes are addressed by entry id at the revision just described",
);

// A write the profile rejects is still in effect for this session (the switch
// must not silently do nothing) and is reported as not persisted.
failWrites = true;
out = await tool.execute({ action: "on", reasoningEffort: "low" });
assert.equal(out.mode, "on", "a rejected write is held for the session");
assert.equal(out.reasoningEffort, "low", "…including the fields of the same call");
assert.equal(out.persisted, false, "…and reported as session-only");
assert.notEqual(entryRaw.mode, "on", "…without the profile being touched");
out = await tool.execute({ action: "get" });
assert.equal(out.mode, "on", "the held value keeps being reported");
assert.equal(out.persisted, false);
// Once the profile accepts writes again, the held values are re-offered and the
// session converges onto the profile.
failWrites = false;
out = await tool.execute({ action: "off" });
assert.equal(out.mode, "off", "the new write applies");
assert.equal(out.reasoningEffort, "low", "the held effort survived until it could land");
assert.equal(out.persisted, true, "…and landed, so the state is backed by the profile again");
assert.equal(entryRaw.mode, "off");
assert.equal(entryRaw.reasoningEffort, "low");
await setConfig({ reasoningEffort: "xhigh" });

// ── fake request + waterfall base (the "stock adapter") ────────────────
const fakeOptions = {
  provider: "sglang",
  model: "Qwen3.8-27B",
  sessionId: "sess_test_42",
  system: "You are a test agent.",
  messages: [
    {
      id: "m1",
      role: "user",
      source: { kind: "user" },
      content: [{ type: "text", text: "weather in berlin?" }],
    },
    {
      id: "m2",
      role: "assistant",
      source: {
        kind: "model",
        provider: "sglang",
        model: "Qwen3.8-27B",
        // The durable replay state the direct route records on its own
        // turns (see the chunks[13] assertions below). Matching envelope →
        // the adapter replays this turn natively, reasoning included.
        replayState: {
          response: {
            kind: "pi-ai",
            version: 2,
            api: "openai-completions",
            provider: "sglang",
            model: "Qwen3.8-27B",
            stopReason: "toolUse",
          },
          blocks: [
            { type: "text" },
            { type: "reasoning", thinkingSignature: "reasoning_content" },
            { type: "tool-call" },
          ],
        },
      },
      content: [
        { type: "text", text: "Let me check." },
        { type: "reasoning", text: "prior thinking" },
        { type: "tool-call", id: "call_prev", name: "get_weather", arguments: '{"city":"Munich"}' },
      ],
    },
    {
      id: "m3",
      role: "user",
      source: { kind: "tool", callId: "call_prev" },
      content: [{ type: "tool-result", toolCallId: "call_prev", content: [{ type: "text", text: "sunny" }] }],
    },
    {
      id: "m4",
      role: "user",
      source: { kind: "user" },
      content: [{ type: "text", text: "now for Berlin" }],
    },
  ],
  tools: [
    {
      name: "get_weather",
      description: "Get weather for a city.",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    },
  ],
  temperature: 0.7,
  maxTokens: 1000,
};

let baseCalls = 0;
function baseFactory() {
  baseCalls += 1;
  return (async function* () {
    yield { type: "block-start", index: 0, blockType: "text" };
    yield { type: "text-delta", index: 0, text: "base" };
    yield { type: "block-end", index: 0, block: { type: "text", text: "base" } };
    yield { type: "usage", usage: { inputTokens: 1, outputTokens: 1 } };
    yield { type: "finish", reason: { kind: "stop" } };
  })();
}

/** Collect the waterfall output for one request. */
async function runWaterfall(options) {
  const chunks = [];
  for await (const chunk of ctx.waterfall(null, "llm/stream", options, baseFactory)) {
    chunks.push(chunk);
  }
  return chunks;
}

/** Re-implemented dsh-llm stream invariant: grammar validation. */
function* validateStream(source) {
  const open = new Map();
  let usageSeen = false;
  let finished = false;
  for (const chunk of source) {
    if (finished) throw new Error(`emitted ${chunk.type} after terminal finish`);
    switch (chunk.type) {
      case "block-start":
        if (open.has(chunk.index)) throw new Error(`repeated block-start index ${chunk.index}`);
        open.set(chunk.index, chunk.blockType);
        break;
      case "text-delta":
      case "reasoning-delta":
      case "tool-call-delta": {
        const expected =
          chunk.type === "text-delta" ? "text" : chunk.type === "reasoning-delta" ? "reasoning" : "tool-call";
        if (open.get(chunk.index) !== expected) throw new Error(`${chunk.type} at ${chunk.index} without open ${expected} block`);
        break;
      }
      case "block-end":
        if (open.get(chunk.index) === undefined) throw new Error(`block-end ${chunk.index} without open block`);
        if (chunk.block.type !== open.get(chunk.index)) throw new Error(`block-end ${chunk.index} type mismatch`);
        open.delete(chunk.index);
        break;
      case "usage":
        if (usageSeen) throw new Error("usage more than once");
        usageSeen = true;
        break;
      case "finish":
        if (open.size > 0 && chunk.reason.kind !== "error" && chunk.reason.kind !== "aborted")
          throw new Error(`finished with ${open.size} open block(s)`);
        finished = true;
        break;
      default:
        throw new Error(`unknown chunk type ${chunk.type}`);
    }
    yield chunk;
  }
  if (!finished) throw new Error("stream ended without a terminal finish chunk");
}

// ── case 1: mode 'off', Qwen model → direct route with tool calls ──────
await setConfig({ mode: "off" });
requests.length = 0;
baseCalls = 0;
let chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0, "stock adapter must not be called in off mode");
assert.equal(requests.length, 1, "exactly one direct HTTP request");

const req = requests[0];
assert.equal(req.url, "/v1/chat/completions");
assert.equal(req.headers.authorization, "Bearer secret-key-123");
assert.equal(req.headers["content-type"], "application/json");
assert.equal(req.body.model, "Qwen3.8-27B");
assert.equal(req.body.stream, true);
assert.deepEqual(req.body.stream_options, { include_usage: true });
assert.deepEqual(req.body.chat_template_kwargs, { enable_thinking: false, preserve_thinking: false });
// Instruct sampling forced per the model card (overrides options.temperature 0.7).
assert.equal(req.body.temperature, 0.7);
assert.equal(req.body.top_p, 0.8);
assert.equal(req.body.top_k, 20);
assert.equal(req.body.min_p, 0.0);
assert.equal(req.body.presence_penalty, 1.5);
assert.equal(req.body.repetition_penalty, 1.0);
// Default effort (xhigh) is the provider default → not sent in off mode.
assert.equal("reasoning_effort" in req.body, false);
// Stock-route body parity (pi-ai buildParams for a local sglang route):
// supportsStore → store:false; maxTokens rides on max_completion_tokens.
assert.equal(req.body.max_completion_tokens, 1000);
assert.equal("max_tokens" in req.body, false);
assert.equal(req.body.store, false);
assert.equal(req.body.tools.length, 1);
assert.equal(req.body.tools[0].type, "function");
assert.equal(req.body.tools[0].function.name, "get_weather");
// supportsStrictMode is false for plain OpenAI-compat routes → no strict key.
assert.equal("strict" in req.body.tools[0].function, false);
// Session affinity: dsh threads GenerateOptions.sessionId through to every
// request; the local sglang fork's session radix cache reads `session_id`
// from the chat-completions BODY (entrypoints/openai/protocol.py
// ChatCompletionRequest.session_id — same field the [PATCH session-id-body]
// in pi-ai sets). Without it the direct route falls off the session cache and
// re-prefills the full conversation server-side on every step.
assert.equal(req.body.session_id, "sess_test_42", "session_id forwarded to the direct route");

// wire message conversion (pi-ai parity shape)
assert.deepEqual(req.body.messages[0], { role: "system", content: "You are a test agent." });
assert.deepEqual(req.body.messages[1], { role: "user", content: "weather in berlin?" });
assert.equal(req.body.messages[2].role, "assistant");
assert.equal(req.body.messages[2].content, "Let me check.");
assert.equal(req.body.messages[2].tool_calls.length, 1);
assert.equal(req.body.messages[2].tool_calls[0].id, "call_prev");
assert.equal(req.body.messages[2].tool_calls[0].function.name, "get_weather");
assert.equal(req.body.messages[2].tool_calls[0].function.arguments, '{"city":"Munich"}');
// Same-model replayState → the thinking block is replayed as reasoning_content
// under the recorded field (the exact shape the stock route sends).
assert.equal(req.body.messages[2].reasoning_content, "prior thinking");
assert.deepEqual(req.body.messages[3], { role: "tool", content: "sunny", tool_call_id: "call_prev" });
assert.deepEqual(req.body.messages[4], { role: "user", content: "now for Berlin" });
assert.equal(req.body.messages.length, 5, "no extra bridge messages on this route");

// chunk sequence: reasoning, text, tool-call, usage, finish(tool-calls)
assert.deepEqual(
  chunks.map((c) => c.type),
  [
    "block-start", "reasoning-delta", "reasoning-delta",
    "block-start", "text-delta",
    "block-start", "tool-call-delta", "tool-call-delta", "tool-call-delta",
    "block-end", "block-end", "block-end", "usage", "finish",
  ],
);
assert.equal(chunks[0].blockType, "reasoning");
assert.equal(chunks[1].text, "Let me ");
assert.equal(chunks[3].blockType, "text");
assert.equal(chunks[4].text, "Sure");
assert.equal(chunks[5].blockType, "tool-call");
assert.equal(chunks[6].id, "call_1");
assert.equal(chunks[6].name, "get_weather");
assert.equal(chunks[6].argumentsDelta, "");
assert.equal(chunks[7].argumentsDelta, '{"ci');
assert.equal(chunks[8].argumentsDelta, 'ty":"Berlin"}');
assert.deepEqual(chunks[11].block, {
  type: "tool-call",
  id: "call_1",
  name: "get_weather",
  arguments: '{"city":"Berlin"}',
});
assert.deepEqual(chunks[12].usage, { inputTokens: 100, outputTokens: 20, cacheReadTokens: 60, reasoningTokens: 8 });
assert.deepEqual(chunks[13].reason, { kind: "tool-calls" });
assert.deepEqual(chunks[13].replayState, {
  response: {
    kind: "pi-ai",
    version: 2,
    api: "openai-completions",
    provider: "sglang",
    model: "Qwen3.8-27B",
    stopReason: "toolUse",
  },
  blocks: [
    { type: "reasoning", thinkingSignature: "reasoning_content" },
    { type: "text" },
    { type: "tool-call" },
  ],
});

// ── replay-state parity with dsh-llm-pi-ai's readReplayState contract ──
// Mirrors the validation the installed adapter applies before replaying an
// assistant turn (dsh-llm-pi-ai >= 0.2: a nested `response` envelope with
// version 2). An envelope failing this check is thrown away with
// INVALID_REPLAY_STATE and the turn degrades to provider-neutral foreign
// history, so intercepted turns must pass it verbatim.
function readReplayStateOrThrow(value) {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("invalid pi-ai replay state: expected a replay envelope");
  const response = value.response;
  if (typeof response !== "object" || response === null || Array.isArray(response))
    throw new Error("invalid pi-ai replay state: expected a response object");
  if (response.kind !== "pi-ai") throw new Error("invalid pi-ai replay state: unknown state kind");
  if (response.version !== 2)
    throw new Error(`invalid pi-ai replay state: unsupported version ${String(response.version)}`);
  for (const key of ["api", "provider", "model"])
    if (typeof response[key] !== "string" || response[key].length === 0)
      throw new Error(`invalid pi-ai replay state: ${key} must be a non-empty string`);
  if (!["stop", "length", "toolUse", "error", "aborted"].includes(String(response.stopReason)))
    throw new Error("invalid pi-ai replay state: unknown stopReason");
  if (!Array.isArray(value.blocks)) throw new Error("invalid pi-ai replay state: blocks must be an array");
  for (const [index, block] of value.blocks.entries()) {
    if (typeof block !== "object" || block === null || Array.isArray(block))
      throw new Error(`invalid pi-ai replay state: block ${index} must be an object`);
    if (!["text", "reasoning", "tool-call"].includes(String(block.type)))
      throw new Error(`invalid pi-ai replay state: block ${index} has an unknown type`);
    for (const signature of ["textSignature", "thinkingSignature", "thoughtSignature", "namespace"])
      if (block[signature] !== undefined && typeof block[signature] !== "string")
        throw new Error(`invalid pi-ai replay state: block ${index} ${signature} must be a string`);
    if (block.redacted !== undefined && typeof block.redacted !== "boolean")
      throw new Error(`invalid pi-ai replay state: block ${index} redacted must be boolean`);
  }
  return { response, blocks: value.blocks };
}
// The finish chunk from the tool-call stream must survive adapter validation.
readReplayStateOrThrow(chunks[13].replayState);
// replayedAssistant-style reconciliation: provider, model and one
// type-matching block per durable content block must line up.
{
  const durableContent = [
    { type: "reasoning", text: "Let me think." },
    { type: "text", text: "Sure" },
    { type: "tool-call", id: "call_1", name: "get_weather", arguments: '{"city":"Berlin"}' },
  ];
  const state = readReplayStateOrThrow(chunks[13].replayState);
  assert.equal(state.response.provider, "sglang");
  assert.equal(state.response.model, "Qwen3.8-27B");
  assert.equal(state.blocks.length, durableContent.length, "block count must match assistant content");
  for (const [index, block] of state.blocks.entries())
    assert.equal(block.type, durableContent[index].type, `block ${index} type mismatch`);
}
// Regression guard: the old flat version-1 form must NOT pass validation
// (this is the exact shape the adapter rejects with INVALID_REPLAY_STATE).
assert.throws(
  () =>
    readReplayStateOrThrow({
      kind: "pi-ai",
      version: 1,
      api: "openai-completions",
      provider: "sglang",
      model: "Qwen3.8-27B",
      stopReason: "stop",
      blocks: [{ type: "text" }],
    }),
  /expected a response object/,
);
// buildReplayState itself always emits the nested version-2 envelope,
// including the reasoning → thinkingSignature and tool-call mappings.
assert.deepEqual(
  buildReplayState({
    provider: "sglang",
    model: "Qwen3.8-27B",
    stopReason: "length",
    blocks: [{ kind: "text" }, { kind: "reasoning", field: "reasoning_content" }, { kind: "tool-call" }],
  }),
  {
    response: {
      kind: "pi-ai",
      version: 2,
      api: "openai-completions",
      provider: "sglang",
      model: "Qwen3.8-27B",
      stopReason: "length",
    },
    blocks: [
      { type: "text" },
      { type: "reasoning", thinkingSignature: "reasoning_content" },
      { type: "tool-call" },
    ],
  },
);
// The mirror above and the production validator (lib/parity.js, the one the
// direct route actually uses) must agree on every envelope shape: a drift
// that only the mirror catches would silently degrade real sessions.
{
  const validResponse = chunks[13].replayState.response;
  const rows = [
    { value: chunks[13].replayState, ok: true },
    { value: { kind: "pi-ai", version: 1, api: "openai-completions", provider: "sglang", model: "Qwen3.8-27B", stopReason: "stop", blocks: [] }, ok: false, re: /expected a response object/ },
    { value: { response: { ...validResponse, version: 3 }, blocks: [] }, ok: false, re: /unsupported version 3/ },
    { value: { response: { ...validResponse, stopReason: "nonsense" }, blocks: [] }, ok: false, re: /unknown stopReason/ },
    { value: { response: validResponse, blocks: [{ type: "bogus" }] }, ok: false, re: /unknown type/ },
    { value: { response: validResponse, blocks: [{ type: "reasoning", thinkingSignature: 42 }] }, ok: false, re: /thinkingSignature must be a string/ },
    { value: { response: validResponse, blocks: [{ type: "text", redacted: "yes" }] }, ok: false, re: /redacted must be boolean/ },
  ];
  for (const row of rows) {
    let mirror = "accept";
    let production = "accept";
    try {
      readReplayStateOrThrow(row.value);
    } catch (error) {
      mirror = error.message;
    }
    try {
      readPiAiReplayState(row.value);
    } catch (error) {
      production = error.message;
    }
    assert.equal(production, mirror, `validators diverge on ${JSON.stringify(row.value)}`);
    if (row.ok) assert.equal(production, "accept");
    else assert.match(production, row.re);
  }
}

// ── case 2: mode 'on' → enable_thinking true, text response ───────────
await setConfig({ mode: "on" });
serverState.sse = SSE_TEXT;
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: false });
// Thinking sampling forced per the model card.
assert.equal(requests[0].body.temperature, 1.0);
assert.equal(requests[0].body.top_p, 0.95);
assert.equal(requests[0].body.top_k, 20);
assert.equal(requests[0].body.min_p, 0.0);
assert.equal(requests[0].body.presence_penalty, 0.0);
assert.equal(requests[0].body.repetition_penalty, 1.0);
// In on mode the (default) effort is sent explicitly.
assert.equal(requests[0].body.reasoning_effort, "xhigh");
// Session affinity rides on the on-mode route too.
assert.equal(requests[0].body.session_id, "sess_test_42");
// No sessionId in the request → the field is absent entirely (not "", not
// "undefined"), so other OpenAI-compatible endpoints see an unchanged body.
requests.length = 0;
chunks = [...validateStream(await runWaterfall({ ...fakeOptions, sessionId: undefined }))];
assert.equal(requests.length, 1);
assert.equal("session_id" in requests[0].body, false, "absent sessionId sends no session_id");
// Falsy-but-defined sessionId values must also send nothing: String(null) or
// String(false) would put a shared bogus id in the body and collide every
// such conversation on one SGLang session cache key.
requests.length = 0;
chunks = [...validateStream(await runWaterfall({ ...fakeOptions, sessionId: null }))];
assert.equal(requests.length, 1);
assert.equal("session_id" in requests[0].body, false, "null sessionId sends no session_id");
requests.length = 0;
chunks = [...validateStream(await runWaterfall({ ...fakeOptions, sessionId: "" }))];
assert.equal(requests.length, 1);
assert.equal("session_id" in requests[0].body, false, "empty sessionId sends no session_id");
// Numeric session ids: only positive integers are usable. 0 / -1 / 1.5 / NaN
// are junk identifiers that would shard or collide the server-side session
// cache if stringified and sent.
for (const junk of [0, -1, 1.5, Number.NaN]) {
  requests.length = 0;
  chunks = [...validateStream(await runWaterfall({ ...fakeOptions, sessionId: junk }))];
  assert.equal(requests.length, 1);
  assert.equal("session_id" in requests[0].body, false, `numeric sessionId ${junk} sends no session_id`);
}
requests.length = 0;
chunks = [...validateStream(await runWaterfall({ ...fakeOptions, sessionId: 42 }))];
assert.equal(requests[0].body.session_id, "42", "positive integer sessionId is sent as a string");
// sessionAffinity: false opts the field out entirely (strict gateways that
// 400 on unknown body fields).
await setConfig({ sessionAffinity: false });
requests.length = 0;
chunks = [...validateStream(await runWaterfall(fakeOptions))];
assert.equal(requests.length, 1);
assert.equal("session_id" in requests[0].body, false, "sessionAffinity false sends no session_id");
await setConfig({ sessionAffinity: true });
assert.deepEqual(
  chunks.map((c) => c.type),
  ["block-start", "reasoning-delta", "block-start", "text-delta", "block-end", "block-end", "usage", "finish"],
);
assert.deepEqual(chunks[7].reason, { kind: "stop" });
assert.equal(chunks[7].replayState.response.stopReason, "stop");
assert.deepEqual(chunks[7].replayState.blocks, [
  { type: "reasoning", thinkingSignature: "reasoning_content" },
  { type: "text" },
]);
assert.deepEqual(chunks[6].usage, { inputTokens: 10, outputTokens: 4 });

// ── case 2b: mode 'on' + preserveThinking true → preserve_thinking true ──
await setConfig({ mode: "on", preserveThinking: true });
serverState.sse = SSE_TEXT;
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: true });
// restore default for subsequent cases
await setConfig({ preserveThinking: false });

// ── case 3: mode 'auto' → passthrough, no HTTP ─────────────────────────
await setConfig({ mode: "auto" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1, "stock adapter serves auto mode");
assert.equal(requests.length, 0, "no direct request in auto mode");
assert.equal(chunks.at(-1).reason.kind, "stop");
assert.equal(chunks.find((c) => c.type === "text-delta").text, "base");

// ── case 4: mode 'off', unmatched model → passthrough ─────────────────
await setConfig({ mode: "off" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall({ ...fakeOptions, model: "deepseek-v4-flash" });
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1);
assert.equal(requests.length, 0);

// ── case 5: HTTP 500 → terminal error finish, code SERVER ─────────────
serverState.status = 500;
await setConfig({ mode: "off" });
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(chunks.at(-1).reason.kind, "error");
assert.equal(chunks.at(-1).reason.failure.code, "SERVER");
assert.match(chunks.at(-1).reason.failure.message, /500/);

// ── case 6: empty stop response → EMPTY_RESPONSE error ────────────────
serverState.status = 200;
serverState.sse = SSE_EMPTY;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(chunks.at(-1).reason.kind, "error");
assert.equal(chunks.at(-1).reason.failure.code, "EMPTY_RESPONSE");

// ── case 7: image content without attachment service → passthrough ────
serverState.sse = SSE_TEXT;
await setConfig({ mode: "off" });
requests.length = 0;
baseCalls = 0;
const imageOptions = {
  ...fakeOptions,
  messages: [
    {
      id: "m9",
      role: "user",
      source: { kind: "user" },
      content: [
        { type: "text", text: "what is this?" },
        {
          type: "image",
          attachment: { attachmentId: "att_1", name: "shot.png", mediaType: "image/png", width: 64, height: 64 },
        },
      ],
    },
  ],
};
chunks = await runWaterfall(imageOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1, "image requests fall through without the attachment service");
assert.equal(requests.length, 0);

// ── case 8: image content WITH attachment service → direct route ──────
const imageTargets = [];
ctx.provide("attachments", {
  // Record the computed request target (asserted exactly in case 12) and
  // echo it back, so the handle text reflects what the route requested.
  readImageRequest: async (ref, target, signal) => {
    imageTargets.push({ ref: ref.attachmentId, target });
    return {
      data: Buffer.from("fakepngbytes"),
      mediaType: "image/png",
      bytes: 12,
      width: target.width,
      height: target.height,
    };
  },
});
await setConfig({ mode: "off" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(imageOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
const imageMsg = requests[0].body.messages.find((m) => Array.isArray(m.content));
assert.ok(imageMsg, "multimodal user message present");
assert.deepEqual(imageMsg.content[0], { type: "text", text: "what is this?" });
// Stock parity: every request image rides with a stable identity handle
// (dsh-llm requestImageHandleText) ahead of the base64 payload.
assert.equal(
  imageMsg.content[1].text,
  'Image "shot.png" (att_1); request preview 64x64px. It may be resized or re-encoded; ' +
    "source dimensions, format, and byte size may differ.",
);
assert.equal(imageMsg.content[2].type, "image_url");
assert.ok(imageMsg.content[2].image_url.url.startsWith("data:image/png;base64,"));
// The route asked for exactly the source dimensions under the default
// 4194304-px budget, with the default 1 MiB byte cap.
assert.deepEqual(imageTargets.at(-1), {
  ref: "att_1",
  target: { width: 64, height: 64, maxBytes: 1048576 },
});

// ── case 9: reasoning_effort behavior matrix ───────────────────────────
serverState.sse = SSE_TEXT;
// on + low → sent (explicit non-default)
await setConfig({ mode: "on", reasoningEffort: "low" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.equal(requests[0].body.enable_thinking === undefined, true);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: false });
assert.equal(requests[0].body.reasoning_effort, "low");

// off + low → sent (user explicitly chose a non-default level)
await setConfig({ mode: "off" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.equal(requests[0].body.reasoning_effort, "low");

// off + xhigh (default) → omitted (provider default, and thinking is off)
await setConfig({ reasoningEffort: "xhigh" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.equal("reasoning_effort" in requests[0].body, false);

// auto → passthrough: the stock adapter is used, effort never touches the wire
await setConfig({ mode: "auto" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1);
assert.equal(requests.length, 0);

// ── case 10: adaptive mode decisions ───────────────────────────────────
await setConfig({ mode: "adaptive", reasoningEffort: "xhigh" });
serverState.sse = SSE_TEXT;

// 10a: short simple last message → thinking OFF (instruct sampling, no effort)
const shortOptions = { ...fakeOptions, messages: [{ id: "s1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "ok thanks" }] }] };
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(shortOptions))];
assert.equal(baseCalls, 0, "adaptive routes directly");
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: false, preserve_thinking: false });
assert.equal(requests[0].body.presence_penalty, 1.5, "instruct preset applied");
out = await tool.execute({ action: "get" });
assert.deepEqual(out.adaptiveDecision, { on: false, reason: "acknowledgement" });

// 10b: complex question → thinking ON (thinking preset, effort sent)
const complexOptions = { ...fakeOptions, messages: [{ id: "s2", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "Why does the interceptor crash on startup? Please debug and fix the root cause." }] }] };
requests.length = 0;
chunks = [...validateStream(await runWaterfall(complexOptions))];
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: false });
assert.equal(requests[0].body.temperature, 1.0);
assert.equal(requests[0].body.reasoning_effort, "xhigh");
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.on, true);
assert.equal(out.adaptiveDecision.reason, "complex-keywords");

// 10c: model hint via a thinking_mode tool call with thinking="fast" → off
const hintedOptions = {
  ...fakeOptions,
  messages: [
    fakeOptions.messages[0],
    {
      id: "m5",
      role: "assistant",
      source: { kind: "model", provider: "sglang", model: "Qwen3.8-27B" },
      content: [
        { type: "tool-call", id: "call_h", name: "thinking_mode", arguments: '{"action":"get","thinking":"fast"}' },
      ],
    },
    {
      id: "m6",
      role: "user",
      source: { kind: "tool", callId: "call_h" },
      content: [{ type: "tool-result", toolCallId: "call_h", content: [{ type: "text", text: "adaptive" }] }],
    },
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(hintedOptions))];
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: false, preserve_thinking: false });
out = await tool.execute({ action: "get" });
assert.deepEqual(out.adaptiveDecision, { on: false, reason: "model-hint:fast" });

// 10d: model hint thinking="thinking" beats the short-message heuristic
const hintedOn = {
  ...shortOptions,
  messages: [
    {
      id: "m7",
      role: "assistant",
      source: { kind: "model", provider: "sglang", model: "Qwen3.8-27B" },
      content: [
        { type: "tool-call", id: "call_h2", name: "thinking_mode", arguments: '{"action":"get","thinking":"thinking"}' },
      ],
    },
    {
      id: "m8",
      role: "user",
      source: { kind: "tool", callId: "call_h2" },
      content: [{ type: "tool-result", toolCallId: "call_h2", content: [{ type: "text", text: "adaptive" }] }],
    },
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(hintedOn))];
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: false });
out = await tool.execute({ action: "get" });
assert.deepEqual(out.adaptiveDecision, { on: true, reason: "model-hint:thinking" });

// 10e: large context → off even for a medium message
await setConfig({ adaptive: { defaultOn: true, shortMessageChars: 40, toolRunThinking: 6, largeContextTokens: 10 } });
requests.length = 0;
chunks = [...validateStream(await runWaterfall(shortOptions))];
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: false, preserve_thinking: false });
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.on, false);
assert.match(out.adaptiveDecision.reason, /^acknowledgement|^large-context/);
// restore defaults
await setConfig({ adaptive: { defaultOn: true, shortMessageChars: 40, toolRunThinking: 6, largeContextTokens: 24000 } });

// 10f: the adaptive runtime-context text is byte-stable across decisions.
// The host re-renders the dynamic context snapshot every step and appends a
// new tail user message on ANY text change (dsh-agent-loop
// RuntimeContextProjection.project), so a per-call decision echo would put a
// fresh block at the end of the prompt on every model call.
await setConfig({ mode: "adaptive" });
const ctxBefore = contexts[0].text();
assert.doesNotMatch(ctxBefore, /Last decision:/, "decision echo off by default");
requests.length = 0;
await runWaterfall(complexOptions); // → thinking ON
assert.equal(contexts[0].text(), ctxBefore, "no text change after an on-decision");
await runWaterfall(shortOptions); // → thinking OFF
assert.equal(contexts[0].text(), ctxBefore, "no text change after an off-decision");
// Opting out restores the echo, and it tracks the latest decision.
await setConfig({ stableAdaptiveContext: false });
assert.match(contexts[0].text(), /Last decision: enable_thinking=false \(acknowledgement\)/);
await setConfig({ stableAdaptiveContext: true });
assert.equal(contexts[0].text(), ctxBefore);

// 10g: harness scaffold tail turns (system-reminder wrapper + runtime-context
// snapshot) must not hijack the heuristics — the real prompt decides. Without
// the scaffold filter the snapshot's "- Workspace: /home/…" line makes every
// live request look like a `path-reference` (thinking silently off).
const scaffoldTurns = [
  { id: "sc1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "<system-reminder>\nWorkspace instructions may be relevant.\n</system-reminder>" }] },
  { id: "sc2", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n- Workspace: /home/lovedheart/Documents/demo\n- Time: 2026-11-22" }] },
];
const scaffoldedComplex = {
  ...fakeOptions,
  messages: [
    { id: "s3", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "Why does the interceptor crash on startup? Please debug and fix the root cause." }] },
    ...scaffoldTurns,
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(scaffoldedComplex))];
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: false });
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.reason, "complex-keywords", "scaffold turns must not hijack the decision");
// A snapshot appended right after the user's own text (same turn) is also cut.
const inlineScaffold = {
  ...fakeOptions,
  messages: [
    { id: "s4", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "Why does the interceptor crash on startup? Please debug and fix the root cause.\n\nCurrent runtime context. This snapshot supersedes earlier snapshots.\n- Workspace: /home/lovedheart/Documents/demo" }] },
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(inlineScaffold))];
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: false });
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.reason, "complex-keywords", "inline snapshot must not hijack the decision");

// 10h: long tool run → thinking ON even though the only real prompt is a
// short one, because the trailing runtime-context snapshot must not break
// the trailing-tool-result count. Six assistant tool-call / user tool-result
// pairs after a short prompt, snapshot last.
const toolRunMessages = [
  { id: "s5", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "跑一遍测试。" }] },
];
for (let t = 0; t < 6; t += 1) {
  toolRunMessages.push({
    id: `a${t}`,
    role: "assistant",
    source: { kind: "model", provider: "sglang", model: "Qwen3.8-27B" },
    content: [{ type: "tool-call", id: `call_r${t}`, name: "execute_shell_command", arguments: '{"command":"true"}' }],
  });
  // The real harness appends each tool result as its own role-"tool" message.
  toolRunMessages.push({
    id: `t${t}`,
    role: "tool",
    source: { kind: "tool", callId: `call_r${t}` },
    toolCallId: `call_r${t}`,
    isError: false,
    content: [{ type: "text", text: "ok" }],
  });
}
toolRunMessages.push(...scaffoldTurns.map((turn, i) => ({ ...turn, id: `sc-run-${i}` })));
requests.length = 0;
chunks = [...validateStream(await runWaterfall({ ...fakeOptions, messages: toolRunMessages }))];
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: false });
out = await tool.execute({ action: "get" });
assert.deepEqual(out.adaptiveDecision, { on: true, reason: "long-tool-run" }, "snapshot tail must not break the tool-run count");

// 10i: scaffold detection must be per-text-block and LINE-START only.
// (a) A marker quoted mid-sentence is the user's own text: cutting there
// would erase the user's path mention and the question, flipping the
// decision from path-reference to short-simple-message.
const quotedMarker = {
  ...fakeOptions,
  messages: [
    {
      id: "qt1",
      role: "user",
      source: { kind: "user" },
      content: [
        { type: "text", text: "请帮我看看这段报错：\"Current runtime context. stale path: /var/notes/todo.md\"，定位根因" },
        { type: "text", text: "第二段同样要保留：Before marker text" },
      ],
    },
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(quotedMarker))];
assert.equal(requests.length, 1);
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.reason, "path-reference", "a mid-sentence quoted marker must not cut the user's text");
assert.equal(out.adaptiveDecision.on, false);
// (b) A scaffold text block must not swallow the user's text held in a LATER
// block of the same message: the decision comes from the real question.
const scaffoldBlockThenUserText = {
  ...fakeOptions,
  messages: [
    {
      id: "qt2",
      role: "user",
      source: { kind: "user" },
      content: [
        { type: "text", text: "Current runtime context. This snapshot supersedes earlier snapshots.\n- Workspace: /home/lovedheart/Documents/demo" },
        { type: "text", text: "Why does startup crash with an exception? debug and fix the root cause." },
      ],
    },
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(scaffoldBlockThenUserText))];
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.reason, "complex-keywords", "only the scaffold block is cut; the user's block decides");
assert.equal(out.adaptiveDecision.on, true);

// 10j: a model-hint lives until the next REAL user prompt; a trailing
// scaffold turn must not consume it (it rides in after every step), and a
// real prompt must.
const hintPlusScaffold = {
  ...fakeOptions,
  messages: [
    ...hintedOptions.messages,
    { id: "sc-hint", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "Current runtime context. This snapshot supersedes earlier snapshots.\n- Workspace: /home/lovedheart/Documents/demo" }] },
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(hintPlusScaffold))];
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.reason, "model-hint:fast", "a scaffold turn must not consume the hint");
const hintThenRealUser = {
  ...fakeOptions,
  messages: [
    ...hintedOptions.messages,
    { id: "ru1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "再帮我 debug 一下登录接口，定位报错根因并给出修复方案。" }] },
  ],
};
requests.length = 0;
chunks = [...validateStream(await runWaterfall(hintThenRealUser))];
out = await tool.execute({ action: "get" });
assert.equal(out.adaptiveDecision.reason, "complex-keywords", "a real user prompt consumes the hint");
assert.equal(out.adaptiveDecision.on, true);

// 10k: line-start detection must accept CR as well as LF. Lone-CR text (old
// Mac / mixed-EOL paste) was the real gap: the marker stayed invisible, the
// snapshot's Workspace path hijacked the decision as path-reference instead
// of the user's short prompt deciding. CRLF rides on the pre-existing \n case.
for (const [eol, label] of [["\r\n", "CRLF"], ["\r", "lone-CR"]]) {
  const crlfScaffold = {
    ...fakeOptions,
    messages: [
      {
        id: "crlf1",
        role: "user",
        source: { kind: "user" },
        content: [{ type: "text", text: `帮我修复报错：${eol}Current runtime context. This snapshot supersedes earlier snapshots.${eol}- Workspace: /home/lovedheart/Documents/demo${eol}- Time: 2026-11-22` }],
      },
    ],
  };
  requests.length = 0;
  chunks = [...validateStream(await runWaterfall(crlfScaffold))];
  out = await tool.execute({ action: "get" });
  assert.equal(out.adaptiveDecision.reason, "short-simple-message", `a ${label}-preceded scaffold marker must still be cut`);
  assert.equal(out.adaptiveDecision.on, false);
}

// ── case 11: stock-route parity edge shapes ────────────────────────────
serverState.sse = SSE_TEXT;
await setConfig({ mode: "off" });

// 11a: history the direct mirror cannot reproduce (tool-change blocks) goes
// back to the stock adapter BEFORE any HTTP request is issued.
const toolChangeTurn = {
  ...fakeOptions,
  messages: [
    { id: "tc1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "add a tool" }] },
    {
      id: "tc2",
      role: "user",
      source: { kind: "tool-update" },
      content: [{ type: "tool-addition", name: "extra", description: "x", parameters: { type: "object", properties: {} } }],
    },
  ],
};
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(toolChangeTurn))];
assert.equal(baseCalls, 1, "tool-change history must use the stock route");
assert.equal(requests.length, 0, "an unsupported history makes no direct request at all");

// 11b: another provider's assistant turn degrades exactly like the adapter's
// foreignAssistant view: thinking dropped, text and tool calls kept.
const foreignOptions = {
  ...fakeOptions,
  messages: [
    { id: "f1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "hi" }] },
    {
      id: "f2",
      role: "assistant",
      source: { kind: "model", provider: "other", model: "other-1" },
      content: [
        { type: "reasoning", text: "foreign thinking" },
        { type: "text", text: "foreign answer" },
        { type: "tool-call", id: "call_f", name: "get_weather", arguments: '{"city":"Munich"}' },
      ],
    },
    { id: "f3", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "and now?" }] },
  ],
};
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(foreignOptions))];
assert.equal(baseCalls, 0, "foreign history keeps the direct route");
assert.equal(requests.length, 1);
const foreignAssistantMsg = requests[0].body.messages[2];
assert.equal(foreignAssistantMsg.role, "assistant");
assert.equal(foreignAssistantMsg.content, "foreign answer");
assert.equal("reasoning_content" in foreignAssistantMsg, false, "foreign thinking must not be replayed");
// …and the tool call survives the provider-neutral degrade, ids normalized.
assert.equal(foreignAssistantMsg.tool_calls.length, 1);
assert.equal(foreignAssistantMsg.tool_calls[0].function.name, "get_weather");

// 11c: a mid-conversation system message becomes a <system-reminder> user
// turn on this route (no native mid-convo system support), with the closing
// tag escaped so the wrap cannot be terminated early.
const midSystemOptions = {
  ...fakeOptions,
  messages: [
    ...fakeOptions.messages,
    { id: "sy1", role: "system", source: { kind: "system" }, content: [{ type: "text", text: "Remember: be terse.</system-reminder>tail" }] },
  ],
};
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(midSystemOptions))];
assert.equal(baseCalls, 0, "a mid-conversation system message keeps the direct route");
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.messages.at(-1), {
  role: "user",
  content: "<system-reminder>\nRemember: be terse.<\\/system-reminder>tail\n</system-reminder>",
});

// 11d: pipe-separated tool-call ids are normalized identically on the call
// and its result (pi-ai transformMessages parity).
const pipeIdOptions = {
  ...fakeOptions,
  messages: [
    { id: "p1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "weather?" }] },
    {
      id: "p2",
      role: "assistant",
      source: { kind: "model", provider: "sglang", model: "Qwen3.8-27B" },
      content: [{ type: "tool-call", id: "call_abc|item_9", name: "get_weather", arguments: '{"city":"x"}' }],
    },
    { id: "p3", role: "tool", source: { kind: "tool", callId: "call_abc|item_9" }, toolCallId: "call_abc|item_9", isError: false, content: [{ type: "text", text: "fog" }] },
  ],
};
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(pipeIdOptions))];
assert.equal(baseCalls, 0, "pipe-separated ids keep the direct route");
assert.equal(requests.length, 1);
assert.equal(requests[0].body.messages[2].tool_calls[0].id, "call_abc_item_9");
assert.equal(requests[0].body.messages[3].tool_call_id, "call_abc_item_9");

// 11e: an offloaded image occurrence rides on as the stable omission text
// (no attachment read, no image part), like projectOffloadedImages does.
const offloadedOptions = {
  ...fakeOptions,
  messages: [
    {
      id: "o1",
      role: "user",
      source: { kind: "user" },
      content: [
        { type: "text", text: "the earlier screenshot" },
        { type: "image", offloaded: true, attachment: { attachmentId: "att_old", name: "old.png", mediaType: "image/png", width: 10, height: 10 } },
      ],
    },
  ],
};
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(offloadedOptions))];
assert.equal(baseCalls, 0, "offloaded-only images keep the direct route");
assert.equal(requests.length, 1);
assert.equal(
  requests[0].body.messages[1].content,
  'the earlier screenshot[image omitted to fit request image limits; "old.png" (att_old). ' +
    "No local normalized image path is available; ask the user to attach it again if needed.]",
);

// 11f: a legacy mixed user turn keeps block order — the user's own text
// precedes the tool result riding in the same message.
const legacyMixedOptions = {
  ...fakeOptions,
  messages: [
    {
      id: "l1",
      role: "user",
      source: { kind: "user" },
      content: [
        { type: "text", text: "look:" },
        { type: "tool-result", toolCallId: "call_l", content: [{ type: "text", text: "sunny" }] },
      ],
    },
  ],
};
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(legacyMixedOptions))];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.messages.slice(1), [
  { role: "user", content: "look:" },
  { role: "tool", content: "sunny", tool_call_id: "call_l" },
]);

// ── case 12: compat resolution branches ────────────────────────────────
// The other side of every flag the wire conversion reads. e2e route: a
// profile whose compat overrides flip store/maxTokens/strict.
serverState.sse = SSE_TEXT;
await setConfig({ mode: "off" });
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall({ ...fakeOptions, provider: "overrides" }))];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.equal("store" in requests[0].body, false, "supportsStore false sends no store field");
assert.equal(requests[0].body.max_tokens, 1000, "maxTokensField override respected");
assert.equal("max_completion_tokens" in requests[0].body, false);
assert.equal(requests[0].body.tools[0].function.strict, false, "supportsStrictMode true emits strict:false");

// requestImagePixelBudget → readImageRequest target scaling: the 1024-px
// budget shrinks the 64×64 source to a 32×32 request preview (the mock
// echoes the computed target into the handle text).
requests.length = 0;
chunks = [...validateStream(await runWaterfall({ ...imageOptions, provider: "budgeted" }))];
assert.equal(requests.length, 1);
const budgetedMsg = requests[0].body.messages.find((m) => Array.isArray(m.content));
assert.ok(budgetedMsg, "budgeted route still sends the multimodal message");
assert.ok(
  budgetedMsg.content[1].text.includes("request preview 32x32px"),
  `expected downscaled preview, got: ${budgetedMsg.content[1].text}`,
);
assert.deepEqual(imageTargets.at(-1), {
  ref: "att_1",
  target: { width: 32, height: 32, maxBytes: 1048576 },
});

// Unit level: pi-ai detectCompat heuristics + override merge precedence.
{
  const ds = resolveCompletionsCompat({}, "deepseek-mirror", "https://api.deepseek.com/v1");
  assert.equal(ds.supportsStore, false, "deepseek.com URL is non-standard (regression: URL match)");
  assert.equal(ds.maxTokensField, "max_tokens");
  assert.equal(ds.requiresReasoningContentOnAssistantMessages, true);
  const merged = resolveCompletionsCompat(
    { compat: { supportsStore: true, maxTokensField: "max_completion_tokens" } },
    "deepseek",
    "http://127.0.0.1:1/v1",
  );
  assert.equal(merged.supportsStore, true, "profile compat overrides win over detection");
  assert.equal(merged.maxTokensField, "max_completion_tokens");
  const plain = resolveCompletionsCompat({}, "sglang", "http://127.0.0.1:8070/v1");
  assert.deepEqual(
    { store: plain.supportsStore, max: plain.maxTokensField, strict: plain.supportsStrictMode },
    { store: true, max: "max_completion_tokens", strict: false },
    "plain local route detection (pinned e2e in case 1 too)",
  );
}

// ── case 13: reasoning_details + id-normalization + access texts ───────
serverState.sse = SSE_TEXT;
await setConfig({ mode: "off" });

// 13a: a JSON reasoning_details thinkingSignature replays as reasoning_details.
const detailsSig = JSON.stringify([{ type: "reasoning.text", text: "hidden", format: "text", index: 0 }]);
const detailsOptions = {
  ...fakeOptions,
  messages: [
    { id: "rd1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "hi" }] },
    {
      id: "rd2",
      role: "assistant",
      source: {
        kind: "model",
        provider: "sglang",
        model: "Qwen3.8-27B",
        replayState: {
          response: {
            kind: "pi-ai",
            version: 2,
            api: "openai-completions",
            provider: "sglang",
            model: "Qwen3.8-27B",
            stopReason: "stop",
          },
          blocks: [{ type: "reasoning", thinkingSignature: detailsSig }, { type: "text" }],
        },
      },
      content: [
        { type: "reasoning", text: "hidden" },
        { type: "text", text: "ok" },
      ],
    },
  ],
};
requests.length = 0;
baseCalls = 0;
chunks = [...validateStream(await runWaterfall(detailsOptions))];
assert.equal(requests.length, 1);
const detailsAssistant = requests[0].body.messages[2];
assert.deepEqual(detailsAssistant.reasoning_details, [{ type: "reasoning.text", text: "hidden", format: "text", index: 0 }]);
assert.equal("reasoning_content" in detailsAssistant, false, "signed details replace the raw reasoning field");
assert.equal(detailsAssistant.content, "ok");

// 13b: tool-call id normalization arms pinned with literals computed from
// pi-ai's own shortHash/normalizeToolCallId (openai-completions.js:865-888).
assert.equal(
  normalizeToolCallId("call_LONG_0123456789abcdefghij|item_0123456789abcdefghij", "sglang"),
  "call_LONG_0123456789abcdefghij_o0i8c09b",
  "hash-compaction arm (literal generated with @earendil-works/pi-ai shortHash)",
);
assert.equal(normalizeToolCallId("x".repeat(50), "openai"), "x".repeat(40), "openai 40-char truncation");
assert.equal(normalizeToolCallId("x".repeat(50), "sglang"), "x".repeat(50), "non-openai providers keep long ids");

// 13c: the normalized-copy access wording (currently never produced on the
// wire) is pinned anyway, so the mirrored text can never drift untested.
{
  const ref = { attachmentId: "a7", name: "n.png", width: 8, height: 8, mediaType: "image/png" };
  assert.equal(
    requestImageHandleText(ref, { width: 4, height: 4 }, { readonlyPath: "/tmp/norm.png" }),
    'Image "n.png" (a7); request preview 4x4px. Normalized copy (read-only; may be resized or ' +
      're-encoded): "/tmp/norm.png" (8x8px, image/png). Source dimensions, format, and byte size ' +
      "may differ. Copy to a writable path ending in .png before editing.",
  );
  assert.equal(
    offloadedImageText(ref, { readonlyPath: "/tmp/norm.png" }),
    '[image omitted to fit request image limits; "n.png" (a7). Normalized copy (read-only; may be ' +
      'resized or re-encoded): "/tmp/norm.png" (8x8px, image/png). Source dimensions, format, and ' +
      "byte size may differ. Copy to a writable path ending in .png before editing.]",
  );
  assert.throws(
    () => requestImageHandleText(ref, { width: undefined, height: 4 }, undefined),
    /no normalized dimensions/,
  );
}

// ── case 14: the full compat flag surface on a fussy provider ──────────
serverState.sse = SSE_TEXT;
requests.length = 0;
baseCalls = 0;
// The assistant turn's replay state must be fussy-native: same provider and
// model are what lets thinking survive the replay at all (11b pinned the
// cross-provider drop already).
const fussyOptions = {
  ...fakeOptions,
  provider: "fussy",
  messages: [
    { id: "fu1", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "weather in berlin?" }] },
    {
      id: "fu2",
      role: "assistant",
      source: {
        kind: "model",
        provider: "fussy",
        model: "Qwen3.8-27B",
        replayState: {
          response: {
            kind: "pi-ai",
            version: 2,
            api: "openai-completions",
            provider: "fussy",
            model: "Qwen3.8-27B",
            stopReason: "toolUse",
          },
          blocks: [
            { type: "text" },
            { type: "reasoning", thinkingSignature: "reasoning_content" },
            { type: "tool-call" },
          ],
        },
      },
      content: [
        { type: "text", text: "Let me check." },
        { type: "reasoning", text: "prior thinking" },
        { type: "tool-call", id: "call_prev", name: "get_weather", arguments: '{"city":"Munich"}' },
      ],
    },
    { id: "fu3", role: "tool", source: { kind: "tool", callId: "call_prev" }, toolCallId: "call_prev", isError: false, content: [{ type: "text", text: "sunny" }] },
    { id: "fu4", role: "user", source: { kind: "user" }, content: [{ type: "text", text: "now for Berlin" }] },
  ],
};
chunks = [...validateStream(await runWaterfall(fussyOptions))];
assert.equal(baseCalls, 0, "fussy flags must not force a stock fallback");
assert.equal(requests.length, 1);
const fussy = requests[0].body;
// supportsDeveloperRole → head instruction uses the developer role.
assert.equal(fussy.messages[0].role, "developer");
// requiresThinkingAsText → thinking rides inside content as a text part;
// requiresReasoningContentOnAssistantMessages seeds reasoning_content:""
// (pi-ai seeds it whenever the field would be absent).
assert.deepEqual(fussy.messages[2].content, [
  { type: "text", text: "prior thinking" },
  { type: "text", text: "Let me check." },
]);
assert.equal(fussy.messages[2].reasoning_content, "");
// requiresToolResultName → the tool result carries the tool name.
assert.equal(fussy.messages[3].name, "get_weather");
// requiresAssistantAfterToolResult → the bridge turn before the next user.
assert.deepEqual(fussy.messages[4], { role: "assistant", content: "I have processed the tool results." });
assert.deepEqual(fussy.messages[5], { role: "user", content: "now for Berlin" });
// pi-ai parity: request tools win when present…
assert.equal(fussy.tools.length, 1);
assert.equal(fussy.tools[0].function.name, "get_weather");
// …and requiresToolsWithToolHistory emits `tools: []` for a tool-carrying
// history once the request itself declares no tools.
requests.length = 0;
chunks = [...validateStream(await runWaterfall({ ...fussyOptions, tools: [] }))];
assert.deepEqual(requests[0].body.tools, []);

// ── write discipline ───────────────────────────────────────────────────
assert.deepEqual(violations, [], "only volatile fields are ever written back");

// ── done ───────────────────────────────────────────────────────────────
server.close();
console.log(`PASS: all thinking-mode plugin integration tests passed`);
console.log(`  - ${writes.length} config writes, ${requests.length} direct requests in the last case, ${baseCalls} passthroughs`);
process.exit(0);
