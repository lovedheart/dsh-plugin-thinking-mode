/**
 * Standalone integration test for dsh-plugin-thinking-mode.
 *
 * Boots a minimal cordis Context with stub tools/systemPrompt/settings
 * services, applies the plugin, and exercises:
 *   - the Config schema defaults
 *   - the thinking_mode tool transitions (get/on/off/auto/toggle)
 *   - the dynamic system-prompt section
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
import { Context } from "@deepseek-ai/cordis";
import { settingsNamespace } from "@deepseek-ai/dsh-settings";

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

// ── stub services on a root cordis context ─────────────────────────────
const ctx = new Context();

const tools = [];
ctx.provide("tools", { register: (tool) => tools.push(tool) });

const sections = [];
ctx.provide("systemPrompt", { section: (section) => sections.push(section) });

const state = { mode: "auto" };
const piSection = {
  providers: {
    sglang: {
      displayName: "SGLang",
      api: "openai-completions",
      baseURL,
      apiKeyEnv: "TEST_THINKING_KEY",
    },
  },
};
let registeredSettingsNs;
ctx.provide("settings", {
  register: (ns, schema, options) => {
    registeredSettingsNs = ns;
    const defaults = schema["~standard"].validate({}).value;
    return {
      get: () => ({ ...defaults, ...state }),
      update: async (patch) => {
        Object.assign(state, patch);
      },
      watch: () => () => {},
    };
  },
  get: (ns) => {
    if (ns === settingsNamespace("llm-pi-ai")) return piSection;
    if (ns === settingsNamespace("agent-default-model"))
      return { provider: "sglang", model: "Qwen3.8-27B" };
    return undefined;
  },
});

process.env.TEST_THINKING_KEY = "secret-key-123";

// ── apply the plugin ───────────────────────────────────────────────────
assert.equal(plugin.name, "thinking-mode");
assert.deepEqual(plugin.inject, ["tools", "systemPrompt", "settings"]);

// Config schema defaults (Standard Schema v1 interface: `~standard.validate`).
const parsedConfig = plugin.Config["~standard"].validate({}).value;
assert.equal(parsedConfig.defaultMode, "auto");
assert.deepEqual(parsedConfig.modelPatterns, ["qwen"]);
assert.equal(parsedConfig.applySampling, true);
assert.deepEqual(parsedConfig.sampling.thinking, {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 0.0,
  repetition_penalty: 1.0,
});
assert.deepEqual(parsedConfig.sampling.instruct, {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 1.5,
  repetition_penalty: 1.0,
});
assert.equal(parsedConfig.defaultReasoningEffort, "xhigh");

plugin.apply(ctx, {
  defaultMode: "auto",
  modelPatterns: ["qwen"],
  applySampling: true,
  sampling: parsedConfig.sampling,
  defaultReasoningEffort: "xhigh",
});

assert.equal(registeredSettingsNs, settingsNamespace("thinking-mode"));
assert.equal(tools.length, 1, "one tool registered");
assert.equal(tools[0].name, "thinking_mode");
assert.equal(sections.length, 1, "one system-prompt section registered");
assert.equal(sections[0].name, "tool:thinking_mode");

// ── tool transitions ───────────────────────────────────────────────────
const tool = tools[0];

let out = await tool.execute({ action: "get" });
assert.equal(out.mode, "auto");
assert.equal(out.changed, false);
assert.equal(out.interceptActive, false);
assert.equal(out.sampling.applySampling, true);
assert.equal(out.sampling.thinking.temperature, 1.0);
assert.equal(out.sampling.instruct.presence_penalty, 1.5);
assert.equal(out.reasoningEffort, "xhigh");

// reasoning_effort is a persisted settings field: update + re-read.
Object.assign(state, { reasoningEffort: "low" });
out = await tool.execute({ action: "get" });
assert.equal(out.reasoningEffort, "low");

// reasoning_effort can now be changed via the tool's `reasoningEffort` param.
out = await tool.execute({ action: "get", reasoningEffort: "medium" });
assert.equal(out.reasoningEffort, "medium", "effort changed via tool param");
assert.equal(out.changed, true, "changed reflects the effort change");
assert.equal(state.reasoningEffort, "medium", "effort persisted to settings");
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
assert.equal(state.preserveThinking, true, "preserveThinking persisted to settings");
out = await tool.execute({ action: "get", preserveThinking: true });
assert.equal(out.preserveThinking, true);
assert.equal(out.changed, false, "no-op when preserveThinking already matches");
// A single call can change mode AND preserveThinking together (merge update).
out = await tool.execute({ action: "on", preserveThinking: false });
assert.equal(out.mode, "on");
assert.equal(out.preserveThinking, false);
assert.equal(out.changed, true);
assert.equal(state.preserveThinking, false);

// A single call can change mode AND effort together (merge update).
out = await tool.execute({ action: "on", reasoningEffort: "xhigh" });
assert.equal(out.mode, "on");
assert.equal(out.reasoningEffort, "xhigh");
assert.equal(out.changed, true);
assert.equal(state.mode, "on");
assert.equal(state.reasoningEffort, "xhigh");

// Restore the low effort the remainder of the suite asserts on.
out = await tool.execute({ action: "get", reasoningEffort: "low" });
assert.equal(out.reasoningEffort, "low");

out = await tool.execute({ action: "off" });
assert.equal(out.mode, "off");
assert.equal(out.changed, true);
assert.equal(out.interceptActive, true, "intercept active for default route in off mode");
assert.equal(state.mode, "off");

assert.match(sections[0].text(), /Current mode: off/);
assert.match(sections[0].text(), /Current reasoning_effort: low/);

out = await tool.execute({ action: "toggle" });
assert.equal(out.mode, "on");
assert.equal(state.mode, "on");
assert.equal(out.interceptActive, true, "intercept active for default route in on mode");
assert.match(sections[0].text(), /Current mode: on/);
assert.match(sections[0].text(), /Current reasoning_effort: low/);

out = await tool.execute({ action: "toggle" });
assert.equal(out.mode, "off");

out = await tool.execute({ action: "auto" });
assert.equal(out.mode, "auto");
assert.equal(out.changed, true);
assert.equal(out.interceptActive, false, "no intercept in auto mode");
Object.assign(state, { reasoningEffort: "xhigh" });

// ── fake request + waterfall base (the "stock adapter") ────────────────
const fakeOptions = {
  provider: "sglang",
  model: "Qwen3.8-27B",
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
      source: { kind: "model", provider: "sglang", model: "Qwen3.8-27B" },
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
state.mode = "off";
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
assert.equal(req.body.max_tokens, 1000);
assert.equal(req.body.tools.length, 1);
assert.equal(req.body.tools[0].type, "function");
assert.equal(req.body.tools[0].function.name, "get_weather");

// wire message conversion (pi-ai parity shape)
assert.deepEqual(req.body.messages[0], { role: "system", content: "You are a test agent." });
assert.deepEqual(req.body.messages[1], { role: "user", content: "weather in berlin?" });
assert.equal(req.body.messages[2].role, "assistant");
assert.equal(req.body.messages[2].content, "Let me check.");
assert.equal(req.body.messages[2].tool_calls.length, 1);
assert.equal(req.body.messages[2].tool_calls[0].id, "call_prev");
assert.equal(req.body.messages[2].tool_calls[0].function.name, "get_weather");
assert.equal(req.body.messages[2].tool_calls[0].function.arguments, '{"city":"Munich"}');
assert.deepEqual(req.body.messages[3], { role: "tool", content: "sunny", tool_call_id: "call_prev" });
assert.deepEqual(req.body.messages[4], { role: "user", content: "now for Berlin" });

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
  kind: "pi-ai",
  version: 1,
  api: "openai-completions",
  provider: "sglang",
  model: "Qwen3.8-27B",
  stopReason: "toolUse",
  blocks: [{ type: "reasoning" }, { type: "text" }, { type: "tool-call" }],
});

// ── case 2: mode 'on' → enable_thinking true, text response ───────────
state.mode = "on";
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
assert.deepEqual(
  chunks.map((c) => c.type),
  ["block-start", "reasoning-delta", "block-start", "text-delta", "block-end", "block-end", "usage", "finish"],
);
assert.deepEqual(chunks[7].reason, { kind: "stop" });
assert.equal(chunks[7].replayState.stopReason, "stop");
assert.deepEqual(chunks[7].replayState.blocks, [{ type: "reasoning" }, { type: "text" }]);
assert.deepEqual(chunks[6].usage, { inputTokens: 10, outputTokens: 4 });

// ── case 2b: mode 'on' + preserveThinking true → preserve_thinking true ──
state.mode = "on";
state.preserveThinking = true;
serverState.sse = SSE_TEXT;
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.deepEqual(requests[0].body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: true });
// restore default for subsequent cases
state.preserveThinking = false;

// ── case 3: mode 'auto' → passthrough, no HTTP ─────────────────────────
state.mode = "auto";
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1, "stock adapter serves auto mode");
assert.equal(requests.length, 0, "no direct request in auto mode");
assert.equal(chunks.at(-1).reason.kind, "stop");
assert.equal(chunks.find((c) => c.type === "text-delta").text, "base");

// ── case 4: mode 'off', unmatched model → passthrough ─────────────────
state.mode = "off";
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall({ ...fakeOptions, model: "deepseek-v4-flash" });
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1);
assert.equal(requests.length, 0);

// ── case 5: HTTP 500 → terminal error finish, code SERVER ─────────────
serverState.status = 500;
state.mode = "off";
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
state.mode = "off";
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
        { type: "image", attachment: { id: "att_1" } },
      ],
    },
  ],
};
chunks = await runWaterfall(imageOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1, "image requests fall through without the attachment service");
assert.equal(requests.length, 0);

// ── case 8: image content WITH attachment service → direct route ──────
ctx.provide("attachments", {
  readImage: async () => ({
    data: Buffer.from("fakepngbytes"),
    ref: { id: "att_1", mediaType: "image/png" },
  }),
});
state.mode = "off";
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(imageOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
const imageMsg = requests[0].body.messages.find((m) => Array.isArray(m.content));
assert.ok(imageMsg, "multimodal user message present");
assert.deepEqual(imageMsg.content[0], { type: "text", text: "what is this?" });
assert.equal(imageMsg.content[1].type, "image_url");
assert.ok(imageMsg.content[1].image_url.url.startsWith("data:image/png;base64,"));

// ── case 9: reasoning_effort behavior matrix ───────────────────────────
serverState.sse = SSE_TEXT;

// on + low → sent (explicit non-default)
state.mode = "on";
Object.assign(state, { reasoningEffort: "low" });
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
state.mode = "off";
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.equal(requests[0].body.reasoning_effort, "low");

// off + xhigh (default) → omitted (provider default, and thinking is off)
Object.assign(state, { reasoningEffort: "xhigh" });
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 0);
assert.equal(requests.length, 1);
assert.equal("reasoning_effort" in requests[0].body, false);

// auto → passthrough: the stock adapter is used, effort never touches the wire
state.mode = "auto";
requests.length = 0;
baseCalls = 0;
chunks = await runWaterfall(fakeOptions);
chunks = [...validateStream(chunks)];
assert.equal(baseCalls, 1);
assert.equal(requests.length, 0);

// ── done ───────────────────────────────────────────────────────────────
server.close();
console.log("PASS: all thinking-mode plugin integration tests passed");
console.log(`  - ${requests.length} direct requests captured, ${baseCalls} passthroughs in the last case`);
process.exit(0);
