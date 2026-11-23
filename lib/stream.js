/**
 * OpenAI-compatible SSE stream → Harness StreamChunk translation.
 *
 * Emits a protocol-valid chunk sequence for one chat-completions response:
 * paired block-start/block-end for text, reasoning, and tool-call blocks in
 * first-seen order, one usage chunk, and a terminal finish chunk carrying a
 * pi-ai-compatible replayState so the stock adapter can replay the assistant
 * turn on later requests.
 */
import { ToolCallId, EMPTY_RESPONSE_CODE } from "@deepseek-ai/dsh-llm";
import {
  PI_REPLAY_API,
  PI_REPLAY_KIND,
  PI_REPLAY_VERSION,
  REASONING_FIELDS,
} from "./parity.js";

/** Map one provider finish_reason to a Harness finish reason. */
function mapFinishReason(reason) {
  switch (reason) {
    case "stop":
    case "end":
      return { kind: "stop" };
    case "length":
      return { kind: "max-tokens" };
    case "tool_calls":
    case "function_call":
      return { kind: "tool-calls" };
    default:
      return {
        kind: "error",
        failure: { message: `Provider finish_reason: ${reason}`, code: "PROVIDER_FINISH" },
      };
  }
}

/** Map one provider usage object to Harness token usage. */
function mapUsage(usage) {
  if (usage === undefined || usage === null) return { inputTokens: 0, outputTokens: 0 };
  const result = {
    inputTokens: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
    outputTokens: typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0,
  };
  const cached = usage.prompt_tokens_details?.cached_tokens;
  if (typeof cached === "number" && cached > 0) result.cacheReadTokens = cached;
  const reasoningTokens = usage.completion_tokens_details?.reasoning_tokens;
  if (typeof reasoningTokens === "number" && reasoningTokens > 0)
    result.reasoningTokens = reasoningTokens;
  return result;
}

/** Classify a transport-level failure into a stable code. */
function classifyTransportError(error) {
  const message = String(error?.message ?? error);
  if (/\btime(?:d)?\s*out\b|timeout|ETIMEDOUT|UND_ERR_HEADERS_TIMEOUT/i.test(message))
    return "TIMEOUT";
  if (
    /\b(?:network|connection|socket|fetch)\b|\bECONN[A-Z]+\b|UND_ERR_SOCKET|UND_ERR_CONNECT/i.test(
      message,
    )
  )
    return "TRANSPORT";
  return "TRANSPORT";
}

/**
 * Build the adapter-private replay state for one successful response.
 *
 * The shape is the version-2 envelope that dsh-llm-pi-ai's readReplayState
 * validates: a nested `response` object (kind/version/api/provider/model/
 * stopReason) beside the top-level `blocks` list. The earlier flat
 * `version: 1` form is rejected ("expected a response object"), which made
 * toPiAssistant degrade every intercepted turn to provider-neutral foreign
 * history (with an INVALID_REPLAY_STATE degrade) instead of a native replay.
 *
 * `thinkingSignature` on a reasoning block records which SSE field carried
 * the reasoning (`reasoning_content`, `reasoning`, ...), so the migration
 * shim can restore the right field on old messages that lack a structured
 * reasoning part. Content is still validated on replay.
 *
 * @param {object} p - provider, model, pi-ai stopReason name, and block list.
 * @returns {object} versioned replay state accepted by the pi-ai adapter.
 */
export function buildReplayState({ provider, model, stopReason, blocks }) {
  return {
    response: {
      kind: PI_REPLAY_KIND,
      version: PI_REPLAY_VERSION,
      api: PI_REPLAY_API,
      provider,
      model,
      stopReason,
    },
    blocks: blocks.map((block) => {
      if (block.kind === "text") return { type: "text" };
      if (block.kind === "reasoning")
        return { type: "reasoning", ...(block.field === undefined ? {} : { thinkingSignature: block.field }) };
      return { type: "tool-call" };
    }),
  };
}

/**
 * Consume one OpenAI-compatible SSE response and yield Harness chunks.
 *
 * @param {Response} response - a successful (2xx) fetch response.
 * @param {object} p - provider and model ids for the replay state, plus the
 *   request's abort signal.
 * @returns {AsyncGenerator<object>} Harness StreamChunks ending in a
 *   terminal finish.
 */
export async function* streamOpenAiSse(response, { provider, model, signal }) {
  const blocks = []; // in first-seen order
  const toolBlockByStreamIndex = new Map();
  const toolBlockById = new Map();
  let textBlock = null;
  let thinkingBlock = null;
  let finishReason = null;
  let hasFinish = false;
  let usage = undefined;

  const openBlock = (kind, meta) => {
    const block =
      kind === "text"
        ? { kind: "text", text: "", index: blocks.length }
        : kind === "reasoning"
          ? { kind: "reasoning", text: "", index: blocks.length }
          : { kind: "tool-call", id: "", name: "", args: "", index: blocks.length };
    if (meta !== undefined) Object.assign(block, meta);
    blocks.push(block);
    return block;
  };
  const ensureText = () => (textBlock ??= openBlock("text"));
  const ensureThinking = (field) => (thinkingBlock ??= openBlock("reasoning", { field }));
  const ensureTool = (toolCall) => {
    const streamIndex = typeof toolCall.index === "number" ? toolCall.index : undefined;
    let block = streamIndex !== undefined ? toolBlockByStreamIndex.get(streamIndex) : undefined;
    if (block === undefined && toolCall.id) block = toolBlockById.get(toolCall.id);
    if (block === undefined) {
      block = openBlock("tool-call");
      if (streamIndex !== undefined) toolBlockByStreamIndex.set(streamIndex, block);
    }
    return block;
  };
  const closeBlocks = function* () {
    for (const block of blocks) {
      if (block.kind === "text") {
        yield { type: "block-end", index: block.index, block: { type: "text", text: block.text } };
      } else if (block.kind === "reasoning") {
        yield {
          type: "block-end",
          index: block.index,
          block: { type: "reasoning", text: block.text },
        };
      } else {
        yield {
          type: "block-end",
          index: block.index,
          block: {
            type: "tool-call",
            id: ToolCallId(block.id),
            name: block.name || "unknown",
            arguments: block.args.length > 0 ? block.args : "{}",
          },
        };
      }
    }
  };

  try {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamDone = false;
    while (!streamDone) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).replace(/\r$/, "");
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          streamDone = true;
          break;
        }
        let event;
        try {
          event = JSON.parse(data);
        } catch {
          continue; // keep-alive or malformed line
        }
        // Some gateways answer errors in-band as an event with no choices.
        if (event.error && !event.choices) {
          yield* closeBlocks();
          yield { type: "usage", usage: mapUsage(usage) };
          yield {
            type: "finish",
            reason: {
              kind: "error",
              failure: {
                message: `Provider stream error: ${event.error.message ?? JSON.stringify(event.error)}`,
                code: "PROVIDER_STREAM_ERROR",
              },
            },
          };
          return;
        }
        if (event.usage) usage = event.usage;
        for (const choice of event.choices ?? []) {
          const delta = choice.delta ?? {};
          if (choice.finish_reason) {
            finishReason = choice.finish_reason;
            hasFinish = true;
          }
          if (typeof delta.content === "string" && delta.content.length > 0) {
            const created = textBlock === null;
            const block = ensureText();
            if (created) yield { type: "block-start", index: block.index, blockType: "text" };
            block.text += delta.content;
            yield { type: "text-delta", index: block.index, text: delta.content };
          }
          const reasoningField = REASONING_FIELDS.find(
            (field) => typeof delta[field] === "string" && delta[field].length > 0
          );
          if (reasoningField !== undefined) {
            const created = thinkingBlock === null;
            const block = ensureThinking(reasoningField);
            if (created)
              yield { type: "block-start", index: block.index, blockType: "reasoning" };
            block.text += delta[reasoningField];
            yield { type: "reasoning-delta", index: block.index, text: delta[reasoningField] };
          }
          for (const toolCall of delta.tool_calls ?? []) {
            const created = !toolBlockByStreamIndex.has(toolCall.index) &&
              !toolBlockById.has(toolCall.id ?? "");
            const block = ensureTool(toolCall);
            if (created) {
              yield { type: "block-start", index: block.index, blockType: "tool-call" };
            }
            if (block.id === "" && toolCall.id) {
              block.id = toolCall.id;
              toolBlockById.set(toolCall.id, block);
            }
            const fn = toolCall.function ?? {};
            if (block.name === "" && fn.name) block.name = fn.name;
            const argumentsDelta = typeof fn.arguments === "string" ? fn.arguments : "";
            block.args += argumentsDelta;
            yield {
              type: "tool-call-delta",
              index: block.index,
              id: ToolCallId(block.id),
              ...(block.name !== "" ? { name: block.name } : {}),
              argumentsDelta,
            };
          }
        }
      }
    }

    yield* closeBlocks();

    if (!hasFinish) {
      yield { type: "usage", usage: mapUsage(usage) };
      yield {
        type: "finish",
        reason: {
          kind: "error",
          failure: {
            message: `stream ended without finish_reason (model "${model}")`,
            code: "STREAM_CLOSED",
          },
        },
      };
      return;
    }

    const reason = mapFinishReason(finishReason);
    yield { type: "usage", usage: mapUsage(usage) };
    if (reason.kind === "stop" && blocks.length === 0) {
      yield {
        type: "finish",
        reason: {
          kind: "error",
          failure: {
            message: `model "${model}" returned a completed response with no content`,
            code: EMPTY_RESPONSE_CODE,
          },
        },
      };
      return;
    }
    if (reason.kind === "stop" || reason.kind === "max-tokens" || reason.kind === "tool-calls") {
      const stopReason =
        reason.kind === "max-tokens" ? "length" : reason.kind === "tool-calls" ? "toolUse" : "stop";
      yield { type: "finish", reason, replayState: buildReplayState({ provider, model, stopReason, blocks }) };
    } else {
      yield { type: "finish", reason };
    }
  } catch (error) {
    const aborted = signal?.aborted === true || error?.name === "AbortError";
    yield* closeBlocks();
    yield { type: "usage", usage: mapUsage(usage) };
    if (aborted) {
      yield {
        type: "finish",
        reason: { kind: "aborted", failure: { message: "thinking-mode intercept request aborted", code: "ABORTED" } },
      };
    } else {
      yield {
        type: "finish",
        reason: {
          kind: "error",
          failure: { message: String(error?.message ?? error), code: classifyTransportError(error) },
        },
      };
    }
  }
}
