/**
 * Wire-parity helpers for the direct route.
 *
 * These are behavior-identical copies of pieces of the stock route, kept in
 * sync manually. Sources:
 *   - @earendil-works/pi-ai  api/openai-completions.js (detectCompat, convertTools,
 *     normalizeToolCallId/shortHash, reasoning-detail parsing, sanitizeSurrogates)
 *   - @deepseek-ai/dsh-llm-pi-ai  harness→pi conversion (replay envelope validation,
 *     tool-result/toolNames mapping, image request policy defaults)
 *   - @deepseek-ai/dsh-llm  requestImageDimensions / image handle texts
 *
 * The plugin's direct route must emit the same tokenized prompt as the stock
 * adapter for the same harness history: any byte drift breaks the server-side
 * prefix/session cache at the first differing block (re-prefill) and changes
 * what the model sees when a session switches routes.
 *
 * @module parity
 */

/** Reasoning fields endpoints use, in priority order (pi-ai parity). */
export const REASONING_FIELDS = ["reasoning_content", "reasoning", "reasoning_text"];

/** Replay-envelope contract, shared by the writer (stream.js) and validator. */
export const PI_REPLAY_KIND = "pi-ai";
export const PI_REPLAY_VERSION = 2;
export const PI_REPLAY_API = "openai-completions";

/** Default request-image policy when the route profile omits the fields. */
export const DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET = 4194304;
export const DEFAULT_REQUEST_IMAGE_MAX_BYTES = 1048576;

/**
 * Thrown while preparing a direct request when the history uses a construct
 * the mirror does not reproduce; the intercept falls back to the stock route.
 */
export class UnsupportedHistory extends Error {}

/** Replace unpaired UTF-16 surrogates (pi-ai sanitize-unicode parity). */
export function sanitizeSurrogates(text) {
  return text.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "");
}

/** pi-ai utils/hash shortHash, verbatim (used by tool-call id normalization). */
export function shortHash(str) {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h2 >>> 0).toString(36) + (h1 >>> 0).toString(36);
}

/**
 * Normalize one tool-call id for Chat Completions (pi-ai transformMessages
 * parity): pipe-separated {call_id}|{item_id} ids are sanitized and, when
 * long, hash-compacted; OpenAI's own provider truncates at 40 chars.
 */
export function normalizeToolCallId(id, provider) {
  if (id.includes("|")) {
    const separatorIndex = id.indexOf("|");
    const callId = id.slice(0, separatorIndex).replace(/[^a-zA-Z0-9_-]/g, "_");
    const itemId = id.slice(separatorIndex + 1).replace(/[^a-zA-Z0-9_-]/g, "_");
    const combinedId = itemId.length > 0 ? `${callId}_${itemId}` : callId;
    if (combinedId.length <= 40) return combinedId;
    const hash = shortHash(id).slice(0, 8);
    const prefix = callId.slice(0, Math.max(1, 40 - hash.length - 1));
    return `${prefix}_${hash}`;
  }
  if (provider === "openai") return id.length > 40 ? id.slice(0, 40) : id;
  return id;
}

/** Parse tool-call argument JSON; tolerate model malformations with {}. */
export function parseArguments(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) return parsed;
  } catch {
    // malformed model output falls through
  }
  return {};
}

/**
 * detectCompat() from pi-ai's openai-completions API for the compat fields the
 * wire conversion consumes, then apply the profile's explicit overrides
 * (a profile compat field only overrides when it is not undefined).
 */
export function resolveCompletionsCompat(profile, provider, baseUrl) {
  const url = String(baseUrl ?? "");
  const isZai = provider === "zai" || provider === "zai-coding-cn" || url.includes("api.z.ai") || url.includes("open.bigmodel.cn");
  const isTogether = provider === "together" || url.includes("api.together.ai") || url.includes("api.together.xyz");
  const isMoonshot = provider === "moonshotai" || provider === "moonshotai-cn" || url.includes("api.moonshot.");
  const isCloudflareWorkersAI = provider === "cloudflare-workers-ai" || url.includes("api.cloudflare.com");
  const isCloudflareAiGateway = provider === "cloudflare-ai-gateway" || url.includes("gateway.ai.cloudflare.com");
  const isNvidia = provider === "nvidia" || url.includes("integrate.api.nvidia.com");
  const isAntLing = provider === "ant-ling" || url.includes("api.ant-ling.com");
  const isCerebras = provider === "cerebras" || url.includes("cerebras.ai");
  const isDeepSeek = provider === "deepseek" || url.toLowerCase().includes("deepseek.com");
  const isNonStandard =
    isNvidia || isCerebras || provider === "xai" || url.includes("api.x.ai") || isTogether ||
    url.includes("chutes.ai") || isDeepSeek || isZai || isMoonshot || provider === "opencode" ||
    url.includes("opencode.ai") || isCloudflareWorkersAI || isCloudflareAiGateway || isAntLing;
  const useMaxTokens =
    url.includes("chutes.ai") || isDeepSeek || isMoonshot || isCloudflareAiGateway ||
    isTogether || isNvidia || isAntLing || isZai;
  // The developer role is OpenAI's; everything else gets the system role.
  const isOpenAI = provider === "openai" || url.includes("api.openai.com");
  const detected = {
    supportsStore: !isNonStandard,
    supportsDeveloperRole: isOpenAI,
    maxTokensField: useMaxTokens ? "max_tokens" : "max_completion_tokens",
    requiresToolResultName: false,
    requiresAssistantAfterToolResult: false,
    requiresNonNullAssistantContent: false,
    requiresToolsWithToolHistory: false,
    requiresThinkingAsText: false,
    requiresReasoningContentOnAssistantMessages: isDeepSeek,
    supportsMidConvoSystemMessages: false,
    supportsStrictMode: false,
  };
  const compat = profile?.compat;
  if (compat === null || typeof compat !== "object") return detected;
  const result = { ...detected };
  for (const key of Object.keys(detected)) {
    if (compat[key] !== undefined) result[key] = compat[key];
  }
  return result;
}

/** Valid pi-ai Chat Completions replay envelope (dsh-llm-pi-ai readReplayState). */
export function readPiAiReplayState(value) {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("invalid pi-ai replay state: expected a replay envelope");
  const response = value.response;
  if (typeof response !== "object" || response === null || Array.isArray(response))
    throw new Error("invalid pi-ai replay state: expected a response object");
  if (response.kind !== PI_REPLAY_KIND) throw new Error("invalid pi-ai replay state: unknown state kind");
  if (response.version !== PI_REPLAY_VERSION)
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

/** pi-ai isReasoningDetailObject + common-field validation, mirrored. */
function isReasoningDetail(detail) {
  if (typeof detail !== "object" || detail === null || Array.isArray(detail)) return false;
  if (
    !(detail.id === undefined || detail.id === null || typeof detail.id === "string") ||
    !(detail.format === undefined || typeof detail.format === "string") ||
    !(detail.index === undefined || typeof detail.index === "number")
  )
    return false;
  switch (detail.type) {
    case "reasoning.summary": return typeof detail.summary === "string";
    case "reasoning.encrypted": return typeof detail.data === "string";
    case "reasoning.text": return typeof detail.text === "string";
    default: return false;
  }
}

/** Parse a JSON reasoning_details array from a thinkingSignature. */
export function parseOpenAIReasoningDetails(signature) {
  if (!signature) return undefined;
  try {
    const parsed = JSON.parse(signature);
    return Array.isArray(parsed) && parsed.length > 0 && parsed.every(isReasoningDetail) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

/** Parse a legacy encrypted reasoning detail from a thoughtSignature. */
export function parseLegacyEncryptedReasoningDetail(signature) {
  if (!signature) return undefined;
  try {
    const parsed = JSON.parse(signature);
    return isReasoningDetail(parsed) &&
      parsed.type === "reasoning.encrypted" &&
      typeof parsed.id === "string" &&
      parsed.id.length > 0 &&
      typeof parsed.data === "string" &&
      parsed.data.length > 0
      ? parsed
      : undefined;
  } catch {
    return undefined;
  }
}

function quoted(value) {
  return JSON.stringify(value);
}

/** dsh-llm imageIdentity, mirrored. */
export function imageIdentity(ref) {
  return ref.name === undefined ? String(ref.attachmentId) : `${quoted(ref.name)} (${ref.attachmentId})`;
}

function extension(mediaType) {
  switch (mediaType) {
    case "image/png": return ".png";
    case "image/jpeg": return ".jpg";
    case "image/webp": return ".webp";
    case "image/gif": return ".gif";
    default: return "";
  }
}

function normalizedAccessText(ref, access) {
  return ` Normalized copy (read-only; may be resized or re-encoded): ${quoted(access.readonlyPath)} (${ref.width}x${ref.height}px, ${ref.mediaType}). Source dimensions, format, and byte size may differ. Copy to a writable path ending in ${extension(ref.mediaType)} before editing.`;
}

/** Stable text standing in for one durable image in the request. */
export function requestImageHandleText(ref, version, access) {
  // `version` (the normalized request bytes) is authoritative for the
  // preview size — exactly like the dsh-llm source, which sizes it from its
  // own normalization output; never "fix" this to read from ref.
  if (typeof version?.width !== "number" || typeof version?.height !== "number")
    throw new UnsupportedHistory(`request image ${ref?.attachmentId} version has no normalized dimensions`);
  const preview = `Image ${imageIdentity(ref)}; request preview ${version.width}x${version.height}px.`;
  return access === undefined
    ? `${preview} It may be resized or re-encoded; source dimensions, format, and byte size may differ.`
    : preview + normalizedAccessText(ref, access);
}

/** Stable placeholder for an image omitted to fit the request budget. */
export function offloadedImageText(ref, access) {
  const identity = `image omitted to fit request image limits; ${imageIdentity(ref)}.`;
  if (access === undefined) return `[${identity} No local normalized image path is available; ask the user to attach it again if needed.]`;
  return `[${identity}${normalizedAccessText(ref, access)}]`;
}

// dsh-attachment is only needed on the image path, and older hosts may not
// provide it at all: load it lazily so the plugin itself never fails to
// boot; a missing package degrades to the stock route via UnsupportedHistory.
let loadRequestImageDimensions;
async function requestImageDimensionsImpl() {
  loadRequestImageDimensions ??= import("@deepseek-ai/dsh-attachment")
    .then((module) => {
      if (typeof module.requestImageDimensions !== "function")
        throw new UnsupportedHistory("@deepseek-ai/dsh-attachment does not export requestImageDimensions");
      return module.requestImageDimensions;
    })
    .catch((error) => {
      if (error instanceof UnsupportedHistory) throw error;
      throw new UnsupportedHistory("attachment package @deepseek-ai/dsh-attachment is unavailable");
    });
  return loadRequestImageDimensions;
}

/** dsh-llm-pi-ai requestImageTarget, mirrored. */
export async function requestImageTarget(ref, budget = {}) {
  if (!Number.isFinite(ref.width) || !Number.isFinite(ref.height))
    throw new UnsupportedHistory(`image ${ref.attachmentId} reference has no usable dimensions`);
  const requestImageDimensions = await requestImageDimensionsImpl();
  return {
    ...requestImageDimensions(ref.width, ref.height, budget.maxPixels ?? DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET),
    maxBytes: budget.maxBytes ?? DEFAULT_REQUEST_IMAGE_MAX_BYTES,
  };
}
