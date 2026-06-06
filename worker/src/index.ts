/**
 * Clicky Proxy Worker
 *
 * Proxies requests to Claude, ElevenLabs, AssemblyAI, and Gemini APIs
 * so the app never ships with raw API keys. Keys are stored as
 * Cloudflare secrets.
 *
 * Routes:
 *   POST /chat              → Anthropic Messages API (streaming)
 *   POST /tts               → ElevenLabs TTS API
 *   POST /stt               → ElevenLabs Scribe speech-to-text (multipart audio)
 *   POST /transcribe-token  → AssemblyAI temp token
 *   POST /youtube           → Generate a tutorial step plan from a YouTube video (Gemini)
 */

interface Env {
  ANTHROPIC_API_KEY: string;
  ELEVENLABS_API_KEY: string;
  ELEVENLABS_VOICE_ID: string;
  ASSEMBLYAI_API_KEY: string;
  GEMINI_API_KEY: string;
  SUPABASE_URL: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  SUPABASE_ANON_KEY?: string;
  // Static beta access token. Required as `x-glide-access-code` on
  // every request. Lets us soft-gate the worker so random people who
  // find the URL can't burn our AI budget. Falls back to "useglide"
  // if the env var isn't set.
  GLIDE_ACCESS_CODE?: string;
}

// CORS allowlist. Wide-open `*` was inviting any site to spend our
// API quota from their visitors' browsers. Now we only accept:
//   - the production app
//   - localhost (dev)
//   - any chrome-extension:// origin (the Glide extension; CWS-signed
//     extensions will have a stable ID, but during private beta we
//     allow all so dev builds work)
//   - empty/null origin (background-script fetches have no origin)
function isAllowedOrigin(origin: string | null): boolean {
  if (!origin || origin === "null") return true;
  if (origin === "https://glide.degreedlabs.com") return true;
  // Vercel-hosted app (production + preview deploys, e.g.
  // https://glide-guidance-abc123.vercel.app). Actual data access still
  // requires the user's Supabase JWT, so this is just the proxy origin gate.
  if (/^https:\/\/[A-Za-z0-9-]+\.vercel\.app$/.test(origin)) return true;
  if (origin.startsWith("chrome-extension://")) return true;
  if (origin.startsWith("http://localhost:")) return true;
  if (origin.startsWith("http://127.0.0.1:")) return true;
  return false;
}

function corsHeaders(origin: string | null): Record<string, string> {
  // If origin is null (background fetches) we send back '*'. For known
  // browser origins we echo them so credentialed requests work in the
  // future if we ever need them.
  const allow = origin && isAllowedOrigin(origin) ? origin : "*";
  return {
    "access-control-allow-origin": allow,
    "access-control-allow-methods": "GET, POST, PATCH, DELETE, OPTIONS",
    "access-control-allow-headers": "Content-Type, Authorization, X-Glide-Access-Code, X-Glide-Anon-Token",
    // Cache the preflight result for 24 h so subsequent cross-origin
    // requests skip the OPTIONS round-trip entirely. Was adding 38–160 ms
    // per request before this; now it fires once per origin/headers
    // combination per day. Chrome caps this internally at 7200 s (2 h)
    // but accepts the larger number gracefully.
    "access-control-max-age": "86400",
    "vary": "Origin",
  };
}

/** Add CORS headers to any Response. */
function withCORS(response: Response, request?: Request): Response {
  const origin = request ? request.headers.get("origin") : null;
  const newHeaders = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders(origin))) {
    newHeaders.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: newHeaders,
  });
}

function jsonError(status: number, error: string): Response {
  return new Response(JSON.stringify({ ok: false, error }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// ---------- Access-code gate ----------------------------------------------
// Every request must include `x-glide-access-code: <value>` where value
// matches GLIDE_ACCESS_CODE (defaults to "useglide"). This is a soft
// gate — the code lives in client bundles, so it's not a secret in the
// cryptographic sense. It's enough to stop random scraping and keeps the
// worker URL useless to anyone who hasn't seen the Glide app.
function verifyAccessCode(request: Request, env: Env): boolean {
  const expected = (env.GLIDE_ACCESS_CODE || "useglide").trim();
  const provided = (request.headers.get("x-glide-access-code") || "").trim();
  return provided === expected;
}

// ---------- Rate limiter --------------------------------------------------
// Sliding-window per (route, key) where key is the user id when we can
// derive one, otherwise the client IP. State is per-isolate — Cloudflare
// recycles isolates often, so this is burst protection rather than a
// hard quota. Pair with per-user spending caps once we have billing.
type RateLimitWindow = { windowMs: number; max: number };
const RATE_LIMITS: Record<string, RateLimitWindow> = {
  // Expensive inference — keep tight.
  "/coach/evaluate": { windowMs: 15 * 60 * 1000, max: 250 },
  "/coach/ask":      { windowMs: 15 * 60 * 1000, max: 150 },
  "/coach/chat":     { windowMs: 15 * 60 * 1000, max: 250 },
  "/chat":           { windowMs: 15 * 60 * 1000, max: 200 },
  // Vision-heavy / quota-tight upstreams.
  "/youtube":        { windowMs: 24 * 60 * 60 * 1000, max: 20 },
  "/page":           { windowMs: 24 * 60 * 60 * 1000, max: 50 },
  // Perplexity is the bottleneck — each call hits an external API with
  // its own quota and cost. Tighter than /page on purpose.
  "/knowledge":      { windowMs: 60 * 60 * 1000, max: 30 },
  // Gemini web-search research for tutorial planning. One call per tutorial
  // created; grounded search is moderately costly.
  "/research":       { windowMs: 60 * 60 * 1000, max: 40 },
  // TTS is cheap per call but high volume during a session.
  "/tts":            { windowMs: 15 * 60 * 1000, max: 600 },
  // STT (side-panel voice "Ask") — one call per spoken question.
  "/stt":            { windowMs: 15 * 60 * 1000, max: 200 },
  // Auth flow + telemetry.
  "/transcribe-token": { windowMs: 60 * 60 * 1000, max: 60 },
  // CRUD on user data — generous; pasting a prompt in the puck can
  // produce a burst of skill-use bumps.
  "/coach/skills":     { windowMs: 15 * 60 * 1000, max: 300 },
  "/coach/skills/use": { windowMs: 15 * 60 * 1000, max: 300 },
  // Knowledge / library routes — modest.
  "/coach/knowledge":  { windowMs: 15 * 60 * 1000, max: 200 },
  "/coach/library":    { windowMs: 15 * 60 * 1000, max: 300 },
  // Boot routes (extension fetches these to know which domains match
  // which tool). Cached client-side for an hour, so per-IP is fine.
  "/coach/library/domains/all": { windowMs: 60 * 60 * 1000, max: 30 },
  "/coach/library/rules/all":   { windowMs: 60 * 60 * 1000, max: 30 },
};

const rateLimitBuckets = new Map<string, number[]>();
function pickLimit(pathname: string): RateLimitWindow | null {
  // Exact match first, then prefix for library routes that have a
  // shared budget across CRUD verbs.
  if (RATE_LIMITS[pathname]) return RATE_LIMITS[pathname];
  if (pathname.startsWith("/coach/library/")) return RATE_LIMITS["/coach/library"];
  return null;
}
function checkRateLimit(key: string, limit: RateLimitWindow): boolean {
  const now = Date.now();
  const bucket = rateLimitBuckets.get(key) || [];
  const fresh = bucket.filter((t) => now - t < limit.windowMs);
  if (fresh.length >= limit.max) {
    rateLimitBuckets.set(key, fresh);
    return false;
  }
  fresh.push(now);
  rateLimitBuckets.set(key, fresh);
  // Prevent the per-isolate map from growing without bound.
  if (rateLimitBuckets.size > 5000) {
    const keys = Array.from(rateLimitBuckets.keys()).slice(0, 1000);
    for (const k of keys) rateLimitBuckets.delete(k);
  }
  return true;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const origin = request.headers.get("origin");

    // CORS preflight first — must respond with correct allow-headers
    // so the browser will let the real request include our custom
    // access-code header.
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    // Origin allowlist. Skip when there's no origin (background
    // fetches from the extension service worker have no origin).
    //
    // Exception: the in-page coach runs as a content script on the
    // *tool's* own origin (bolt.new, chatgpt.com, …) and calls these
    // endpoints cross-origin. They can't satisfy the app/extension
    // allowlist by design, so they're exempt here — still gated by the
    // access-code check below and the anon-token rate limiter. Keep
    // this set tight: only endpoints genuinely invoked from arbitrary
    // tool pages belong here.
    const PAGE_ORIGIN_COACH_PATHS = new Set([
      "/coach/evaluate",
      "/coach/knowledge",
    ]);
    if (origin && !isAllowedOrigin(origin) && !PAGE_ORIGIN_COACH_PATHS.has(url.pathname)) {
      return withCORS(jsonError(403, "Origin not allowed."), request);
    }

    // Access-code gate. Single check up front; no route bypasses it.
    if (!verifyAccessCode(request, env)) {
      return withCORS(jsonError(403, "Missing or invalid access code."), request);
    }

    // Library + skills routes accept full CRUD verbs; everything else is POST-only.
    const isLibraryRoute =
      url.pathname.startsWith("/coach/library/") &&
      (request.method === "POST" || request.method === "DELETE" || request.method === "GET");
    const isSkillsRoute =
      url.pathname === "/coach/skills" &&
      (request.method === "POST" || request.method === "DELETE" || request.method === "GET" || request.method === "PATCH");
    if (request.method !== "POST" && !isLibraryRoute && !isSkillsRoute) {
      return withCORS(new Response("Method not allowed", { status: 405 }), request);
    }

    // Rate limit. Try to use the signed-in user id as the key (one
    // budget across all their devices); fall back to client IP for
    // unauth boot routes.
    const limit = pickLimit(url.pathname);
    if (limit) {
      let rateKey: string | null = null;
      try {
        const auth = await tryReadUser(request, env);
        if (auth.userId) rateKey = `${url.pathname}:user:${auth.userId}`;
      } catch { /* ignore */ }
      if (!rateKey) {
        const ip = request.headers.get("cf-connecting-ip")
          || request.headers.get("x-forwarded-for")
          || "unknown";
        rateKey = `${url.pathname}:ip:${ip}`;
      }
      if (!checkRateLimit(rateKey, limit)) {
        return withCORS(
          jsonError(429, "Rate limit hit. Slow down and try again shortly."),
          request,
        );
      }
    }

    try {
      if (url.pathname === "/chat") {
        return withCORS(await handleChat(request, env), request);
      }

      if (url.pathname === "/tts") {
        return withCORS(await handleTTS(request, env), request);
      }

      if (url.pathname === "/stt") {
        return withCORS(await handleSTT(request, env), request);
      }

      if (url.pathname === "/transcribe-token") {
        return withCORS(await handleTranscribeToken(request, env), request);
      }

      if (url.pathname === "/youtube") {
        return withCORS(await handleYoutube(request, env), request);
      }

      if (url.pathname === "/page") {
        return withCORS(await handlePage(request, env), request);
      }

      if (url.pathname === "/knowledge") {
        return withCORS(await handleKnowledge(request, env), request);
      }

      if (url.pathname === "/research") {
        return withCORS(await handleResearch(request, env), request);
      }

      if (url.pathname === "/coach/ask") {
        return withCORS(await handleCoachAsk(request, env), request);
      }

      if (url.pathname === "/coach/chat") {
        return withCORS(await handleCoachChat(request, env), request);
      }

      if (url.pathname === "/coach/knowledge") {
        return withCORS(await handleCoachKnowledge(request, env), request);
      }

      if (url.pathname === "/coach/library/tools") {
        return withCORS(await handleLibraryTools(request, env), request);
      }
      if (url.pathname === "/coach/library/list") {
        return withCORS(await handleLibraryList(request, env), request);
      }
      if (url.pathname === "/coach/library/chunk") {
        return withCORS(await handleLibraryChunk(request, env), request);
      }
      if (url.pathname === "/coach/library/guide") {
        return withCORS(await handleLibraryGuide(request, env), request);
      }
      if (url.pathname === "/coach/library/rubric") {
        return withCORS(await handleLibraryRubric(request, env), request);
      }
      if (url.pathname === "/coach/library/domains") {
        return withCORS(await handleLibraryDomains(request, env), request);
      }
      if (url.pathname === "/coach/library/domains/all") {
        return withCORS(await handleLibraryDomainsAll(request, env), request);
      }
      if (url.pathname === "/coach/library/rules") {
        return withCORS(await handleLibraryRules(request, env), request);
      }
      if (url.pathname === "/coach/library/rules/all") {
        return withCORS(await handleLibraryRulesAll(request, env), request);
      }

      if (url.pathname === "/coach/skills") {
        return withCORS(await handleUserSkills(request, env), request);
      }
      if (url.pathname === "/coach/skills/use") {
        return withCORS(await handleUserSkillUse(request, env), request);
      }

      if (url.pathname === "/coach/evaluate") {
        return withCORS(await handleCoachEvaluate(request, env), request);
      }

    } catch (error) {
      console.error(`[${url.pathname}] Unhandled error:`, error);
      return withCORS(new Response(
        JSON.stringify({ error: String(error) }),
        { status: 500, headers: { "content-type": "application/json" } }
      ), request);
    }

    return withCORS(new Response("Not found", { status: 404 }), request);
  },
};

async function handleChat(request: Request, env: Env): Promise<Response> {
  // Allow anonymous trial usage via x-glide-anon-token. The welcome
  // flow runs an unauthenticated demo tutorial before the user signs
  // up; rate-limit guards against abuse.
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }

  // Read the body so we can inspect provider/model. If the caller picked
  // Gemini, we translate the Anthropic-shaped request to Gemini's API
  // and translate the response back into Anthropic's content[] shape so
  // the client doesn't need to know which model answered.
  const rawBody = await request.text();
  let parsedBody: AnthropicBody | null = null;
  try { parsedBody = JSON.parse(rawBody); } catch { /* keep null, pass through */ }

  const provider = (parsedBody && parsedBody.provider) || inferProviderFromModel(parsedBody?.model);
  if (provider === "gemini") {
    return handleChatViaGemini(parsedBody, env);
  }

  // Anthropic passthrough (default). We re-serialize so we can strip our
  // own non-Anthropic fields (provider) before forwarding.
  const upstreamBody = parsedBody ? stripWorkerFields(parsedBody) : rawBody;
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: typeof upstreamBody === "string" ? upstreamBody : JSON.stringify(upstreamBody),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`[/chat] Anthropic API error ${response.status}: ${errorBody}`);
    return new Response(errorBody, {
      status: response.status,
      headers: { "content-type": "application/json" },
    });
  }

  return new Response(response.body, {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") || "text/event-stream",
      "cache-control": "no-cache",
    },
  });
}

// ---------------------------------------------------------------------------
// Provider routing helpers for /chat. The client sends Anthropic-shaped
// requests; if `provider: "gemini"` or model starts with "gemini-", we
// route to Google's Generative Language API and translate the response.
// ---------------------------------------------------------------------------

interface AnthropicBody {
  provider?: string;
  model?: string;
  max_tokens?: number;
  temperature?: number;
  system?: string;
  messages?: AnthropicMessage[];
  tools?: unknown[];
  betas?: string[];
  // Anything else is forwarded as-is to Anthropic.
  [k: string]: unknown;
}

interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: { type: "base64"; media_type: string; data: string } | { type: "url"; url: string } }
  | { type: string; [k: string]: unknown };

function extractSystemText(system: unknown): string {
  if (typeof system === "string") return system.trim();
  if (Array.isArray(system)) {
    return system
      .map((b: any) =>
        b && b.type === "text" && typeof b.text === "string" ? b.text : ""
      )
      .filter((s) => s.length > 0)
      .join("\n\n")
      .trim();
  }
  return "";
}

function inferProviderFromModel(model: unknown): "anthropic" | "gemini" | undefined {
  if (typeof model !== "string") return undefined;
  if (model.startsWith("gemini")) return "gemini";
  if (model.startsWith("claude")) return "anthropic";
  return undefined;
}

function stripWorkerFields<T extends AnthropicBody>(body: T): Omit<T, "provider"> {
  const { provider: _drop, ...rest } = body;
  return rest;
}

async function handleChatViaGemini(body: AnthropicBody | null, env: Env): Promise<Response> {
  if (!body) {
    return new Response(JSON.stringify({ error: "Body must be JSON for gemini provider" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  if (body.stream === true) {
    return handleChatViaGeminiStream(body, env);
  }
  if (!env.GEMINI_API_KEY) {
    return new Response(JSON.stringify({ error: "GEMINI_API_KEY is not configured" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }

  // Default to Google's "latest flash" alias so the experiment picks up
  // model upgrades automatically. Caller can override via body.model.
  const model = (typeof body.model === "string" && body.model) || "gemini-flash-latest";

  const contents: Array<{ role: "user" | "model"; parts: GeminiPart[] }> = [];
  for (const m of body.messages || []) {
    const role: "user" | "model" = m.role === "assistant" ? "model" : "user";
    const parts = anthropicContentToGeminiParts(m.content);
    if (parts.length) contents.push({ role, parts });
  }

  const reqBody: Record<string, unknown> = {
    contents,
    generationConfig: {
      temperature: typeof body.temperature === "number" ? body.temperature : 0.4,
      maxOutputTokens: typeof body.max_tokens === "number" ? body.max_tokens : 768,
      // Force JSON output. Without this, Gemini frequently wraps its
      // response in ```json ... ``` markdown fences or adds prose
      // before/after the object — both of which trip the client's
      // /\{[\s\S]*\}/ extractor and surface as the "let me take
      // another look at your screen" parse-failure fallback. With
      // responseMimeType=application/json (and no responseSchema), the
      // model returns a raw JSON object that JSON.parse handles
      // directly. Claude doesn't need this — it already follows the
      // "valid JSON only" system-prompt instruction.
      responseMimeType: "application/json",
      // Disable extended thinking. 2.5-flash thinks by default, and
      // thinking tokens are drawn from maxOutputTokens — so a modest cap
      // gets spent reasoning and the JSON output truncates mid-object
      // ("unterminated JSON object" on the client). For these structured,
      // latency-sensitive calls we want the full budget on output and no
      // thinking latency.
      thinkingConfig: { thinkingBudget: 0 },
    },
  };
  // Anthropic accepts both `system: "string"` and `system: [{ type:'text',
  // text:'...', cache_control: {...} }]`. The latter is what we send for
  // prompt-caching. Translate both forms to Gemini's systemInstruction.
  const systemText = extractSystemText(body.system);
  if (systemText) {
    reqBody.systemInstruction = { parts: [{ text: systemText }] };
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const upstream = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-goog-api-key": env.GEMINI_API_KEY,
    },
    body: JSON.stringify(reqBody),
  });

  if (!upstream.ok) {
    const errorBody = await upstream.text();
    console.error(`[/chat gemini] ${upstream.status}: ${errorBody}`);
    // Surface Gemini's error in the Anthropic error shape so the client
    // logs say something useful.
    return new Response(
      JSON.stringify({
        type: "error",
        error: { type: "upstream_error", message: errorBody.slice(0, 400) },
      }),
      { status: upstream.status, headers: { "content-type": "application/json" } },
    );
  }

  const data: any = await upstream.json();
  // Pull text out of the first candidate's parts. Gemini puts text in
  // candidates[0].content.parts[].text — concatenate them.
  const candidate = data?.candidates?.[0];
  const textParts: string[] = [];
  for (const p of candidate?.content?.parts || []) {
    if (typeof p?.text === "string") textParts.push(p.text);
  }
  const responseText = textParts.join("");

  // Translate back to Anthropic-shaped { content: [{ type: 'text', text }] }
  // so the client's extractTextContent path Just Works.
  const stopReason = mapGeminiFinishReason(candidate?.finishReason);
  const anthropicShape = {
    id: data?.responseId || `gemini_${Date.now()}`,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: responseText }],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: data?.usageMetadata?.promptTokenCount || 0,
      output_tokens: data?.usageMetadata?.candidatesTokenCount || 0,
    },
    // Diagnostic — lets the client confirm which provider answered without
    // re-checking the request body.
    _provider: "gemini",
  };
  return new Response(JSON.stringify(anthropicShape), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

// Streaming Gemini path. Connects to :streamGenerateContent?alt=sse and
// translates Gemini's per-chunk JSON into the Anthropic SSE shape that
// the client's consumeAnthropicStream already speaks. Lets early-TTS,
// early-highlight, and first-token timing kick in for Gemini calls
// without any client-side changes beyond removing the
// provider-anthropic gate.
async function handleChatViaGeminiStream(body: AnthropicBody, env: Env): Promise<Response> {
  if (!env.GEMINI_API_KEY) {
    return new Response(JSON.stringify({ error: "GEMINI_API_KEY is not configured" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }

  const model = (typeof body.model === "string" && body.model) || "gemini-flash-latest";

  const contents: Array<{ role: "user" | "model"; parts: GeminiPart[] }> = [];
  for (const m of body.messages || []) {
    const role: "user" | "model" = m.role === "assistant" ? "model" : "user";
    const parts = anthropicContentToGeminiParts(m.content);
    if (parts.length) contents.push({ role, parts });
  }

  const reqBody: Record<string, unknown> = {
    contents,
    generationConfig: {
      temperature: typeof body.temperature === "number" ? body.temperature : 0.4,
      maxOutputTokens: typeof body.max_tokens === "number" ? body.max_tokens : 768,
      responseMimeType: "application/json",
      // See non-streaming handler: disable thinking so maxOutputTokens is
      // spent entirely on the JSON output (avoids truncation) and latency
      // stays low.
      thinkingConfig: { thinkingBudget: 0 },
    },
  };
  const systemText = extractSystemText(body.system);
  if (systemText) {
    reqBody.systemInstruction = { parts: [{ text: systemText }] };
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`;
  const upstream = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-goog-api-key": env.GEMINI_API_KEY,
    },
    body: JSON.stringify(reqBody),
  });

  if (!upstream.ok || !upstream.body) {
    const errorBody = await upstream.text().catch(() => "");
    console.error(`[/chat gemini stream] ${upstream.status}: ${errorBody}`);
    return new Response(
      JSON.stringify({
        type: "error",
        error: { type: "upstream_error", message: errorBody.slice(0, 400) },
      }),
      { status: upstream.status, headers: { "content-type": "application/json" } },
    );
  }

  const encoder = new TextEncoder();
  const messageId = `gemini_${Date.now()}`;

  // Serialize one Anthropic-format SSE frame. The client only reads the
  // `data:` line, but we emit `event:` too for human readability of any
  // tcpdumps / curl probes.
  function frame(eventType: string, payload: object): Uint8Array {
    return encoder.encode(`event: ${eventType}\ndata: ${JSON.stringify(payload)}\n\n`);
  }

  const upstreamBody = upstream.body;
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      // Anthropic's message_start: tells the client the model and zero-initial usage.
      controller.enqueue(frame("message_start", {
        type: "message_start",
        message: {
          id: messageId,
          type: "message",
          role: "assistant",
          content: [],
          model,
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
          },
          _provider: "gemini",
        },
      }));
      controller.enqueue(frame("content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      }));

      let inputTokens = 0;
      let outputTokens = 0;
      let stopReason = "end_turn";

      try {
        const reader = upstreamBody.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });

          // Gemini's SSE: each frame is `data: {...}\n\n`. Parse
          // line-by-line and skip non-data lines.
          let nl;
          while ((nl = buf.indexOf("\n")) !== -1) {
            const line = buf.slice(0, nl).replace(/\r$/, "");
            buf = buf.slice(nl + 1);
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (!payload) continue;
            let chunk: any;
            try { chunk = JSON.parse(payload); } catch { continue; }

            const candidate = chunk?.candidates?.[0];
            const parts = candidate?.content?.parts || [];
            for (const p of parts) {
              if (typeof p?.text === "string" && p.text) {
                controller.enqueue(frame("content_block_delta", {
                  type: "content_block_delta",
                  index: 0,
                  delta: { type: "text_delta", text: p.text },
                }));
              }
            }
            if (chunk?.usageMetadata) {
              inputTokens = chunk.usageMetadata.promptTokenCount || inputTokens;
              outputTokens = chunk.usageMetadata.candidatesTokenCount || outputTokens;
            }
            if (candidate?.finishReason) {
              stopReason = mapGeminiFinishReason(candidate.finishReason);
            }
          }
        }
      } catch (err) {
        console.error("[/chat gemini stream] read error:", err);
      }

      // Closeout: content_block_stop, message_delta (final usage), message_stop.
      controller.enqueue(frame("content_block_stop", {
        type: "content_block_stop",
        index: 0,
      }));
      controller.enqueue(frame("message_delta", {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { input_tokens: inputTokens, output_tokens: outputTokens },
      }));
      controller.enqueue(frame("message_stop", { type: "message_stop" }));
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      // Disable Cloudflare's edge buffering so chunks hit the client
      // as Gemini emits them. Without this the stream tends to coalesce
      // into a single flush, defeating the whole point.
      "x-accel-buffering": "no",
    },
  });
}

type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } };

function anthropicContentToGeminiParts(content: AnthropicMessage["content"]): GeminiPart[] {
  if (typeof content === "string") {
    return content ? [{ text: content }] : [];
  }
  const parts: GeminiPart[] = [];
  for (const block of content || []) {
    if (block.type === "text" && typeof block.text === "string" && block.text) {
      parts.push({ text: block.text });
    } else if (block.type === "image") {
      const src = (block as any).source;
      if (src && src.type === "base64" && typeof src.data === "string") {
        parts.push({
          inlineData: {
            mimeType: typeof src.media_type === "string" ? src.media_type : "image/png",
            data: src.data,
          },
        });
      } else if (src && src.type === "url" && typeof src.url === "string") {
        // Gemini supports inline images only; a URL would require fetching.
        // For our use case (screenshots come as base64) this path is rare.
        // We could fetch and inline it here if it shows up in practice.
      }
    }
    // Anthropic-specific blocks (tool_use, tool_result, computer_use) have
    // no clean Gemini equivalent yet. Drop them silently — Gemini provider
    // is a vision-only experiment for now.
  }
  return parts;
}

function mapGeminiFinishReason(reason: unknown): string {
  switch (reason) {
    case "STOP": return "end_turn";
    case "MAX_TOKENS": return "max_tokens";
    case "SAFETY": return "stop_sequence";
    case "RECITATION": return "stop_sequence";
    default: return "end_turn";
  }
}

async function handleTranscribeToken(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }

  const response = await fetch(
    "https://streaming.assemblyai.com/v3/token?expires_in_seconds=480",
    {
      method: "GET",
      headers: {
        authorization: env.ASSEMBLYAI_API_KEY,
      },
    }
  );

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`[/transcribe-token] AssemblyAI token error ${response.status}: ${errorBody}`);
    return new Response(errorBody, {
      status: response.status,
      headers: { "content-type": "application/json" },
    });
  }

  const data = await response.text();
  return new Response(data, {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/* -------------------------------------------------------------------------- */
/*                                   YouTube                                  */
/* -------------------------------------------------------------------------- */

const YOUTUBE_URL_REGEX =
  /^https?:\/\/(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)([A-Za-z0-9_-]{6,15})(?:[?&#].*)?$/;

const GEMINI_MODEL = "gemini-2.5-flash";

const STEP_OBJECT_SCHEMA = {
  type: "object",
  properties: {
    number: { type: "integer" },
    description: { type: "string" },
    stepType: {
      type: "string",
      enum: ["navigate", "click", "type", "verify", "wait", "info"],
    },
    visualHint: { type: "string" },
    demoInput: { type: "string" },
    expectedDuration: { type: "integer" },
  },
  required: ["number", "description", "stepType"],
};

const STEP_PLAN_SCHEMA = {
  type: "object",
  properties: {
    refusal: {
      type: "string",
      description:
        "If this is not an instructional source, set this to a brief reason and omit the other fields.",
    },
    kind: {
      type: "string",
      enum: ["tutorial", "journey"],
      description:
        "tutorial: single coherent walkthrough (use steps[]). journey: multi-phase workflow split into separate tutorials (use tutorials[]).",
    },
    title: { type: "string" },
    summary: { type: "string" },
    browserCompatible: { type: "boolean" },
    shareRecommendation: {
      type: "object",
      properties: {
        scope: { type: "string", enum: ["browser", "window", "screen"] },
        reason: { type: "string" },
      },
      required: ["scope", "reason"],
    },
    steps: {
      type: "array",
      items: STEP_OBJECT_SCHEMA,
      description:
        "Use this for kind=tutorial only. Single flat list of steps.",
    },
    tutorials: {
      type: "array",
      description:
        "Use this for kind=journey only. Each item is a coherent phase with its own title, summary, and steps.",
      items: {
        type: "object",
        properties: {
          title: { type: "string" },
          summary: { type: "string" },
          steps: { type: "array", items: STEP_OBJECT_SCHEMA },
        },
        required: ["title", "steps"],
      },
    },
  },
};

const STEP_PLAN_PROMPT = `You are turning the attached video into a guided, do-it-yourself tutorial for someone who HAS NOT WATCHED THE VIDEO and never will.

This is the most important constraint and you should treat it as inviolable: the user only sees what is on the screen and hears the spoken instruction for the current step. They have zero memory of anything the presenter explained. Every step must stand on its own.

Output JSON matching the response schema.

How to translate the video into a self-contained tutorial:
- USE "info" STEPS SPARINGLY. At MOST ONE opening info step (1–2 sentences) to set the goal and prerequisites. Avoid info steps elsewhere. If you'd write "Now we'll do X" as a standalone info step, instead fold that one-clause framing into the very next action step's description: "Open Settings to grant calendar access — Settings is the gear icon in the top right." This keeps the user moving instead of stopping to read narration.
- NEVER use info steps to summarize what just happened ("Great, you're now on the dashboard!"). The screen already shows it.
- NEVER use info steps as section headers ("Now configure the agent"). Just start the next action step.
- Each action step must be SELF-EXPLANATORY: include a brief reason or context inline when the action would otherwise be opaque. Bad: "Click the gear icon." Good: "Click the gear icon in the top right to open Settings — that's where the integration toggle lives."
- Whenever a step uses a TERM, NAME, or CONCEPT introduced earlier in the video (like "your tracking dashboard", "the linked workspace", "the source file"), define it briefly in the step itself. Don't assume the user remembers what the presenter called it.
- For "type" steps, if the presenter typed specific example data, include both a generic adaptable placeholder AND a one-line explanation of what kind of value belongs there. Bad: 'Type "Coffee with Jane".' Good: 'Type a short title for your event (e.g. "Coffee with Jane") — this is what shows up on the calendar block.'
- For "verify" steps, describe the expected screen state in CONCRETE visible terms, not "what the presenter showed". Bad: "Confirm the result looks like in the video." Good: "Confirm a green 'Connected' badge appears next to the integration name."
- Fill in obvious prerequisites the video skips. If the video opens on a logged-in dashboard, add steps for signing in / opening the right app first. If it assumes an account exists, add a note about creating one.
- Smooth out gaps. If the presenter narrates two actions in one sentence, split them into two steps. If they gloss over a clearly-needed step (closing a modal, accepting cookies, scrolling, waiting for a load), include it.
- Drop presenter-specific framing. "As you can see on my screen" → describe what the user should see on theirs. "I'll click here" → "Click X." Never reference "the video", "the presenter", "earlier", or "as we discussed".
- Where the video is unclear about what comes next, infer a reasonable next action based on the platform's conventions. Better to give a confident, plausible step than leave the user stranded.

Anti-patterns that cause users to feel lost — REJECT these in your own output:
- Steps that say only "click X" with no reason when the reason was given in the video.
- Verify steps that don't say what to look for in concrete visible terms.
- Type steps with no demoInput AND no description of what to enter.
- Pronouns referring to something only the video named ("click that button", "select the one we made earlier").

DO NOT include the presenter's intro/outro, sponsor reads, calls to subscribe, or commentary about themselves or their channel.
DO NOT invent steps that contradict the video. Filling gaps means adding small connectives, not inventing a different workflow.

Single tutorial vs. journey:
- If the entire workflow fits in roughly 10 atomic steps or fewer, output a single tutorial: set kind="tutorial", populate steps[], and leave tutorials empty.
- If the workflow is bigger than that, BREAK IT into a journey of multiple tutorials. Set kind="journey", leave the top-level steps empty, and populate tutorials[]. Each tutorial covers one coherent phase (typically 5–10 steps) — pick natural phase boundaries like "set up the account", "configure the integration", "send your first message", "verify it worked". Don't just chop every 10 steps; the boundaries should feel like meaningful checkpoints where the user could pause.
- Each tutorial in the journey gets its own title (action-oriented, user-perspective) and a one-sentence summary. The top-level title and summary describe the overall journey.
- The same self-contained-step rules apply inside every tutorial of a journey — each one should make sense as a standalone read.

Field rules:
- title: short, action-oriented, user-perspective (max 60 chars). Example: "Post your first article on LinkedIn".
- summary: one or two sentences describing what the user will accomplish and walk away with.
- kind: "tutorial" or "journey" — see the rules above.
- steps: atomic user actions or short info beats. Number sequentially starting at 1. Use only when kind="tutorial". Aim for 5–10 steps for a single tutorial; if you'd write more, you should be making a journey instead.
- tutorials: array of tutorial objects. Use only when kind="journey". Each has its own { title, summary, steps[] }; numbering inside each tutorial restarts at 1.
- stepType must be one of: navigate, click, type, verify, wait, info.
- description: a single instruction in the second person, written so a cold reader understands WHAT to do, WHERE, and WHY.
- visualHint: a short cue for what the user should see once the step is complete. Required for click/navigate/type/verify steps. Concrete and visible — text labels, badges, color changes, panels appearing.
- demoInput: for type steps, a realistic adaptable sample value the user can copy if they don't have their own.
- expectedDuration: rough seconds for an average user to complete this step.
- browserCompatible: true ONLY if the entire workflow happens inside a web browser.
- shareRecommendation.scope: "browser" if entirely in a browser tab; "window" if a single desktop app; "screen" if it spans multiple windows.

If this video is NOT a tutorial — music video, vlog, ad, abstract demo with no actionable user workflow — set the "refusal" field to a one-sentence reason and omit the other fields. Do that check before generating any steps.`;

interface YoutubeRequestBody {
  videoUrl?: string;
}

async function handleYoutube(request: Request, env: Env): Promise<Response> {
  // Part of the anonymous trial so the side panel can build a tutorial
  // from a YouTube video. Rate-limited (20/day per IP) to protect the
  // Gemini video quota.
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  if (!env.GEMINI_API_KEY) {
    return new Response(
      JSON.stringify({ error: "GEMINI_API_KEY is not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  let body: YoutubeRequestBody;
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Body must be JSON: { videoUrl }" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const rawUrl = (body.videoUrl || "").trim();
  const match = rawUrl.match(YOUTUBE_URL_REGEX);
  if (!match) {
    return new Response(
      JSON.stringify({
        error:
          "videoUrl must be a public YouTube URL (youtube.com/watch?v=…, youtu.be/…, or shorts/embed).",
      }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }
  // Normalize: strip playlist (`list=`), timestamp (`t=`), share (`si=`),
  // and index params before sending to Gemini. Gemini's fileData.fileUri
  // wants the canonical single-video URL — anything else has triggered
  // INVALID_ARGUMENT in practice (e.g. when the link came from a video
  // playing inside a playlist).
  const videoId = match[1];
  const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;

  const geminiBody = {
    contents: [
      {
        parts: [
          { fileData: { fileUri: videoUrl, mimeType: "video/*" } },
          { text: STEP_PLAN_PROMPT },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.4,
      responseMimeType: "application/json",
      responseSchema: STEP_PLAN_SCHEMA,
    },
  };

  const geminiUrl =
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=` +
    encodeURIComponent(env.GEMINI_API_KEY);

  const response = await fetch(geminiUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(geminiBody),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`[/youtube] Gemini API error ${response.status}: ${errorBody}`);
    // Surface the real failure mode to the client so the UI can show a
    // useful message instead of a generic "Bad Gateway". Common cases:
    //  - 429 RESOURCE_EXHAUSTED: daily quota hit
    //  - 400 INVALID_ARGUMENT: video is private/age-gated/too-long/region-locked
    //  - 403: API key/billing issue
    const parsed = safeJsonParse(errorBody) as any;
    const geminiStatus = parsed?.error?.status || "";
    const geminiMessage = parsed?.error?.message || "";
    let userMessage = "We couldn't generate steps from this video.";
    if (response.status === 429 || geminiStatus === "RESOURCE_EXHAUSTED") {
      userMessage =
        "Daily video-processing quota reached. Try again in a few hours, " +
        "or paste the video link as a prompt instead.";
    } else if (response.status === 400 || geminiStatus === "INVALID_ARGUMENT") {
      userMessage =
        "Gemini couldn't read this video. It might be private, " +
        "age-restricted, region-locked, or longer than the free tier allows. " +
        "Try a different public video.";
    } else if (response.status === 403) {
      userMessage =
        "Video processing is not enabled on the configured Gemini key.";
    }
    // Pass through Gemini's status so the client can branch on it (e.g.
    // show a different banner for 429 vs 400). Wrap in 200 with ok:false
    // so fetch's response.ok-based error handling doesn't swallow it.
    return new Response(
      JSON.stringify({
        ok: false,
        error: userMessage,
        upstreamStatus: response.status,
        upstreamCode: geminiStatus,
        upstreamMessage: geminiMessage,
      }),
      {
        status: response.status === 429 ? 429 : 502,
        headers: { "content-type": "application/json" },
      }
    );
  }

  let payload: any;
  try {
    payload = await response.json();
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Gemini returned a non-JSON body" }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const text: string | undefined =
    payload?.candidates?.[0]?.content?.parts?.[0]?.text;

  if (!text) {
    console.error("[/youtube] Gemini returned no text candidate", payload);
    return new Response(
      JSON.stringify({ error: "Gemini returned no content", raw: payload }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const plan = safeJsonParse(text);
  if (!plan || typeof plan !== "object") {
    return new Response(
      JSON.stringify({ error: "Gemini returned non-JSON content", raw: text }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  return new Response(JSON.stringify({ plan, videoUrl }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/* -------------------------------------------------------------------------- */
/*                          Page (article / how-to)                           */
/* -------------------------------------------------------------------------- */

const PAGE_PROMPT = `You are turning the attached web page into a guided, do-it-yourself tutorial for someone who HAS NOT READ THE ARTICLE and never will.

This is the most important constraint and you should treat it as inviolable: the user only sees what is on the screen and hears the spoken instruction for the current step. They have zero memory of anything the article explained. Every step must stand on its own.

Output JSON matching the response schema.

How to translate the article into a self-contained tutorial:
- OPEN with one or two short "info" steps that orient the user: what they're about to accomplish, what they'll need (account, file, app already open), and any one-time setup. The user should never wonder "wait, where am I supposed to be?"
- Each action step must be SELF-EXPLANATORY: include a brief reason or context inline when the action would otherwise be opaque. Bad: "Click the gear icon." Good: "Click the gear icon in the top right to open Settings — that's where the integration toggle lives."
- Whenever a step uses a TERM, NAME, or CONCEPT introduced earlier in the article (like "your tracking dashboard", "the linked workspace", "the source file"), define it briefly in the step itself. Don't assume the user remembers what the article called it.
- For "type" steps, if the article gave specific example data, include both a generic adaptable placeholder AND a one-line explanation of what kind of value belongs there.
- For "verify" steps, describe the expected screen state in CONCRETE visible terms — text labels, badges, color changes, panels appearing — not "what the article showed".
- Fill in obvious prerequisites the article skips. If it assumes an account, an open app, or an installed tool, add explicit steps for those.
- Smooth out gaps. If a paragraph mentions two actions, split them into two steps. If the article glosses over a clearly-needed step (closing a modal, accepting cookies, scrolling, waiting for a load), include it.
- Drop author commentary, intro/outro, sponsor reads, and anything about the writer's own experience. Never reference "the article", "the author", "earlier", or "as we discussed".
- Use "info" steps at any transition between phases when the goal of the next phase isn't obvious from the next action alone.

Anti-patterns that cause users to feel lost — REJECT these in your own output:
- Steps that say only "click X" with no reason when the reason was given in the article.
- Verify steps that don't say what to look for in concrete visible terms.
- Type steps with no demoInput AND no description of what to enter.
- Pronouns referring to something only the article named ("click that button", "select the one we made earlier").

DO NOT invent steps that contradict the article. Filling gaps means adding small connectives, not inventing a different workflow.

Single tutorial vs. journey:
- If the entire workflow fits in roughly 10 atomic steps or fewer, output a single tutorial: set kind="tutorial", populate steps[], leave tutorials empty.
- If the workflow is bigger than that, BREAK IT into a journey of multiple tutorials. Set kind="journey", leave the top-level steps empty, and populate tutorials[]. Each tutorial covers one coherent phase (typically 5–10 steps). Pick natural phase boundaries — meaningful checkpoints — not arbitrary chunks.
- Each tutorial gets its own title and summary. The top-level title and summary describe the overall journey.
- Self-contained-step rules apply inside every tutorial of a journey.

Field rules:
- title: short, action-oriented, user-perspective (max 60 chars).
- summary: one or two sentences describing what the user will accomplish.
- kind: "tutorial" or "journey" — see the rules above.
- steps: use only when kind="tutorial". Aim for 5–10 steps; if you'd write more, you should be making a journey instead.
- tutorials: use only when kind="journey". Each has { title, summary, steps[] }; numbering inside each tutorial restarts at 1.
- stepType must be one of: navigate, click, type, verify, wait, info.
- description: a single instruction in the second person, written so a cold reader understands WHAT to do, WHERE, and WHY.
- visualHint: a concrete visible cue for what the user should see once the step is complete (required for click/navigate/type/verify).
- demoInput: for type steps, a realistic adaptable sample value the user can copy if they don't have their own.
- expectedDuration: rough seconds for an average user.
- browserCompatible: true ONLY if the entire workflow happens inside a web browser.
- shareRecommendation.scope: "browser" if entirely in a browser tab; "window" if a single desktop app; "screen" if it spans multiple windows.

If this page is NOT a how-to / instructional article — opinion piece, news, marketing landing page, social post, abstract reference with no user workflow — set the "refusal" field to a one-sentence reason and omit the other fields. Check before generating any steps.`;

const MAX_PAGE_BYTES = 400_000; // ~400KB of HTML
const MAX_TEXT_CHARS = 60_000; // hard cap text we send to Gemini

const PAGE_URL_REGEX = /^https?:\/\//i;

interface PageRequestBody {
  pageUrl?: string;
  pageTitle?: string;
  contentText?: string;
}

async function handlePage(request: Request, env: Env): Promise<Response> {
  // Part of the anonymous trial so the side panel can build a tutorial
  // from the current page's content (sent as contentText). Rate-limited.
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  if (!env.GEMINI_API_KEY) {
    return new Response(
      JSON.stringify({ error: "GEMINI_API_KEY is not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  let body: PageRequestBody;
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Body must be JSON: { pageUrl, pageTitle?, contentText? }" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const pageUrl = (body.pageUrl || "").trim();
  if (!PAGE_URL_REGEX.test(pageUrl)) {
    return new Response(
      JSON.stringify({ error: "pageUrl must be an http(s) URL." }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  // Use caller-supplied content if present; otherwise fetch and extract.
  let pageTitle = (body.pageTitle || "").trim();
  let pageText = (body.contentText || "").trim();

  if (!pageText) {
    try {
      const res = await fetch(pageUrl, {
        headers: {
          "user-agent":
            "Mozilla/5.0 (compatible; ClickyProxy/1.0; +https://claude.com/code)",
          accept: "text/html,application/xhtml+xml,*/*;q=0.8",
        },
        redirect: "follow",
        cf: { cacheTtl: 300, cacheEverything: false } as any,
      });
      if (!res.ok) {
        return new Response(
          JSON.stringify({
            error: `Could not fetch the page (HTTP ${res.status}). It may require sign-in or block automated fetches.`,
          }),
          { status: 502, headers: { "content-type": "application/json" } }
        );
      }
      const reader = res.body?.getReader();
      if (!reader) {
        return new Response(
          JSON.stringify({ error: "Empty response when fetching the page." }),
          { status: 502, headers: { "content-type": "application/json" } }
        );
      }
      // Stream and cap at MAX_PAGE_BYTES.
      const chunks: Uint8Array[] = [];
      let total = 0;
      while (total < MAX_PAGE_BYTES) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;
        const remaining = MAX_PAGE_BYTES - total;
        chunks.push(value.length > remaining ? value.slice(0, remaining) : value);
        total += value.length;
      }
      try { reader.cancel(); } catch { /* ignore */ }
      const html = new TextDecoder("utf-8", { fatal: false })
        .decode(concat(chunks));

      if (!pageTitle) {
        const m = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
        if (m) pageTitle = decodeEntities(stripTags(m[1])).trim();
      }
      pageText = htmlToText(html);
    } catch (err) {
      return new Response(
        JSON.stringify({
          error: `Could not fetch the page: ${String(err).slice(0, 200)}`,
        }),
        { status: 502, headers: { "content-type": "application/json" } }
      );
    }
  }

  if (pageText.length > MAX_TEXT_CHARS) {
    pageText = pageText.slice(0, MAX_TEXT_CHARS);
  }

  if (pageText.replace(/\s+/g, "").length < 200) {
    return new Response(
      JSON.stringify({
        plan: {
          refusal:
            "Not enough readable text on this page to extract a tutorial. The page may rely on dynamic content that loads after sign-in.",
        },
        pageUrl,
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }

  const userText = `Source URL: ${pageUrl}\nPage title: ${pageTitle || "(untitled)"}\n\n--- BEGIN PAGE TEXT ---\n${pageText}\n--- END PAGE TEXT ---`;

  const geminiBody = {
    contents: [
      { parts: [{ text: PAGE_PROMPT }, { text: userText }] },
    ],
    generationConfig: {
      temperature: 0.4,
      responseMimeType: "application/json",
      responseSchema: STEP_PLAN_SCHEMA,
    },
  };

  const geminiUrl =
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=` +
    encodeURIComponent(env.GEMINI_API_KEY);

  const response = await fetch(geminiUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(geminiBody),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`[/page] Gemini API error ${response.status}: ${errorBody}`);
    return new Response(
      JSON.stringify({
        error: "Gemini request failed",
        status: response.status,
        detail: safeJsonParse(errorBody),
      }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  let payload: any;
  try {
    payload = await response.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Gemini returned a non-JSON body" }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const text: string | undefined =
    payload?.candidates?.[0]?.content?.parts?.[0]?.text;

  if (!text) {
    console.error("[/page] Gemini returned no text candidate", payload);
    return new Response(
      JSON.stringify({ error: "Gemini returned no content", raw: payload }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const plan = safeJsonParse(text);
  if (!plan || typeof plan !== "object") {
    return new Response(
      JSON.stringify({ error: "Gemini returned non-JSON content", raw: text }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  return new Response(
    JSON.stringify({ plan, pageUrl, pageTitle }),
    { status: 200, headers: { "content-type": "application/json" } }
  );
}

function concat(chunks: Uint8Array[]): Uint8Array {
  let len = 0;
  for (const c of chunks) len += c.length;
  const out = new Uint8Array(len);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

function stripTags(s: string): string {
  return s.replace(/<[^>]+>/g, " ");
}

function decodeEntities(s: string): string {
  return s
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCharCode(parseInt(n, 16)));
}

/* -------------------------------------------------------------------------- */
/*  Knowledge — proxy for the side panel's "research before generating" pass  */
/*                                                                            */
/*  Forwards to the Supabase Edge Function `search-knowledge`, which uses     */
/*  Perplexity to pull current docs / best practices for a task title. The   */
/*  function dedupes + caches in supabase by taskKey, so refresh < search.   */
/* -------------------------------------------------------------------------- */

// Gemini web-search research for tutorial planning. Replaces Perplexity at
// the tutorial-creation research step: Gemini grounds on a live Google Search,
// so it reports the CURRENT product UI / paths / gotchas the planner needs to
// navigate correctly. Returns { ok, knowledge_text, sources } — same shape the
// client expects from /knowledge, so it's a drop-in.
async function handleResearch(request: Request, env: Env): Promise<Response> {
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status, headers: { "content-type": "application/json" },
    });
  }
  if (!env.GEMINI_API_KEY) {
    return new Response(JSON.stringify({ error: "GEMINI_API_KEY is not configured" }), {
      status: 500, headers: { "content-type": "application/json" },
    });
  }
  let body: { query?: string; taskTitle?: string; model?: string };
  try { body = await request.json(); } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }
  const query = String(body.query || body.taskTitle || "").trim().slice(0, 600);
  if (!query) {
    return new Response(JSON.stringify({ error: "query is required" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }
  // gemini-2.5-flash grounds reliably with google_search (it actually issues
  // searches ~every call). gemini-flash-latest was intermittent — it answered
  // from training memory most of the time, which defeats the point of a live
  // search. Measured: 2.5-flash grounded 6/6 vs flash-latest ~1/4.
  const model = String(body.model || "gemini-2.5-flash");
  const prompt = `Using web search, research the CURRENT, up-to-date way to do the task below and write a tight reference brief for an AI that will guide a user through it step by step in the live product UI.

You MUST perform at least one web search before answering and base the brief on what those current results say — do not answer from memory alone, since product UIs change often.

Task: ${query}

Find and report, concretely and specific to the real product as it exists NOW:
- the exact UI path to the feature: which menu/area it lives in, the real button/menu labels, and crucially whether it's tucked inside a "More" / "..." / overflow / settings menu rather than a top-level button.
- any recent UI changes, renames, or relocations (a feature that moved or was folded into another).
- distinct-but-similar features that must NOT be confused (e.g. a "custom GPT" is NOT a "Project"; a "Gem" is its own thing) — name the right one and how to tell them apart.
- prerequisites and gotchas: plan-tier / paywall requirements, things that must be set up first, common mistakes.

Be concrete and current. Output a concise brief only — no preamble, no "I searched for…", no inline citations.`;

  const reqBody = {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    tools: [{ google_search: {} }],
    // Headroom so the brief isn't truncated mid-sentence: Gemini's hidden
    // thinking tokens are drawn from maxOutputTokens too, and grounding adds
    // more. 1800 was too low (briefs cut off at "...Version History: Tucked
    // inside"). This is a cap, not a target — generation still stops when the
    // brief is done, so it doesn't add latency for short answers.
    generationConfig: { temperature: 0.2, maxOutputTokens: 4096 },
  };
  // google_search grounding is model-decided and intermittent: the same query
  // sometimes searches the live web (returns sources) and sometimes answers
  // from training memory (no groundingMetadata at all). Since the whole point
  // is CURRENT product UI, retry once when the first attempt didn't ground —
  // keep the best (grounded) result, and fall back to the memory answer only
  // if every attempt declined to search. This retry is hidden behind the
  // clarifying-questions step, so the occasional extra round-trip is free.
  const MAX_ATTEMPTS = 2;
  let text = "";
  let sources: Array<{ url: string; title: string }> = [];
  let grounded = false;
  let lastErr = "";
  for (let i = 0; i < MAX_ATTEMPTS; i++) {
    let upstream: Response;
    try {
      upstream = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
        {
          method: "POST",
          headers: { "content-type": "application/json", "x-goog-api-key": env.GEMINI_API_KEY },
          body: JSON.stringify(reqBody),
        }
      );
    } catch (e) { lastErr = String(e); continue; }
    if (!upstream.ok) {
      lastErr = await upstream.text();
      console.error(`[/research] Gemini ${upstream.status}: ${lastErr}`);
      continue;
    }
    const data = await upstream.json().catch(() => null) as any;
    const cand = data && data.candidates && data.candidates[0];
    const t = cand && cand.content && Array.isArray(cand.content.parts)
      ? cand.content.parts.map((p: any) => (p && p.text) || "").join("").trim()
      : "";
    const gm = (cand && cand.groundingMetadata) || null;
    const chunks = (gm && gm.groundingChunks) || [];
    const srcs = chunks
      .map((c: any) => (c && c.web && c.web.uri) ? { url: c.web.uri, title: c.web.title || "" } : null)
      .filter(Boolean)
      .slice(0, 8) as Array<{ url: string; title: string }>;
    // Prefer a grounded result; otherwise keep the first non-empty answer as
    // a fallback so we still return something if no attempt grounds.
    if (srcs.length) {
      text = t; sources = srcs; grounded = true;
      break;
    }
    if (t && !text) text = t;
  }
  if (!text && lastErr) {
    return new Response(JSON.stringify({ error: "research failed", detail: lastErr.slice(0, 300) }), {
      status: 502, headers: { "content-type": "application/json" },
    });
  }
  return new Response(JSON.stringify({ ok: true, knowledge_text: text, sources, grounded }), {
    headers: { "content-type": "application/json" },
  });
}

async function handleKnowledge(request: Request, env: Env): Promise<Response> {
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  // search-knowledge's gateway rejects user JWTs (ES256) and our
  // service-role JWT, so we send the anon key as bearer (same trick the
  // web app uses). The function uses service-role internally, so the
  // bearer is just gateway auth.
  const anonKey = env.SUPABASE_ANON_KEY;
  if (!env.SUPABASE_URL || !anonKey) {
    return new Response(
      JSON.stringify({ error: "Supabase not configured (missing anon key)" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  let body: any;
  try { body = await request.json(); } catch {
    return new Response(
      JSON.stringify({ error: "Body must be JSON: { taskTitle, taskSteps?, taskKey?, mode? }" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const taskTitle = String(body.taskTitle || "").trim().slice(0, 240);
  if (!taskTitle) {
    return new Response(
      JSON.stringify({ error: "taskTitle is required" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }
  const taskSteps = Array.isArray(body.taskSteps)
    ? body.taskSteps.slice(0, 30).map((s: any) => String(s || "").slice(0, 300))
    : [];
  const taskKey = String(
    body.taskKey ||
      ("custom_" + taskTitle.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, ""))
  ).slice(0, 200);
  const allowedModes = new Set(["search", "refresh", "peek"]);
  const mode = allowedModes.has(String(body.mode)) ? String(body.mode) : "search";

  try {
    const upstream = await fetch(`${env.SUPABASE_URL}/functions/v1/search-knowledge`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        apikey: anonKey,
        authorization: `Bearer ${anonKey}`,
      },
      body: JSON.stringify({ taskTitle, taskSteps, taskKey, mode }),
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Knowledge upstream failed: " + String(err).slice(0, 200) }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }
}

/* -------------------------------------------------------------------------- */
/*               Coach: Knowledge (per-tool reference content)                */
/* -------------------------------------------------------------------------- */

async function handleCoachKnowledge(request: Request, env: Env): Promise<Response> {
  // Runs from third-party target tabs and feeds tool-specific knowledge
  // into prompt coaching. Accept the anon token so coaching gets its
  // context even when the user isn't signed in on this origin. Read-only,
  // no per-user data.
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_ROLE_KEY) {
    return new Response(
      JSON.stringify({ error: "Supabase not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  let body: { tool?: string };
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Body must be JSON: { tool }" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }
  const tool = String(body.tool || "").slice(0, 64);
  if (!tool) {
    return new Response(
      JSON.stringify({ error: "tool is required" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const supaHeaders = {
    apikey: env.SUPABASE_SERVICE_ROLE_KEY,
    authorization: `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
  };

  const [chunksRes, guidesRes] = await Promise.all([
    fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_knowledge_chunks?tool_id=eq.${encodeURIComponent(tool)}&select=title,body,position&order=position.asc`,
      { headers: supaHeaders }
    ),
    fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_guides?tool_id=eq.${encodeURIComponent(tool)}&select=id,title,summary,difficulty,duration_minutes,steps,position,kind&order=position.asc`,
      { headers: supaHeaders }
    ),
  ]);

  if (!chunksRes.ok || !guidesRes.ok) {
    const detail = await Promise.all([chunksRes.text(), guidesRes.text()]);
    console.error("[/coach/knowledge] Supabase error", chunksRes.status, guidesRes.status, detail);
    return new Response(
      JSON.stringify({ error: "Could not fetch knowledge", detail }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const chunks: any[] = await chunksRes.json();
  const guidesRaw: any[] = await guidesRes.json();

  const knowledge = chunks.map((r) => ({
    title: String(r.title || ""),
    body: String(r.body || ""),
  }));

  const guides = guidesRaw.map((g) => ({
    id: String(g.id),
    title: String(g.title || ""),
    summary: String(g.summary || ""),
    difficulty: String(g.difficulty || "beginner"),
    durationMinutes: Number(g.duration_minutes) || 5,
    steps: Array.isArray(g.steps) ? g.steps.map((s: any) => String(s)) : [],
    kind: g.kind === "smart" ? "smart" : "basic",
  }));

  return new Response(
    JSON.stringify({ ok: true, tool, knowledge, guides }),
    {
      status: 200,
      headers: {
        "content-type": "application/json",
        // Brief cache so repeated launcher boots don't hammer Supabase.
        "cache-control": "public, max-age=300",
      },
    }
  );
}

/* -------------------------------------------------------------------------- */
/*                          Coach: Ask (Q&A on a tool)                        */
/* -------------------------------------------------------------------------- */

interface CoachKnowledgeChunk {
  title?: string;
  body?: string;
}
interface CoachAskBody {
  tool?: string;
  toolLabel?: string;
  question?: string;
  knowledge?: CoachKnowledgeChunk[];
}

async function handleCoachAsk(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  if (!env.ANTHROPIC_API_KEY) {
    return new Response(
      JSON.stringify({ error: "ANTHROPIC_API_KEY is not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  let body: CoachAskBody;
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Body must be JSON: { tool, question, knowledge? }" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const tool = String(body.tool || "").slice(0, 64);
  const toolLabel = String(body.toolLabel || tool || "an AI tool").slice(0, 64);
  const question = String(body.question || "").trim().slice(0, 2000);
  const rawKnowledge = Array.isArray(body.knowledge)
    ? body.knowledge.slice(0, 12)
    : [];

  if (!question) {
    return new Response(
      JSON.stringify({ error: "question is required" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const knowledgeText = rawKnowledge
    .filter(
      (k): k is { title: string; body: string } =>
        !!k && typeof k.title === "string" && typeof k.body === "string"
    )
    .map((k) => `### ${k.title.slice(0, 120)}\n${k.body.slice(0, 1200)}`)
    .join("\n\n");

  const systemPrompt = `You are Glide, an ambient coach helping someone use ${toolLabel}.

Style:
- Plain language, second person, warm and direct.
- Lead with the answer; explain why only if it helps the user act.
- Keep answers under 180 words unless the user explicitly asks for depth.
- If you don't know, say what you'd need to know to help further. Never invent features.
- Speak like a friend who's done this before, not a manual.

Paste-able prompts:
- When you suggest a prompt the user should literally paste into ${toolLabel}, wrap it in a markdown code fence with the language 'prompt', like:
  \`\`\`prompt
  The actual prompt, written in clear plain language with all the context filled in.
  \`\`\`
- Only use \`\`\`prompt fences for text the user should paste verbatim into ${toolLabel}. Do NOT wrap general explanation, tips, or shell commands in prompt fences.
- Prefer one short paragraph of explanation followed by the prompt block. The user can click the block to paste.

Use the reference knowledge below when it's relevant. The knowledge is hand-curated and trustworthy. If the user's question isn't covered there, fall back to general best practices for ${toolLabel}.`;

  const userMessage = knowledgeText
    ? `Reference knowledge for ${toolLabel}:\n\n${knowledgeText}\n\n---\n\nQuestion: ${question}`
    : `Question (about ${toolLabel}): ${question}`;

  const claudeRes = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5",
      max_tokens: 800,
      system: systemPrompt,
      messages: [{ role: "user", content: userMessage }],
    }),
  });

  if (!claudeRes.ok) {
    const errBody = await claudeRes.text();
    console.error(`[/coach/ask] Anthropic error ${claudeRes.status}: ${errBody}`);
    return new Response(
      JSON.stringify({
        error: "Claude request failed",
        status: claudeRes.status,
        detail: errBody.slice(0, 500),
      }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  let claudeData: any;
  try {
    claudeData = await claudeRes.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Claude returned non-JSON body" }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const parts = Array.isArray(claudeData?.content) ? claudeData.content : [];
  const answer = parts
    .filter((p: any) => p && p.type === "text" && typeof p.text === "string")
    .map((p: any) => p.text)
    .join("\n")
    .trim();

  if (!answer) {
    return new Response(
      JSON.stringify({ error: "Claude returned no text content", raw: claudeData }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  return new Response(
    JSON.stringify({ ok: true, answer }),
    { status: 200, headers: { "content-type": "application/json" } }
  );
}

/* -------------------------------------------------------------------------- */
/*               Coach Chat — multi-turn, optionally page-aware              */
/* -------------------------------------------------------------------------- */

interface CoachChatMessage {
  role?: string;       // 'user' | 'assistant'
  content?: string;
}

interface CoachChatPage {
  url?: string;
  title?: string;
  tree?: string;       // pre-built accessibility outline (built client-side)
}

interface CoachChatBody {
  tool?: string;
  toolLabel?: string;
  messages?: CoachChatMessage[];
  knowledge?: { title?: string; body?: string }[];
  page?: CoachChatPage | null;
}

const PAGE_TREE_LIMIT = 18_000; // chars — keeps a hefty AX tree but caps the bill.

async function handleCoachChat(request: Request, env: Env): Promise<Response> {
  // The puck's "ask Glide" chat runs in target tabs — accept the anon
  // token like the other coach endpoints so it works without a same-
  // origin Supabase session. Stateless; no per-user data written.
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  if (!env.ANTHROPIC_API_KEY) {
    return new Response(
      JSON.stringify({ error: "ANTHROPIC_API_KEY is not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  let body: CoachChatBody;
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Body must be JSON: { tool, messages, knowledge?, page? }" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const tool = String(body.tool || "").slice(0, 64);
  const toolLabel = String(body.toolLabel || tool || "an AI tool").slice(0, 64);
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  const messages = rawMessages
    .filter((m) => m && (m.role === "user" || m.role === "assistant"))
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: String(m.content || "").slice(0, 4000),
    }))
    .filter((m) => m.content.length > 0)
    .slice(-20); // keep last 20 turns

  if (!messages.length || messages[messages.length - 1].role !== "user") {
    return new Response(
      JSON.stringify({ error: "messages must end with a user turn" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const rawKnowledge = Array.isArray(body.knowledge) ? body.knowledge.slice(0, 12) : [];
  const knowledgeText = rawKnowledge
    .filter(
      (k): k is { title: string; body: string } =>
        !!k && typeof k.title === "string" && typeof k.body === "string"
    )
    .map((k) => `### ${k.title.slice(0, 120)}\n${k.body.slice(0, 1200)}`)
    .join("\n\n");

  const page = body.page || null;
  const pageUrl = page && typeof page.url === "string" ? page.url.slice(0, 400) : "";
  const pageTitle = page && typeof page.title === "string" ? page.title.slice(0, 200) : "";
  const pageTree =
    page && typeof page.tree === "string"
      ? page.tree.slice(0, PAGE_TREE_LIMIT)
      : "";

  const seePageGuidance = pageTree
    ? `

The user has "view page" turned on, so you can see what they're currently looking at. Below is an accessibility-tree outline of the page in their browser. Use it to ground specific questions ("what's that error on this screen?", "where is the deploy button?") in what they're actually seeing. Don't recite the tree — refer to specific UI elements by their visible label.

Current page: ${pageTitle || "(untitled)"} — ${pageUrl}
Accessibility tree:
\`\`\`
${pageTree}
\`\`\`
`
    : `

The user has "view page" turned OFF, so you cannot see what's currently on their screen. If a question is about a specific element on their page ("what does this button do?"), ask them to turn view-page on or describe what they're seeing.`;

  const systemPrompt = `You are Glide, an ambient coach helping someone use ${toolLabel}. You converse with the user in a chat panel that sits inside their ${toolLabel} tab.

Style — keep replies SHORT:
- Default to 1-3 sentences. Hard cap: 80 words unless the user explicitly says "explain more" or "in depth".
- Lead with the answer. Skip preamble like "Great question" or "I'd be happy to". No restating the question.
- One idea per reply. If the user asks two things, answer the main one and offer to cover the other next.
- Plain language, second person, warm and direct. Talk like a friend who's done this before, not a manual.
- Reference earlier turns when relevant. If you don't know, say what you'd need.

Formatting:
- Markdown is rendered: **bold**, *italic*, \`code\`, [links](https://…), and short bullet/numbered lists are supported. Use them sparingly.
- Use bullets only when listing 2+ distinct items; never use a bullet for a single point.
- No headings unless the user asked for a multi-section explanation.

Paste-able prompts:
- When you suggest a prompt the user should literally paste into ${toolLabel}, wrap it in a markdown code fence with the language 'prompt':
  \`\`\`prompt
  The actual prompt, written in clear plain language.
  \`\`\`
- Only use \`\`\`prompt fences for text the user should paste verbatim. Do NOT wrap explanation or shell commands in prompt fences.

Knowledge:
- Reference knowledge below is hand-curated and trustworthy — use it when relevant.
- If the question isn't covered there, fall back to general best practices for ${toolLabel}. Never invent features.${seePageGuidance}`;

  const knowledgePrefix = knowledgeText
    ? `Reference knowledge for ${toolLabel}:\n\n${knowledgeText}\n\n---\n\n`
    : "";

  // Inject the knowledge prefix into the FIRST user message so Claude sees it
  // once, regardless of how long the conversation gets. (Anthropic doesn't
  // accept system + injected prefix as separate messages; this is the
  // standard pattern.)
  const claudeMessages = messages.map((m, i) =>
    i === 0 && m.role === "user" && knowledgePrefix
      ? { role: m.role, content: knowledgePrefix + m.content }
      : m
  );

  const claudeRes = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5",
      max_tokens: 350, // short replies — 80-word cap + a paste-able prompt at most
      system: systemPrompt,
      messages: claudeMessages,
    }),
  });

  if (!claudeRes.ok) {
    const errBody = await claudeRes.text();
    console.error(`[/coach/chat] Anthropic error ${claudeRes.status}: ${errBody}`);
    return new Response(
      JSON.stringify({
        error: "Claude request failed",
        status: claudeRes.status,
        detail: errBody.slice(0, 500),
      }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  let claudeData: any;
  try {
    claudeData = await claudeRes.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Claude returned non-JSON body" }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const parts = Array.isArray(claudeData?.content) ? claudeData.content : [];
  const answer = parts
    .filter((p: any) => p && p.type === "text" && typeof p.text === "string")
    .map((p: any) => p.text)
    .join("\n")
    .trim();

  return new Response(
    JSON.stringify({
      ok: true,
      answer,
      sawPage: !!pageTree,
      usage: claudeData?.usage || null,
    }),
    { status: 200, headers: { "content-type": "application/json" } }
  );
}

/* -------------------------------------------------------------------------- */
/*               Coach Library: CRUD over tool_knowledge + tool_guides        */
/* -------------------------------------------------------------------------- */

// Validates the caller's Supabase access token. Any signed-in Glide user
// can read/write — we trust that anyone with a valid token in this account
// is allowed to author content. (Tighten to admin-flagged users later if
// needed.)
// Best-effort variant — returns the userId if the token is valid, or
// null if it isn't / isn't present. Used by the top-level rate limiter
// so signed-in callers share one budget across devices, but anonymous
// boot routes (extension) still fall back to per-IP keys.
async function tryReadUser(request: Request, env: Env): Promise<{ userId: string | null }> {
  const auth = request.headers.get("authorization") || "";
  const token = auth.replace(/^Bearer\s+/i, "").trim();
  if (!token || !env.SUPABASE_URL) return { userId: null };
  try {
    const r = await fetch(`${env.SUPABASE_URL}/auth/v1/user`, {
      headers: {
        apikey: env.SUPABASE_SERVICE_ROLE_KEY,
        authorization: `Bearer ${token}`,
      },
    });
    if (!r.ok) return { userId: null };
    const data = (await r.json()) as { id?: string };
    return { userId: data?.id || null };
  } catch {
    return { userId: null };
  }
}

// ---------------------------------------------------------------------
// Anonymous-trial rate limiter. Keyed by x-glide-anon-token (a client-
// minted UUID stored in chrome.storage or cookie). In-memory map per
// worker instance — good enough as a safety net because (a) the welcome
// page only exposes ONE bundled demo tutorial, capping intended use,
// and (b) Workers tend to live for many requests so the same instance
// catches abuse within its window. Tighten to KV/DO if we ever need a
// hard cross-instance limit.
type AnonRateEntry = { count: number; windowStart: number };
const anonRateMap = new Map<string, AnonRateEntry>();
const ANON_RATE_WINDOW_MS = 60 * 60 * 1000;   // 1 hour
// 40 was sized for a single short demo and was too low for a real tutorial —
// both engines call the model per page-change, so an active session blew the
// budget in minutes (the stress test logged 52 and 139 rate-limit errors).
// 150/hour comfortably covers a couple of full tutorials per installation;
// the access-code gate + per-installation token keep accidental runaway and
// casual abuse bounded. Signed-in users use the higher per-user route limits.
const ANON_RATE_MAX = 150;

function checkAnonRate(token: string): { ok: boolean; reset: number; remaining: number } {
  const now = Date.now();
  const entry = anonRateMap.get(token);
  if (!entry || now - entry.windowStart > ANON_RATE_WINDOW_MS) {
    anonRateMap.set(token, { count: 1, windowStart: now });
    return { ok: true, reset: now + ANON_RATE_WINDOW_MS, remaining: ANON_RATE_MAX - 1 };
  }
  if (entry.count >= ANON_RATE_MAX) {
    return { ok: false, reset: entry.windowStart + ANON_RATE_WINDOW_MS, remaining: 0 };
  }
  entry.count += 1;
  return { ok: true, reset: entry.windowStart + ANON_RATE_WINDOW_MS, remaining: ANON_RATE_MAX - entry.count };
}

// Accept either a real Supabase bearer (logged-in user) OR an
// `x-glide-anon-token` header (anonymous trial). Endpoints that opt in
// to anon use this in place of requireSupabaseUser.
async function requireAuthOrAnon(request: Request, env: Env):
  Promise<{ ok: true; userId: string | null; anon: boolean } | { ok: false; status: number; error: string }> {
  const bearer = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "").trim();
  if (bearer) {
    const r = await requireSupabaseUser(request, env);
    if (r.ok) return { ok: true, userId: r.userId, anon: false };
    return r;
  }
  const anonToken = (request.headers.get("x-glide-anon-token") || "").trim();
  // UUID-ish minimum length to keep noisy headers out of the rate map.
  if (!anonToken || anonToken.length < 16 || anonToken.length > 64) {
    return { ok: false, status: 401, error: "Missing bearer or anon token" };
  }
  const rate = checkAnonRate(anonToken);
  if (!rate.ok) {
    return { ok: false, status: 429, error: "Anonymous rate limit exceeded" };
  }
  return { ok: true, userId: null, anon: true };
}

async function requireSupabaseUser(request: Request, env: Env):
  Promise<{ ok: true; userId: string } | { ok: false; status: number; error: string }> {
  const auth = request.headers.get("authorization") || "";
  const token = auth.replace(/^Bearer\s+/i, "").trim();
  if (!token) return { ok: false, status: 401, error: "Missing bearer token" };
  if (!env.SUPABASE_URL) return { ok: false, status: 500, error: "Supabase not configured" };
  try {
    const r = await fetch(`${env.SUPABASE_URL}/auth/v1/user`, {
      headers: {
        apikey: env.SUPABASE_SERVICE_ROLE_KEY,
        authorization: `Bearer ${token}`,
      },
    });
    if (!r.ok) return { ok: false, status: 401, error: "Invalid or expired token" };
    const user = await r.json() as any;
    if (!user || !user.id) return { ok: false, status: 401, error: "No user" };
    return { ok: true, userId: user.id };
  } catch (err) {
    return { ok: false, status: 502, error: "Could not verify token" };
  }
}

function supaHeaders(env: Env) {
  return {
    apikey: env.SUPABASE_SERVICE_ROLE_KEY,
    authorization: `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
    "content-type": "application/json",
  };
}

async function handleLibraryTools(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  // Aggregate counts per tool_id from both tables.
  const [chunksRes, guidesRes] = await Promise.all([
    fetch(`${env.SUPABASE_URL}/rest/v1/tool_knowledge_chunks?select=tool_id`, { headers: supaHeaders(env) }),
    fetch(`${env.SUPABASE_URL}/rest/v1/tool_guides?select=tool_id`, { headers: supaHeaders(env) }),
  ]);
  if (!chunksRes.ok || !guidesRes.ok) {
    return new Response(JSON.stringify({ error: "Could not list tools" }), {
      status: 502, headers: { "content-type": "application/json" },
    });
  }
  const chunks: Array<{ tool_id: string }> = await chunksRes.json();
  const guides: Array<{ tool_id: string }> = await guidesRes.json();
  const map = new Map<string, { tool_id: string; chunkCount: number; guideCount: number }>();
  for (const c of chunks) {
    if (!c.tool_id) continue;
    const cur = map.get(c.tool_id) || { tool_id: c.tool_id, chunkCount: 0, guideCount: 0 };
    cur.chunkCount++;
    map.set(c.tool_id, cur);
  }
  for (const g of guides) {
    if (!g.tool_id) continue;
    const cur = map.get(g.tool_id) || { tool_id: g.tool_id, chunkCount: 0, guideCount: 0 };
    cur.guideCount++;
    map.set(g.tool_id, cur);
  }
  const tools = Array.from(map.values()).sort((a, b) => a.tool_id.localeCompare(b.tool_id));
  return new Response(JSON.stringify({ ok: true, tools }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

async function handleLibraryList(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status, headers: { "content-type": "application/json" },
    });
  }
  const url = new URL(request.url);
  const tool = (url.searchParams.get("tool") || "").trim();
  if (!tool) {
    return new Response(JSON.stringify({ error: "?tool=… is required" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }
  const [chunksRes, guidesRes] = await Promise.all([
    fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_knowledge_chunks?tool_id=eq.${encodeURIComponent(tool)}&select=id,tool_id,title,body,position,updated_at&order=position.asc`,
      { headers: supaHeaders(env) }
    ),
    fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_guides?tool_id=eq.${encodeURIComponent(tool)}&select=id,tool_id,title,summary,difficulty,duration_minutes,steps,position,kind,updated_at&order=position.asc`,
      { headers: supaHeaders(env) }
    ),
  ]);
  if (!chunksRes.ok || !guidesRes.ok) {
    return new Response(JSON.stringify({ error: "Could not list content" }), {
      status: 502, headers: { "content-type": "application/json" },
    });
  }
  const knowledge = await chunksRes.json();
  const guides = await guidesRes.json();
  return new Response(
    JSON.stringify({ ok: true, tool, knowledge, guides }),
    { status: 200, headers: { "content-type": "application/json" } }
  );
}

interface ChunkUpsertBody {
  id?: string;
  tool_id?: string;
  title?: string;
  body?: string;
  position?: number;
}

async function handleLibraryChunk(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  if (request.method === "DELETE") {
    let body: { id?: string };
    try { body = await request.json(); } catch { body = {}; }
    const id = String(body.id || "").trim();
    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400, headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_knowledge_chunks?id=eq.${encodeURIComponent(id)}`,
      { method: "DELETE", headers: { ...supaHeaders(env), prefer: "return=minimal" } }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Delete failed", detail }), {
        status: 502, headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200, headers: { "content-type": "application/json" },
    });
  }

  // POST: create or update
  let body: ChunkUpsertBody;
  try { body = await request.json(); } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }
  const tool_id = String(body.tool_id || "").trim().slice(0, 64);
  const title = String(body.title || "").trim().slice(0, 200);
  const text = String(body.body || "").trim().slice(0, 4000);
  const position = Number.isFinite(Number(body.position)) ? Number(body.position) : 0;
  if (!tool_id || !title || !text) {
    return new Response(JSON.stringify({ error: "tool_id, title, body are required" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }

  if (body.id) {
    // Update
    const id = String(body.id).trim();
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_knowledge_chunks?id=eq.${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { ...supaHeaders(env), prefer: "return=representation" },
        body: JSON.stringify({ tool_id, title, body: text, position, updated_at: new Date().toISOString() }),
      }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Update failed", detail }), {
        status: 502, headers: { "content-type": "application/json" },
      });
    }
    const rows = await r.json();
    return new Response(JSON.stringify({ ok: true, chunk: rows[0] || null }), {
      status: 200, headers: { "content-type": "application/json" },
    });
  }

  // Create
  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/tool_knowledge_chunks`, {
    method: "POST",
    headers: { ...supaHeaders(env), prefer: "return=representation" },
    body: JSON.stringify({ tool_id, title, body: text, position }),
  });
  if (!r.ok) {
    const detail = await r.text();
    return new Response(JSON.stringify({ error: "Create failed", detail }), {
      status: 502, headers: { "content-type": "application/json" },
    });
  }
  const rows = await r.json();
  return new Response(JSON.stringify({ ok: true, chunk: rows[0] || null }), {
    status: 200, headers: { "content-type": "application/json" },
  });
}

interface GuideUpsertBody {
  id?: string;
  tool_id?: string;
  title?: string;
  summary?: string;
  difficulty?: string;
  duration_minutes?: number;
  steps?: string[];
  position?: number;
  kind?: string; // 'basic' | 'smart'
}

async function handleLibraryGuide(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status, headers: { "content-type": "application/json" },
    });
  }
  if (request.method === "DELETE") {
    let body: { id?: string };
    try { body = await request.json(); } catch { body = {}; }
    const id = String(body.id || "").trim();
    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400, headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_guides?id=eq.${encodeURIComponent(id)}`,
      { method: "DELETE", headers: { ...supaHeaders(env), prefer: "return=minimal" } }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Delete failed", detail }), {
        status: 502, headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200, headers: { "content-type": "application/json" },
    });
  }

  // POST: create or update
  let body: GuideUpsertBody;
  try { body = await request.json(); } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }
  const tool_id = String(body.tool_id || "").trim().slice(0, 64);
  const title = String(body.title || "").trim().slice(0, 120);
  const summary = String(body.summary || "").trim().slice(0, 500);
  const difficulty = String(body.difficulty || "beginner").trim().slice(0, 32);
  const duration_minutes = Number.isFinite(Number(body.duration_minutes))
    ? Math.max(1, Math.min(180, Number(body.duration_minutes)))
    : 5;
  const stepsRaw = Array.isArray(body.steps) ? body.steps : [];
  const steps = stepsRaw
    .map((s) => String(s || "").trim())
    .filter(Boolean)
    .slice(0, 40)
    .map((s) => s.slice(0, 1500));
  const position = Number.isFinite(Number(body.position)) ? Number(body.position) : 0;
  const kindRaw = String(body.kind || "basic").trim().toLowerCase();
  const kind = kindRaw === "smart" ? "smart" : "basic";
  if (!tool_id || !title || !steps.length) {
    return new Response(JSON.stringify({ error: "tool_id, title, and at least one step are required" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }

  if (body.id) {
    const id = String(body.id).trim();
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_guides?id=eq.${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { ...supaHeaders(env), prefer: "return=representation" },
        body: JSON.stringify({
          tool_id, title, summary, difficulty, duration_minutes, steps, position, kind,
          updated_at: new Date().toISOString(),
        }),
      }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Update failed", detail }), {
        status: 502, headers: { "content-type": "application/json" },
      });
    }
    const rows = await r.json();
    return new Response(JSON.stringify({ ok: true, guide: rows[0] || null }), {
      status: 200, headers: { "content-type": "application/json" },
    });
  }

  // Create — pk is the guide id (string). Generate one if not provided.
  const id = `${tool_id}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/tool_guides`, {
    method: "POST",
    headers: { ...supaHeaders(env), prefer: "return=representation" },
    body: JSON.stringify({
      id, tool_id, title, summary, difficulty, duration_minutes, steps, position, kind,
    }),
  });
  if (!r.ok) {
    const detail = await r.text();
    return new Response(JSON.stringify({ error: "Create failed", detail }), {
      status: 502, headers: { "content-type": "application/json" },
    });
  }
  const rows = await r.json();
  return new Response(JSON.stringify({ ok: true, guide: rows[0] || null }), {
    status: 200, headers: { "content-type": "application/json" },
  });
}

/* -------------------------------------------------------------------------- */
/*           Prompt-coaching rubrics — defaults + per-tool overrides          */
/* -------------------------------------------------------------------------- */

interface RubricRow {
  rule_id: string;
  summary: string;
  fires_on: string[];
  silent_on: string[];
  rewrite_style: string[];
  cadence: string;
  custom_instructions: string;
  is_override?: boolean;
  updated_at?: string;
}

// The rule definitions Glide ships with. Per-tool override rows in
// tool_prompt_rubrics are layered on top — any field that's set in the
// override replaces the default; unset fields fall through.
const DEFAULT_RUBRICS: Record<string, RubricRow> = {
  "api-key-safety": {
    rule_id: "api-key-safety",
    summary:
      "Detects pasted credentials before they leak into source code or prompts. Runs entirely in the browser — no prompt text leaves your machine.",
    fires_on: [
      "Anthropic / OpenAI keys (sk-…)",
      "Google API keys (AIza…)",
      "Stripe keys (pk/sk/rk_live or _test_…)",
      "GitHub personal access tokens (ghp_…, github_pat_…)",
      "Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-, xoxs-)",
      "AWS access key IDs (AKIA…)",
      "JWTs (eyJ…)",
      "Supabase service role keys",
    ],
    silent_on: [],
    rewrite_style: [],
    cadence:
      "Instant. Fires the moment a matching pattern appears in the prompt input. Stays visible until you remove the key or dismiss the card.",
    custom_instructions: "",
  },
  "prompt-quality": {
    rule_id: "prompt-quality",
    summary:
      "Evaluates your draft via Claude Haiku ~800ms after you stop typing. If the prompt would meaningfully benefit from a sharper rewrite, surfaces a paste-able suggestion.",
    fires_on: [
      'Vague style language with no concrete description ("make it nicer", "more professional")',
      "Missing audience, constraints, or success criteria when those would change the AI's output",
      "Multiple distinct features asked at once (>2 independent asks) — single-feature prompts produce better output",
      "Demo data pasted without instruction (e.g. raw JSON with no task)",
      'Persistence implied ("save", "track", "remember") with no data model described',
    ],
    silent_on: [
      "Short, focused prompts that already name what they want",
      "Prompts that name a target outcome plus at least one concrete constraint",
      "Casual phrasing that's still clear",
      "Anything where the rewrite probably wouldn't materially improve output",
    ],
    rewrite_style: [
      "Preserves your intent and voice",
      "Adds the missing structure (audience, constraints, success criteria)",
      "Keeps the friendly conversational tone",
      "Aims for 30–120 words — long enough to be specific, short enough to skim",
    ],
    cadence:
      "Per-rule cooldown of 30 seconds after each dismissal. Skipped entirely on prompts under 25 characters or when the existing API-key rule has already matched.",
    custom_instructions: "",
  },
};

function defaultRubric(ruleId: string): RubricRow {
  const def = DEFAULT_RUBRICS[ruleId];
  if (!def) {
    return {
      rule_id: ruleId,
      summary: "",
      fires_on: [],
      silent_on: [],
      rewrite_style: [],
      cadence: "",
      custom_instructions: "",
    };
  }
  // Clone so callers can mutate freely.
  return JSON.parse(JSON.stringify(def));
}

function asStringArray(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.map((x) => String(x || "").trim()).filter(Boolean);
}

// Merge a DB row over the defaults. Any string-array field that comes back
// non-empty replaces the default; an empty array is treated as "use default"
// so admins don't accidentally erase the rubric by clearing all bullets.
function mergeRubricRow(ruleId: string, row: any): RubricRow {
  const merged = defaultRubric(ruleId);
  if (!row) return merged;
  if (typeof row.summary === "string" && row.summary.trim()) {
    merged.summary = row.summary;
  }
  const firesOn = asStringArray(row.fires_on);
  if (firesOn.length) merged.fires_on = firesOn;
  const silentOn = asStringArray(row.silent_on);
  if (silentOn.length) merged.silent_on = silentOn;
  const rewriteStyle = asStringArray(row.rewrite_style);
  if (rewriteStyle.length) merged.rewrite_style = rewriteStyle;
  if (typeof row.cadence === "string" && row.cadence.trim()) {
    merged.cadence = row.cadence;
  }
  if (typeof row.custom_instructions === "string") {
    merged.custom_instructions = row.custom_instructions;
  }
  merged.is_override = true;
  if (row.updated_at) merged.updated_at = row.updated_at;
  return merged;
}

async function loadRubricRow(
  env: Env,
  toolId: string,
  ruleId: string
): Promise<any | null> {
  const r = await fetch(
    `${env.SUPABASE_URL}/rest/v1/tool_prompt_rubrics?tool_id=eq.${encodeURIComponent(
      toolId
    )}&rule_id=eq.${encodeURIComponent(ruleId)}&select=*`,
    { headers: supaHeaders(env) }
  );
  if (!r.ok) return null;
  const rows = (await r.json()) as any[];
  return rows && rows[0] ? rows[0] : null;
}

async function handleLibraryRubric(
  request: Request,
  env: Env
): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  const url = new URL(request.url);

  if (request.method === "GET") {
    const tool = (url.searchParams.get("tool") || "").trim();
    if (!tool) {
      return new Response(JSON.stringify({ error: "?tool=… is required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_prompt_rubrics?tool_id=eq.${encodeURIComponent(
        tool
      )}&select=*`,
      { headers: supaHeaders(env) }
    );
    if (!r.ok) {
      return new Response(JSON.stringify({ error: "Could not load rubrics" }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    const rows = (await r.json()) as any[];
    const byRule = new Map<string, any>();
    for (const row of rows) {
      if (row && typeof row.rule_id === "string") byRule.set(row.rule_id, row);
    }
    const rubrics = Object.keys(DEFAULT_RUBRICS).map((ruleId) =>
      mergeRubricRow(ruleId, byRule.get(ruleId) || null)
    );
    return new Response(
      JSON.stringify({ ok: true, tool, rubrics }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }

  if (request.method === "DELETE") {
    let body: { tool_id?: string; rule_id?: string };
    try {
      body = await request.json();
    } catch {
      body = {};
    }
    const tool_id = String(body.tool_id || "").trim().slice(0, 64);
    const rule_id = String(body.rule_id || "").trim().slice(0, 64);
    if (!tool_id || !rule_id) {
      return new Response(
        JSON.stringify({ error: "tool_id and rule_id are required" }),
        { status: 400, headers: { "content-type": "application/json" } }
      );
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_prompt_rubrics?tool_id=eq.${encodeURIComponent(
        tool_id
      )}&rule_id=eq.${encodeURIComponent(rule_id)}`,
      { method: "DELETE", headers: { ...supaHeaders(env), prefer: "return=minimal" } }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(
        JSON.stringify({ error: "Reset failed", detail }),
        { status: 502, headers: { "content-type": "application/json" } }
      );
    }
    const merged = mergeRubricRow(rule_id, null);
    return new Response(JSON.stringify({ ok: true, rubric: merged }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  // POST → upsert
  let body: any;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  const tool_id = String(body.tool_id || "").trim().slice(0, 64);
  const rule_id = String(body.rule_id || "").trim().slice(0, 64);
  if (!tool_id || !rule_id) {
    return new Response(
      JSON.stringify({ error: "tool_id and rule_id are required" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }
  if (!DEFAULT_RUBRICS[rule_id]) {
    return new Response(
      JSON.stringify({ error: `Unknown rule_id: ${rule_id}` }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const summary = (typeof body.summary === "string" ? body.summary : "").slice(0, 1200);
  const cadence = (typeof body.cadence === "string" ? body.cadence : "").slice(0, 1200);
  const custom_instructions = (typeof body.custom_instructions === "string"
    ? body.custom_instructions
    : ""
  ).slice(0, 2000);
  const fires_on = asStringArray(body.fires_on).slice(0, 20).map((s) => s.slice(0, 400));
  const silent_on = asStringArray(body.silent_on).slice(0, 20).map((s) => s.slice(0, 400));
  const rewrite_style = asStringArray(body.rewrite_style).slice(0, 20).map((s) => s.slice(0, 400));

  // Upsert via on_conflict on the (tool_id, rule_id) unique index.
  const r = await fetch(
    `${env.SUPABASE_URL}/rest/v1/tool_prompt_rubrics?on_conflict=tool_id,rule_id`,
    {
      method: "POST",
      headers: {
        ...supaHeaders(env),
        prefer: "return=representation,resolution=merge-duplicates",
      },
      body: JSON.stringify({
        tool_id,
        rule_id,
        summary,
        fires_on,
        silent_on,
        rewrite_style,
        cadence,
        custom_instructions,
        updated_at: new Date().toISOString(),
      }),
    }
  );
  if (!r.ok) {
    const detail = await r.text();
    return new Response(
      JSON.stringify({ error: "Save failed", detail }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }
  const rows = await r.json();
  const merged = mergeRubricRow(rule_id, rows[0] || null);
  return new Response(JSON.stringify({ ok: true, rubric: merged }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/* -------------------------------------------------------------------------- */
/*                  Tool domains — per-tool hostname patterns                 */
/* -------------------------------------------------------------------------- */

async function handleLibraryDomainsAll(
  request: Request,
  env: Env
): Promise<Response> {
  // No auth — the extension is unauthenticated and needs this at boot.
  // The data is just hostname patterns (already visible in the bundled
  // extension anyway), and writes still require auth.
  if (request.method !== "GET") {
    return new Response(JSON.stringify({ error: "GET only" }), {
      status: 405,
      headers: { "content-type": "application/json" },
    });
  }
  const r = await fetch(
    `${env.SUPABASE_URL}/rest/v1/tool_domains?select=tool_id,kind,pattern&order=tool_id.asc,position.asc`,
    { headers: supaHeaders(env) }
  );
  if (!r.ok) {
    return new Response(JSON.stringify({ error: "Could not list domains" }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const domains = await r.json();
  return new Response(JSON.stringify({ ok: true, domains }), {
    status: 200,
    headers: {
      "content-type": "application/json",
      "cache-control": "public, max-age=300",
    },
  });
}

async function handleLibraryDomains(
  request: Request,
  env: Env
): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  const url = new URL(request.url);

  if (request.method === "GET") {
    const tool = (url.searchParams.get("tool") || "").trim();
    if (!tool) {
      return new Response(JSON.stringify({ error: "?tool=… is required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_domains?tool_id=eq.${encodeURIComponent(
        tool
      )}&select=*&order=position.asc`,
      { headers: supaHeaders(env) }
    );
    if (!r.ok) {
      return new Response(JSON.stringify({ error: "Could not list domains" }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    const domains = await r.json();
    return new Response(JSON.stringify({ ok: true, tool, domains }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  if (request.method === "DELETE") {
    let body: { id?: string };
    try {
      body = await request.json();
    } catch {
      body = {};
    }
    const id = String(body.id || "").trim();
    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_domains?id=eq.${encodeURIComponent(id)}`,
      { method: "DELETE", headers: { ...supaHeaders(env), prefer: "return=minimal" } }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Delete failed", detail }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  // POST → create/update
  let body: any;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  const tool_id = String(body.tool_id || "").trim().slice(0, 64);
  const kindRaw = String(body.kind || "substring").trim().toLowerCase();
  const kind = kindRaw === "regex" ? "regex" : "substring";
  const pattern = String(body.pattern || "").trim().slice(0, 400);
  const note = String(body.note || "").trim().slice(0, 400);
  const position = Number.isFinite(Number(body.position)) ? Number(body.position) : 0;

  if (!tool_id || !pattern) {
    return new Response(
      JSON.stringify({ error: "tool_id and pattern are required" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  if (kind === "regex") {
    try {
      // Validate the regex compiles. We don't actually use it here, but
      // failing fast in the editor beats a broken extension.
      new RegExp(pattern, "i");
    } catch (err: any) {
      return new Response(
        JSON.stringify({ error: `Invalid regex: ${err?.message || String(err)}` }),
        { status: 400, headers: { "content-type": "application/json" } }
      );
    }
  }

  if (body.id) {
    const id = String(body.id).trim();
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_domains?id=eq.${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { ...supaHeaders(env), prefer: "return=representation" },
        body: JSON.stringify({
          tool_id,
          kind,
          pattern,
          note,
          position,
          updated_at: new Date().toISOString(),
        }),
      }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Update failed", detail }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    const rows = await r.json();
    return new Response(JSON.stringify({ ok: true, domain: rows[0] || null }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/tool_domains`, {
    method: "POST",
    headers: { ...supaHeaders(env), prefer: "return=representation" },
    body: JSON.stringify({ tool_id, kind, pattern, note, position }),
  });
  if (!r.ok) {
    const detail = await r.text();
    return new Response(JSON.stringify({ error: "Create failed", detail }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const rows = await r.json();
  return new Response(JSON.stringify({ ok: true, domain: rows[0] || null }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/* -------------------------------------------------------------------------- */
/*               Tool coaching rules — custom regex / AI hints                */
/* -------------------------------------------------------------------------- */

function sanitizeRule(input: any): any {
  const tool_id = String(input.tool_id || "").trim().slice(0, 64);
  const kindRaw = String(input.kind || "").trim().toLowerCase();
  const kind = kindRaw === "regex" ? "regex" : kindRaw === "ai" ? "ai" : "";
  const title = String(input.title || "").trim().slice(0, 200);
  const sevRaw = String(input.severity || "tip").trim().toLowerCase();
  const severity = ["tip", "warn", "block"].indexOf(sevRaw) !== -1 ? sevRaw : "tip";
  const enabled = input.enabled === false ? false : true;
  const summary = String(input.summary || "").trim().slice(0, 1200);
  const position = Number.isFinite(Number(input.position)) ? Number(input.position) : 0;
  const out: any = { tool_id, kind, title, severity, enabled, summary, position };

  if (kind === "regex") {
    out.pattern = String(input.pattern || "").trim().slice(0, 400);
    out.match_label = String(input.match_label || "").trim().slice(0, 200);
  } else if (kind === "ai") {
    out.fires_on = asStringArray(input.fires_on).slice(0, 20).map((s) => s.slice(0, 400));
    out.silent_on = asStringArray(input.silent_on).slice(0, 20).map((s) => s.slice(0, 400));
    out.rewrite_style = asStringArray(input.rewrite_style).slice(0, 20).map((s) => s.slice(0, 400));
    out.custom_instructions = String(input.custom_instructions || "").slice(0, 2000);
    out.min_length = Number.isFinite(Number(input.min_length))
      ? Math.max(1, Math.min(500, Number(input.min_length)))
      : 25;
    out.cooldown_ms = Number.isFinite(Number(input.cooldown_ms))
      ? Math.max(1000, Math.min(600000, Number(input.cooldown_ms)))
      : 30000;
  }
  return out;
}

async function handleLibraryRulesAll(
  request: Request,
  env: Env
): Promise<Response> {
  // No auth — extension fetches at boot.
  if (request.method !== "GET") {
    return new Response(JSON.stringify({ error: "GET only" }), {
      status: 405,
      headers: { "content-type": "application/json" },
    });
  }
  const r = await fetch(
    `${env.SUPABASE_URL}/rest/v1/tool_coaching_rules?enabled=eq.true&select=id,tool_id,kind,title,severity,summary,pattern,match_label,min_length,cooldown_ms&order=tool_id.asc,position.asc`,
    { headers: supaHeaders(env) }
  );
  if (!r.ok) {
    return new Response(JSON.stringify({ error: "Could not list rules" }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const rules = await r.json();
  // Note: AI rubric fields are intentionally omitted here so the
  // extension only sees what it needs to detect/dispatch. The full
  // rubric stays server-side and is composed into the system prompt
  // inside /coach/evaluate.
  return new Response(JSON.stringify({ ok: true, rules }), {
    status: 200,
    headers: {
      "content-type": "application/json",
      "cache-control": "public, max-age=300",
    },
  });
}

async function handleLibraryRules(
  request: Request,
  env: Env
): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  const url = new URL(request.url);

  if (request.method === "GET") {
    const tool = (url.searchParams.get("tool") || "").trim();
    if (!tool) {
      return new Response(JSON.stringify({ error: "?tool=… is required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_coaching_rules?tool_id=eq.${encodeURIComponent(
        tool
      )}&select=*&order=position.asc`,
      { headers: supaHeaders(env) }
    );
    if (!r.ok) {
      return new Response(JSON.stringify({ error: "Could not list rules" }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    const rules = await r.json();
    return new Response(JSON.stringify({ ok: true, tool, rules }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  if (request.method === "DELETE") {
    let body: { id?: string };
    try {
      body = await request.json();
    } catch {
      body = {};
    }
    const id = String(body.id || "").trim();
    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_coaching_rules?id=eq.${encodeURIComponent(id)}`,
      { method: "DELETE", headers: { ...supaHeaders(env), prefer: "return=minimal" } }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Delete failed", detail }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  // POST: create/update
  let body: any;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  const clean = sanitizeRule(body);
  if (!clean.tool_id) {
    return new Response(JSON.stringify({ error: "tool_id required" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  if (!clean.kind) {
    return new Response(JSON.stringify({ error: "kind must be 'regex' or 'ai'" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  if (!clean.title) {
    return new Response(JSON.stringify({ error: "title required" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  if (clean.kind === "regex") {
    if (!clean.pattern) {
      return new Response(JSON.stringify({ error: "Regex rule needs a pattern" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    try {
      new RegExp(clean.pattern, "i");
    } catch (err: any) {
      return new Response(
        JSON.stringify({ error: `Invalid regex: ${err?.message || String(err)}` }),
        { status: 400, headers: { "content-type": "application/json" } }
      );
    }
  } else {
    if (!clean.fires_on.length) {
      return new Response(
        JSON.stringify({ error: "AI rule needs at least one 'fires on' bullet" }),
        { status: 400, headers: { "content-type": "application/json" } }
      );
    }
  }

  if (body.id) {
    const id = String(body.id).trim();
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/tool_coaching_rules?id=eq.${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { ...supaHeaders(env), prefer: "return=representation" },
        body: JSON.stringify({ ...clean, updated_at: new Date().toISOString() }),
      }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Update failed", detail }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    const rows = await r.json();
    return new Response(JSON.stringify({ ok: true, rule: rows[0] || null }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  const id = `${clean.tool_id}-${clean.kind}-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 6)}`;
  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/tool_coaching_rules`, {
    method: "POST",
    headers: { ...supaHeaders(env), prefer: "return=representation" },
    body: JSON.stringify({ id, ...clean }),
  });
  if (!r.ok) {
    const detail = await r.text();
    return new Response(JSON.stringify({ error: "Create failed", detail }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const rows = await r.json();
  return new Response(JSON.stringify({ ok: true, rule: rows[0] || null }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

async function loadCustomAiRule(
  env: Env,
  ruleId: string
): Promise<any | null> {
  const r = await fetch(
    `${env.SUPABASE_URL}/rest/v1/tool_coaching_rules?id=eq.${encodeURIComponent(
      ruleId
    )}&kind=eq.ai&select=*`,
    { headers: supaHeaders(env) }
  );
  if (!r.ok) return null;
  const rows = (await r.json()) as any[];
  return rows && rows[0] ? rows[0] : null;
}

/* -------------------------------------------------------------------------- */
/*               User skills — personal prompt library (v1)                   */
/* -------------------------------------------------------------------------- */

// /coach/skills routes:
//   GET    /coach/skills            list the caller's skills
//   POST   /coach/skills            create a new skill (body: { title, outcome?, prompt_body, tool_id?, source? })
//   PATCH  /coach/skills            update an existing skill (body: { id, ...fields })
//   DELETE /coach/skills            delete a skill (body: { id })
//
// All require an authenticated Supabase bearer token. The worker uses the
// service role to talk to Postgres but always filters by user_id to enforce
// per-user scoping (RLS is also on as defense-in-depth).
async function handleUserSkills(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  const userId = auth.userId;

  if (request.method === "GET") {
    const url = new URL(request.url);
    const tool = (url.searchParams.get("tool") || "").trim();
    let path = `${env.SUPABASE_URL}/rest/v1/user_skills?user_id=eq.${encodeURIComponent(
      userId
    )}&select=*&order=updated_at.desc`;
    if (tool) {
      // Either skills scoped to this tool OR tool-agnostic skills.
      path += `&or=(tool_id.eq.${encodeURIComponent(tool)},tool_id.eq.)`;
    }
    const r = await fetch(path, { headers: supaHeaders(env) });
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "List failed", detail }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    const skills = await r.json();
    return new Response(JSON.stringify({ ok: true, skills }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  if (request.method === "DELETE") {
    let body: { id?: string };
    try { body = await request.json(); } catch { body = {}; }
    const id = String(body.id || "").trim();
    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/user_skills?id=eq.${encodeURIComponent(id)}&user_id=eq.${encodeURIComponent(userId)}`,
      { method: "DELETE", headers: { ...supaHeaders(env), prefer: "return=minimal" } }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Delete failed", detail }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  // POST = create, PATCH = update
  let body: any;
  try { body = await request.json(); } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  const title = String(body.title || "").trim().slice(0, 200);
  const outcome = String(body.outcome || "").trim().slice(0, 500);
  const promptBody = String(body.prompt_body || "").trim().slice(0, 8000);
  const toolId = String(body.tool_id || "").trim().slice(0, 64);
  const source = (body.source && typeof body.source === "object") ? body.source : { kind: "manual" };

  if (!title || !promptBody) {
    return new Response(
      JSON.stringify({ error: "title and prompt_body are required" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  if (request.method === "PATCH") {
    const id = String(body.id || "").trim();
    if (!id) {
      return new Response(JSON.stringify({ error: "id required" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }
    const r = await fetch(
      `${env.SUPABASE_URL}/rest/v1/user_skills?id=eq.${encodeURIComponent(id)}&user_id=eq.${encodeURIComponent(userId)}`,
      {
        method: "PATCH",
        headers: { ...supaHeaders(env), prefer: "return=representation" },
        body: JSON.stringify({
          title,
          outcome,
          prompt_body: promptBody,
          tool_id: toolId,
          source,
          updated_at: new Date().toISOString(),
        }),
      }
    );
    if (!r.ok) {
      const detail = await r.text();
      return new Response(JSON.stringify({ error: "Update failed", detail }), {
        status: 502,
        headers: { "content-type": "application/json" },
      });
    }
    const rows = await r.json();
    return new Response(JSON.stringify({ ok: true, skill: rows[0] || null }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  // POST → create
  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/user_skills`, {
    method: "POST",
    headers: { ...supaHeaders(env), prefer: "return=representation" },
    body: JSON.stringify({
      user_id: userId,
      title,
      outcome,
      prompt_body: promptBody,
      tool_id: toolId,
      source,
    }),
  });
  if (!r.ok) {
    const detail = await r.text();
    return new Response(JSON.stringify({ error: "Create failed", detail }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const rows = await r.json();
  return new Response(JSON.stringify({ ok: true, skill: rows[0] || null }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

// POST /coach/skills/use { id, count? } — increments use_count and sets
// last_used_at = now() on a single user-owned skill. Called by the Glide
// app when it drains the puck's queued use events.
//
// Optional `count` allows batching multiple uses of the same skill that
// happened before the app could sync (e.g., user pasted the same prompt
// three times in a row while offline). Defaults to 1.
async function handleUserSkillUse(request: Request, env: Env): Promise<Response> {
  const auth = await requireSupabaseUser(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  const userId = auth.userId;

  let body: { id?: string; count?: number };
  try { body = await request.json(); } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  const id = String(body.id || "").trim();
  const count = Math.max(1, Math.min(100, Number(body.count) || 1));
  if (!id) {
    return new Response(JSON.stringify({ error: "id required" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }

  // First read the current use_count so we can increment server-side.
  const getRes = await fetch(
    `${env.SUPABASE_URL}/rest/v1/user_skills?id=eq.${encodeURIComponent(id)}&user_id=eq.${encodeURIComponent(userId)}&select=id,use_count`,
    { headers: supaHeaders(env) }
  );
  if (!getRes.ok) {
    const detail = await getRes.text();
    return new Response(JSON.stringify({ error: "Use lookup failed", detail }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const existing = (await getRes.json()) as any[];
  if (!existing || !existing[0]) {
    return new Response(JSON.stringify({ error: "Skill not found" }), {
      status: 404,
      headers: { "content-type": "application/json" },
    });
  }
  const newCount = (Number(existing[0].use_count) || 0) + count;
  const r = await fetch(
    `${env.SUPABASE_URL}/rest/v1/user_skills?id=eq.${encodeURIComponent(id)}&user_id=eq.${encodeURIComponent(userId)}`,
    {
      method: "PATCH",
      headers: { ...supaHeaders(env), prefer: "return=representation" },
      body: JSON.stringify({
        use_count: newCount,
        last_used_at: new Date().toISOString(),
      }),
    }
  );
  if (!r.ok) {
    const detail = await r.text();
    return new Response(JSON.stringify({ error: "Use update failed", detail }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const rows = await r.json();
  return new Response(JSON.stringify({ ok: true, skill: rows[0] || null }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/* -------------------------------------------------------------------------- */
/*               Coach Evaluate — judgment-based remote rules                 */
/* -------------------------------------------------------------------------- */

interface CoachEvaluateBody {
  tool?: string;
  toolLabel?: string;
  kind?: string;          // 'prompt-quality' (built-in) or 'custom-ai' (per-tool rule by id)
  ruleId?: string;        // required when kind === 'custom-ai'
  promptText?: string;
  // 'in-the-moment' (default): the user is still typing — coach if there
  //   is something fixable BEFORE they hit send.
  // 'after-send': the user already submitted the prompt. Coach is
  //   retrospective ("for next time, try…"); same response shape, the
  //   intro/UX framing differs.
  mode?: "in-the-moment" | "after-send";
  knowledge?: { title?: string; body?: string }[];
  recentHistory?: string[];
  // Attachment context staged alongside the prompt — file chips,
  // image previews, selected-element pills from the tool's preview
  // pane. The model uses this to avoid flagging "your prompt is too
  // vague" when the user has clearly attached visual context.
  attachments?: {
    count?: number;
    types?: string[];   // 'image' | 'file' | 'target' (selected element) | …
    labels?: string[];  // alt / filename / pill text
  };
}

async function handleCoachEvaluate(request: Request, env: Env): Promise<Response> {
  // Prompt coaching runs in third-party target tabs (bolt.new, v0, etc.)
  // where the extension can't read the user's Supabase session — it lives
  // in the Glide app on a different origin. Accept the anon trial token
  // (sent by glideFetchPatch.js) so coaching works for everyone with the
  // extension installed, the same way /chat and /tts do. Rate-limited per
  // token in requireAuthOrAnon.
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }
  if (!env.ANTHROPIC_API_KEY) {
    return new Response(
      JSON.stringify({ error: "ANTHROPIC_API_KEY is not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }
  let body: CoachEvaluateBody;
  try {
    body = await request.json();
  } catch {
    return new Response(
      JSON.stringify({ error: "Body must be JSON" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  const tool = String(body.tool || "").slice(0, 64);
  const toolLabel = String(body.toolLabel || tool || "an AI tool").slice(0, 64);
  const kind = String(body.kind || "").slice(0, 32);
  const ruleId = String(body.ruleId || "").trim().slice(0, 200);
  const promptText = String(body.promptText || "").trim().slice(0, 4000);
  const mode = body.mode === "after-send" ? "after-send" : "in-the-moment";
  const knowledge = Array.isArray(body.knowledge) ? body.knowledge.slice(0, 6) : [];

  // Normalize attachment context. We cap counts + label sizes so a
  // malicious caller can't blow up the prompt.
  const attachments = body.attachments && typeof body.attachments === "object"
    ? {
        count: Math.max(0, Math.min(50, Number(body.attachments.count) || 0)),
        types: Array.isArray(body.attachments.types)
          ? body.attachments.types.filter((t): t is string => typeof t === "string").slice(0, 8)
          : [],
        labels: Array.isArray(body.attachments.labels)
          ? body.attachments.labels
              .filter((l): l is string => typeof l === "string")
              .map((l) => l.slice(0, 80))
              .slice(0, 8)
          : [],
      }
    : { count: 0, types: [], labels: [] };

  if (!kind || !promptText) {
    return new Response(
      JSON.stringify({ error: "kind and promptText are required" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }
  if (kind !== "prompt-quality" && kind !== "custom-ai") {
    return new Response(
      JSON.stringify({ error: `Unsupported kind: ${kind}` }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  // Resolve the rubric for this call. Two paths:
  //   1. built-in 'prompt-quality' → shipped defaults + per-tool override row.
  //   2. 'custom-ai' → a row in tool_coaching_rules with kind='ai'.
  let rubric: RubricRow;
  let minLength = 25;
  let customTitle = "";
  if (kind === "custom-ai") {
    if (!ruleId) {
      return new Response(
        JSON.stringify({ error: "ruleId is required for custom-ai" }),
        { status: 400, headers: { "content-type": "application/json" } }
      );
    }
    const row = await loadCustomAiRule(env, ruleId);
    if (!row) {
      return new Response(
        JSON.stringify({ error: `Unknown rule: ${ruleId}` }),
        { status: 404, headers: { "content-type": "application/json" } }
      );
    }
    rubric = {
      rule_id: ruleId,
      summary: row.summary || "",
      fires_on: asStringArray(row.fires_on),
      silent_on: asStringArray(row.silent_on),
      rewrite_style: asStringArray(row.rewrite_style),
      cadence: "",
      custom_instructions: row.custom_instructions || "",
    };
    minLength = Number.isFinite(Number(row.min_length))
      ? Number(row.min_length)
      : 25;
    customTitle = String(row.title || "").slice(0, 200);
  } else {
    const overrideRow = tool ? await loadRubricRow(env, tool, "prompt-quality") : null;
    rubric = mergeRubricRow("prompt-quality", overrideRow);
  }

  if (promptText.length < minLength) {
    return new Response(
      JSON.stringify({ ok: true, fire: false }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }

  const knowledgeText = knowledge
    .filter((k): k is { title: string; body: string } =>
      !!k && typeof k.title === "string" && typeof k.body === "string"
    )
    .map((k) => `### ${k.title.slice(0, 120)}\n${k.body.slice(0, 1000)}`)
    .join("\n\n");

  const bullets = (lines: string[]) =>
    lines.map((l) => `- ${l}`).join("\n");
  // Two registers:
  // - in-the-moment (default): the user is still drafting; coach to
  //   improve BEFORE they send. Use present tense.
  // - after-send: they already submitted. Coach in retrospective voice
  //   ("for next time, try…") so the tip feels like a coach reviewing
  //   what happened, not a tap on the shoulder mid-thought.
  const scopeNote = kind === "custom-ai" && customTitle
    ? ` Your scope is narrow: ${customTitle}.`
    : "";
  const intro = mode === "after-send"
    ? `You are a prompt-quality coach. The user just sent the following prompt to ${toolLabel}.${scopeNote} Your job is a brief retrospective tip for next time — only if there is a concrete, useful improvement to surface. Be honest and specific; phrase suggestions as "for next time" guidance, not as instructions to re-send the prompt now.`
    : `You are a prompt-quality coach for a non-technical user about to send a prompt to ${toolLabel}.${scopeNote}`;

  // Attachment-aware preamble. The model needs to know that "this
  // prompt seems short / vague" might be the wrong call when an image
  // or selected element is already staged. Keep it concise — most
  // requests have count=0 and we want to keep the system prompt small.
  const attachmentNote = attachments.count > 0
    ? `\n\nIMPORTANT — attachment context:
The user has staged ${attachments.count} non-text attachment${attachments.count === 1 ? "" : "s"} alongside the prompt${attachments.types.length ? ` (${attachments.types.join(", ")})` : ""}${attachments.labels.length ? `: ${attachments.labels.join("; ")}` : ""}.
DO NOT flag the prompt as "too vague," "missing context," or "needs more detail" purely because the text is short or refers to "this" or "that" — the attachment IS the context. Fire only if the text itself contains a concrete issue the attachment cannot resolve (e.g. ambiguous goal, contradictory instruction).`
    : "";

  const systemPrompt = `${intro}

Your only job: identify the SINGLE most actionable improvement to the prompt, OR return fire:false if the prompt is already good. Be honest and specific. The user will see your suggestion and can accept or dismiss it.

Bias toward fire:false. Only fire when the issue is concrete and the rewrite would meaningfully change the AI's output. Do NOT nag about minor stylistic things.

What counts as fire-worthy:
${bullets(rubric.fires_on)}

What does NOT count as fire-worthy:
${bullets(rubric.silent_on)}${attachmentNote}

Output JSON only, matching this schema:
{
  "fire": boolean,
  "severity": "tip",        // always "tip" for prompt-quality
  "title": string,           // ≤80 chars, plain language
  "body": string,            // 1-2 sentences explaining why
  "suggestedPrompt": string  // a complete paste-able rewrite
}

If fire is false, return { "fire": false } only.

When you suggest a rewrite, follow these rewrite-style guidelines:
${bullets(rubric.rewrite_style)}${rubric.custom_instructions
  ? `\n\nAdditional guidance for ${toolLabel}:\n${rubric.custom_instructions}`
  : ""
}`;

  const promptLabel = mode === "after-send" ? "User's submitted prompt" : "User's draft prompt";
  const userMessage = knowledgeText
    ? `Reference knowledge for ${toolLabel}:\n\n${knowledgeText}\n\n---\n\n${promptLabel}:\n"""\n${promptText}\n"""`
    : `${promptLabel} for ${toolLabel}:\n"""\n${promptText}\n"""`;

  const claudeRes = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5",
      max_tokens: 600,
      temperature: 0.3,
      system: systemPrompt,
      messages: [{ role: "user", content: userMessage }],
    }),
  });

  if (!claudeRes.ok) {
    const errBody = await claudeRes.text();
    console.error(`[/coach/evaluate] Anthropic ${claudeRes.status}: ${errBody}`);
    return new Response(
      JSON.stringify({ error: "Coach call failed", detail: errBody.slice(0, 300) }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  const claudeData: any = await claudeRes.json();
  const parts = Array.isArray(claudeData?.content) ? claudeData.content : [];
  const text = parts
    .filter((p: any) => p && p.type === "text" && typeof p.text === "string")
    .map((p: any) => p.text)
    .join("\n")
    .trim();

  // Strip code fences if Claude wrapped it
  const stripped = text.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const parsed = safeJsonParse(stripped) as any;

  if (!parsed || typeof parsed !== "object") {
    // If we can't parse, fail closed — don't fire a hint.
    return new Response(
      JSON.stringify({ ok: true, fire: false, debug: { unparseable: text.slice(0, 200) } }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }

  const result: any = { ok: true, fire: !!parsed.fire };
  if (parsed.fire) {
    result.severity = "tip";
    result.title = String(parsed.title || "Try a sharper prompt").slice(0, 200);
    result.body = String(parsed.body || "").slice(0, 800);
    if (parsed.suggestedPrompt && typeof parsed.suggestedPrompt === "string") {
      result.suggestedPrompt = parsed.suggestedPrompt.slice(0, 2000);
    }
  }

  return new Response(JSON.stringify(result), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function htmlToText(html: string): string {
  // Remove script, style, nav, footer, header, aside — whatever is unlikely
  // to be the article body. Then strip remaining tags and decode entities.
  let stripped = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<svg[\s\S]*?<\/svg>/gi, " ")
    .replace(/<header[\s\S]*?<\/header>/gi, " ")
    .replace(/<footer[\s\S]*?<\/footer>/gi, " ")
    .replace(/<nav[\s\S]*?<\/nav>/gi, " ")
    .replace(/<aside[\s\S]*?<\/aside>/gi, " ");

  // If the HTML contains an <article> tag, prefer its inner content.
  const articleMatch = stripped.match(/<article[^>]*>([\s\S]*?)<\/article>/i);
  if (articleMatch) {
    stripped = articleMatch[1];
  } else {
    const mainMatch = stripped.match(/<main[^>]*>([\s\S]*?)<\/main>/i);
    if (mainMatch) stripped = mainMatch[1];
  }

  // Preserve paragraph breaks before stripping tags.
  stripped = stripped
    .replace(/<\/(p|div|li|h[1-6]|tr|br)\s*>/gi, "\n")
    .replace(/<br\s*\/?\s*>/gi, "\n");

  return decodeEntities(stripTags(stripped))
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}


async function handleTTS(request: Request, env: Env): Promise<Response> {
  // TTS is part of the anonymous trial — the demo tutorial must be
  // able to speak aloud during the welcome flow.
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }

  const body = await request.text();

  // Let the caller choose the voice via body.voice_id; fall back to the
  // worker's configured default. ElevenLabs picks the voice from the URL
  // path, not the body, so we have to extract and substitute here.
  let voiceId = env.ELEVENLABS_VOICE_ID;
  const parsed = safeJsonParse(body) as { voice_id?: string } | null;
  if (parsed && typeof parsed.voice_id === "string" && /^[A-Za-z0-9]{15,40}$/.test(parsed.voice_id)) {
    voiceId = parsed.voice_id;
  }

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": env.ELEVENLABS_API_KEY,
        "content-type": "application/json",
        accept: "audio/mpeg",
      },
      body,
    }
  );

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`[/tts] ElevenLabs API error ${response.status}: ${errorBody}`);
    return new Response(errorBody, {
      status: response.status,
      headers: { "content-type": "application/json" },
    });
  }

  return new Response(response.body, {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") || "audio/mpeg",
    },
  });
}

// Speech-to-text for the side panel's voice "Ask". The Web Speech API
// doesn't work in extension pages, so the panel records the mic itself
// (getUserMedia + MediaRecorder) and POSTs the clip here as multipart
// form-data ({ file, model_id }); we forward it to ElevenLabs Scribe and
// return its JSON ({ text, ... }). Part of the anonymous trial, same as TTS.
async function handleSTT(request: Request, env: Env): Promise<Response> {
  const auth = await requireAuthOrAnon(request, env);
  if (!auth.ok) {
    return new Response(JSON.stringify({ error: auth.error }), {
      status: auth.status,
      headers: { "content-type": "application/json" },
    });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return new Response(JSON.stringify({ error: "Expected multipart/form-data" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }

  const file = form.get("file");
  if (!(file instanceof Blob)) {
    return new Response(JSON.stringify({ error: "Missing audio 'file' field" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }

  const rawModel = form.get("model_id");
  const modelId = typeof rawModel === "string" && rawModel ? rawModel : "scribe_v1";

  const upstream = new FormData();
  upstream.append("file", file, "audio.webm");
  upstream.append("model_id", modelId);

  const response = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
    method: "POST",
    headers: {
      // Don't set content-type — fetch derives the multipart boundary.
      "xi-api-key": env.ELEVENLABS_API_KEY,
    },
    body: upstream,
  });

  const text = await response.text();
  if (!response.ok) {
    console.error(`[/stt] ElevenLabs STT error ${response.status}: ${text}`);
  }
  return new Response(text, {
    status: response.status,
    headers: { "content-type": "application/json" },
  });
}
