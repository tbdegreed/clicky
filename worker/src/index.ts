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
 *   POST /transcribe-token  → AssemblyAI temp token
 *   POST /youtube           → Generate a tutorial step plan from a YouTube video (Gemini)
 */

interface Env {
  ANTHROPIC_API_KEY: string;
  ELEVENLABS_API_KEY: string;
  ELEVENLABS_VOICE_ID: string;
  ASSEMBLYAI_API_KEY: string;
  GEMINI_API_KEY: string;
}

const CORS_HEADERS: Record<string, string> = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "Content-Type",
};

/** Add CORS headers to any Response. */
function withCORS(response: Response): Response {
  const newHeaders = new Headers(response.headers);
  for (const [key, value] of Object.entries(CORS_HEADERS)) {
    newHeaders.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: newHeaders,
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (request.method !== "POST") {
      return withCORS(new Response("Method not allowed", { status: 405 }));
    }

    try {
      if (url.pathname === "/chat") {
        return withCORS(await handleChat(request, env));
      }

      if (url.pathname === "/tts") {
        return withCORS(await handleTTS(request, env));
      }

      if (url.pathname === "/transcribe-token") {
        return withCORS(await handleTranscribeToken(env));
      }

      if (url.pathname === "/youtube") {
        return withCORS(await handleYoutube(request, env));
      }

      if (url.pathname === "/page") {
        return withCORS(await handlePage(request, env));
      }
    } catch (error) {
      console.error(`[${url.pathname}] Unhandled error:`, error);
      return withCORS(new Response(
        JSON.stringify({ error: String(error) }),
        { status: 500, headers: { "content-type": "application/json" } }
      ));
    }

    return withCORS(new Response("Not found", { status: 404 }));
  },
};

async function handleChat(request: Request, env: Env): Promise<Response> {
  const body = await request.text();

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body,
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

async function handleTranscribeToken(env: Env): Promise<Response> {
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

const STEP_PLAN_SCHEMA = {
  type: "object",
  properties: {
    refusal: {
      type: "string",
      description:
        "If this is not an instructional video, set this to a brief reason and omit the other fields.",
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
      items: {
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
      },
    },
  },
};

const STEP_PLAN_PROMPT = `You are turning the attached video into a guided, do-it-yourself tutorial.

The output is NOT a transcript or a literal step-by-step copy of what the presenter does. It is a self-contained tutorial that an end user can follow on their own machine, even though they will not have the presenter's voice in their ear. Your job is to translate, not transcribe.

Output JSON matching the response schema.

Translate the video into a cohesive experience:
- Fill in obvious prerequisites the video skips. If the video opens with a logged-in dashboard, add steps for signing in / opening the right app first. If it assumes an account exists, add a note about creating one.
- Smooth out gaps. If the presenter narrates two actions in one sentence, split them into two steps. If the presenter glosses over a clearly-needed step (closing a modal, accepting cookies, scrolling), include it.
- Drop presenter-specific framing. "As you can see on my screen" becomes a clean instruction about what the user should see on theirs. "I'll click here" becomes "Click X."
- Replace presenter-specific data with realistic placeholders the user can adapt. If they type "Coffee with Jane on Tuesday," use a generic equivalent like "your meeting title."
- Use "info" steps sparingly to set context at the start, between phases, and at the end — e.g. a one-line orientation before the user dives in, or a transition like "Now we'll switch to the analytics tab." Do not narrate the entire video as info steps.
- Use "verify" steps after meaningful state changes so the user can confirm progress before moving on.
- Where the video is unclear about what comes next, infer a reasonable next action based on the platform's conventions. Better to give a confident, plausible step than to leave the user stranded.
- DO NOT include the presenter's intro/outro, sponsor reads, calls to subscribe, or commentary about themselves or their channel.
- DO NOT invent steps that contradict the video. Filling gaps means adding the small connectives the video assumed; it does not mean inventing an entirely different workflow.

Field rules:
- title: short, action-oriented, user-perspective (max 60 chars). Example: "Post your first article on LinkedIn".
- summary: one or two sentences describing what the user will accomplish and walk away with.
- steps: atomic user actions or short info beats. Number sequentially starting at 1. Aim for 6–25 steps; err on the side of more, smaller steps if the workflow is intricate.
- stepType must be one of: navigate, click, type, verify, wait, info.
- description: a single instruction in the second person ("Click the Save button in the top right"). No presenter references.
- visualHint: a short cue for what the user should see once the step is complete. Required for click/navigate/type/verify steps.
- demoInput: only for type steps where a sample value would help the user move forward.
- expectedDuration: rough seconds for an average user to complete this step.
- browserCompatible: true ONLY if the entire workflow happens inside a web browser.
- shareRecommendation.scope: "browser" if entirely in a browser tab; "window" if a single desktop app; "screen" if it spans multiple windows.

If this video is NOT a tutorial — music video, vlog, ad, abstract demo with no actionable user workflow — set the "refusal" field to a one-sentence reason and omit the other fields. Do that check before generating any steps.`;

interface YoutubeRequestBody {
  videoUrl?: string;
}

async function handleYoutube(request: Request, env: Env): Promise<Response> {
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

  const videoUrl = (body.videoUrl || "").trim();
  if (!YOUTUBE_URL_REGEX.test(videoUrl)) {
    return new Response(
      JSON.stringify({
        error:
          "videoUrl must be a public YouTube URL (youtube.com/watch?v=…, youtu.be/…, or shorts/embed).",
      }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

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

const PAGE_PROMPT = `You are turning the attached web page into a guided, do-it-yourself tutorial.

The output is NOT a transcript or a literal copy of the article. It is a self-contained tutorial that an end user can follow on their own machine. Translate, don't transcribe.

Output JSON matching the response schema.

Translate the article into a cohesive experience:
- Fill in obvious prerequisites the article skips. If it assumes the reader already has an account, opened the right app, or installed something, add explicit steps for those.
- Smooth out gaps. If a paragraph mentions two actions, split them into two steps. If the article glosses over a clearly-needed step (closing a modal, accepting cookies, scrolling), include it.
- Drop author commentary, intro/outro, calls to subscribe, and anything about the writer's own experience that the user doesn't need to copy.
- Replace any example data ("Coffee with Jane on Tuesday") with adaptable placeholders like "your meeting title."
- Use "info" steps sparingly to set context at the start and at transitions between phases. Do not narrate the entire article as info steps.
- Use "verify" steps after meaningful state changes so the user can confirm progress.
- Where the article is unclear about what comes next, infer a reasonable next action based on the platform's conventions. Better to give a confident, plausible step than to leave the user stranded.
- DO NOT invent steps that contradict the article. Filling gaps means adding small connectives, not inventing a different workflow.

Field rules:
- title: short, action-oriented, user-perspective (max 60 chars).
- summary: one or two sentences describing what the user will accomplish.
- steps: atomic user actions or short info beats. Number sequentially starting at 1. Aim for 6–25 steps.
- stepType must be one of: navigate, click, type, verify, wait, info.
- description: a single instruction in the second person.
- visualHint: a short cue for what the user should see once the step is complete (required for click/navigate/type/verify).
- demoInput: only for type steps where a sample value would help.
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
