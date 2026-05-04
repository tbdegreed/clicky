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
  INVISIBLE_SIGNING_SECRET: string;
  SUPABASE_URL: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
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

      if (url.pathname === "/invisible-webhook") {
        return withCORS(await handleInvisibleWebhook(request, env));
      }

      if (url.pathname === "/invisible-callback") {
        return withCORS(await handleInvisibleCallback(request, env));
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
- OPEN with one or two short "info" steps that orient the user: what they're about to accomplish, what they'll need (account, file, app already open), and any one-time setup. The user should never wonder "wait, where am I supposed to be?"
- Each action step must be SELF-EXPLANATORY: include a brief reason or context inline when the action would otherwise be opaque. Bad: "Click the gear icon." Good: "Click the gear icon in the top right to open Settings — that's where the integration toggle lives."
- Whenever a step uses a TERM, NAME, or CONCEPT introduced earlier in the video (like "your tracking dashboard", "the linked workspace", "the source file"), define it briefly in the step itself. Don't assume the user remembers what the presenter called it.
- For "type" steps, if the presenter typed specific example data, include both a generic adaptable placeholder AND a one-line explanation of what kind of value belongs there. Bad: 'Type "Coffee with Jane".' Good: 'Type a short title for your event (e.g. "Coffee with Jane") — this is what shows up on the calendar block.'
- For "verify" steps, describe the expected screen state in CONCRETE visible terms, not "what the presenter showed". Bad: "Confirm the result looks like in the video." Good: "Confirm a green 'Connected' badge appears next to the integration name."
- Fill in obvious prerequisites the video skips. If the video opens on a logged-in dashboard, add steps for signing in / opening the right app first. If it assumes an account exists, add a note about creating one.
- Smooth out gaps. If the presenter narrates two actions in one sentence, split them into two steps. If they gloss over a clearly-needed step (closing a modal, accepting cookies, scrolling, waiting for a load), include it.
- Drop presenter-specific framing. "As you can see on my screen" → describe what the user should see on theirs. "I'll click here" → "Click X." Never reference "the video", "the presenter", "earlier", or "as we discussed".
- Use "info" steps not just at the start but at any transition between phases, especially when the goal of the next phase isn't obvious from the next action alone (e.g. "Now we'll grant the app access to your calendar so it can create events on your behalf.").
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

/* -------------------------------------------------------------------------- */
/*                           Invisible integration                            */
/* -------------------------------------------------------------------------- */

const MAX_SUMMARY_LEN = 200;
const TIMESTAMP_TOLERANCE_S = 5 * 60; // 5 minutes

async function hmacHexSha256(secret: string, message: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function timingSafeEqualStr(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function handleInvisibleWebhook(request: Request, env: Env): Promise<Response> {
  if (!env.INVISIBLE_SIGNING_SECRET) {
    return new Response(
      JSON.stringify({ error: "INVISIBLE_SIGNING_SECRET is not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }
  if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_ROLE_KEY) {
    return new Response(
      JSON.stringify({ error: "Supabase is not configured for the worker" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  const timestamp = request.headers.get("x-invisible-timestamp") || "";
  const signature = request.headers.get("x-invisible-signature") || "";
  const rawBody = await request.text();

  // Replay protection: reject stale timestamps.
  const ts = Number(timestamp);
  if (!Number.isFinite(ts)) {
    return new Response(JSON.stringify({ error: "Bad timestamp" }), {
      status: 401, headers: { "content-type": "application/json" },
    });
  }
  const nowS = Math.floor(Date.now() / 1000);
  if (Math.abs(nowS - ts) > TIMESTAMP_TOLERANCE_S) {
    return new Response(JSON.stringify({ error: "Timestamp out of tolerance" }), {
      status: 401, headers: { "content-type": "application/json" },
    });
  }

  const expected = await hmacHexSha256(
    env.INVISIBLE_SIGNING_SECRET,
    `${timestamp}:${rawBody}`
  );
  const received = signature.replace(/^v1=/, "");
  if (!timingSafeEqualStr(expected, received)) {
    return new Response(JSON.stringify({ error: "Bad signature" }), {
      status: 401, headers: { "content-type": "application/json" },
    });
  }

  let payload: any;
  try {
    payload = JSON.parse(rawBody);
  } catch {
    return new Response(JSON.stringify({ error: "Body is not JSON" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }

  const issuanceId = String(payload?.issuance_id || "").slice(0, 64);
  const userToken = String(payload?.user_token || "").slice(0, 256);
  const teamId = String(payload?.team_id || "").slice(0, 64);
  const callbackUrl = String(payload?.callback_url || "").slice(0, 1024);
  const summary = String(payload?.summary || "").slice(0, MAX_SUMMARY_LEN);

  if (!issuanceId || !userToken || !callbackUrl) {
    return new Response(JSON.stringify({
      error: "Missing required fields: issuance_id, user_token, callback_url",
    }), { status: 400, headers: { "content-type": "application/json" } });
  }
  if (!/^https?:\/\//i.test(callbackUrl)) {
    return new Response(JSON.stringify({ error: "callback_url must be http(s)" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }

  const upsertBody = [{
    issuance_id: issuanceId,
    user_token: userToken,
    team_id: teamId || null,
    callback_url: callbackUrl,
    summary,
    status: "pending",
  }];

  const dbRes = await fetch(
    `${env.SUPABASE_URL}/rest/v1/invisible_sessions?on_conflict=issuance_id`,
    {
      method: "POST",
      headers: {
        apikey: env.SUPABASE_SERVICE_ROLE_KEY,
        authorization: `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
        "content-type": "application/json",
        prefer: "resolution=merge-duplicates,return=minimal",
      },
      body: JSON.stringify(upsertBody),
    }
  );

  if (!dbRes.ok) {
    const text = await dbRes.text();
    console.error("[/invisible-webhook] Supabase upsert failed", dbRes.status, text);
    return new Response(JSON.stringify({
      error: "Could not store session",
      detail: text,
    }), { status: 502, headers: { "content-type": "application/json" } });
  }

  return new Response(JSON.stringify({ ok: true }), {
    status: 200, headers: { "content-type": "application/json" },
  });
}

interface CallbackBody {
  issuance_id?: string;
  user_token?: string;
  status?: "completed" | "abandoned" | "in_progress";
  observations?: { text?: string; confidence?: number }[];
}

async function handleInvisibleCallback(request: Request, env: Env): Promise<Response> {
  if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_ROLE_KEY) {
    return new Response(JSON.stringify({ error: "Supabase is not configured" }), {
      status: 500, headers: { "content-type": "application/json" },
    });
  }

  let body: CallbackBody;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Body must be JSON" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }

  const issuanceId = String(body.issuance_id || "").slice(0, 64);
  const userToken = String(body.user_token || "").slice(0, 256);
  const status = body.status;
  const observations = Array.isArray(body.observations) ? body.observations.slice(0, 20) : [];

  if (!issuanceId || !userToken) {
    return new Response(JSON.stringify({ error: "issuance_id and user_token required" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }
  if (status !== "completed" && status !== "abandoned" && status !== "in_progress") {
    return new Response(JSON.stringify({ error: "Invalid status" }), {
      status: 400, headers: { "content-type": "application/json" },
    });
  }

  // Look up the session to authenticate and to get the callback_url.
  const lookup = await fetch(
    `${env.SUPABASE_URL}/rest/v1/invisible_sessions?issuance_id=eq.${encodeURIComponent(issuanceId)}&select=*`,
    {
      headers: {
        apikey: env.SUPABASE_SERVICE_ROLE_KEY,
        authorization: `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
      },
    }
  );
  if (!lookup.ok) {
    return new Response(JSON.stringify({ error: "Could not look up session" }), {
      status: 502, headers: { "content-type": "application/json" },
    });
  }
  const rows: any[] = await lookup.json();
  const row = rows[0];
  if (!row) {
    return new Response(JSON.stringify({ error: "No such session" }), {
      status: 404, headers: { "content-type": "application/json" },
    });
  }
  if (!timingSafeEqualStr(String(row.user_token || ""), userToken)) {
    return new Response(JSON.stringify({ error: "Bad user_token" }), {
      status: 401, headers: { "content-type": "application/json" },
    });
  }
  const callbackUrl = String(row.callback_url || "");
  if (!/^https?:\/\//i.test(callbackUrl)) {
    return new Response(JSON.stringify({ error: "Stored callback_url is invalid" }), {
      status: 500, headers: { "content-type": "application/json" },
    });
  }

  // Sanitize observations to match Invisible's schema exactly.
  const cleanObservations = observations
    .filter((o) => o && typeof o.text === "string" && o.text.trim())
    .map((o) => {
      const item: { text: string; confidence?: number } = {
        text: String(o.text).slice(0, 10_000),
      };
      if (typeof o.confidence === "number" && Number.isFinite(o.confidence)) {
        item.confidence = Math.max(0, Math.min(1, o.confidence));
      }
      return item;
    });

  const callbackBody = {
    issuance_id: issuanceId,
    user_token: userToken,
    status,
    observations: cleanObservations,
  };

  const cb = await fetch(callbackUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(callbackBody),
  });
  const cbText = await cb.text();

  // Update local row with the latest status (best-effort).
  await fetch(
    `${env.SUPABASE_URL}/rest/v1/invisible_sessions?issuance_id=eq.${encodeURIComponent(issuanceId)}`,
    {
      method: "PATCH",
      headers: {
        apikey: env.SUPABASE_SERVICE_ROLE_KEY,
        authorization: `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
        "content-type": "application/json",
        prefer: "return=minimal",
      },
      body: JSON.stringify({ status, completed_at: new Date().toISOString() }),
    }
  );

  if (!cb.ok) {
    console.error("[/invisible-callback] forward failed", cb.status, cbText);
    return new Response(JSON.stringify({
      error: "Invisible rejected the callback",
      status: cb.status,
      detail: cbText,
    }), { status: 502, headers: { "content-type": "application/json" } });
  }

  return new Response(JSON.stringify({ ok: true }), {
    status: 200, headers: { "content-type": "application/json" },
  });
}
