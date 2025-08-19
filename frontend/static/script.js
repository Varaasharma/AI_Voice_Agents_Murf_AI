// Day 12: WebSocket-based real-time audio streaming
// Connects to /ws/{session_id} and streams audio bytes while recording

let recorder; // kept for compatibility but we now stream via Web Audio API
let websocket;
let isRecording = false;
let isConnected = false;
let pcmBuffer = []; // buffer (not used with new pipeline, kept for compatibility)
let audioCtx = null;
let sourceNode = null;
let processorNode = null;
let inputSampleRate = 48000;
let bytesSentTotal = 0;

const micBtn   = document.getElementById("micBtn");
const statusEl = document.getElementById("status");
const dotEl    = document.getElementById("dot");
const barsEl   = document.getElementById("bars");
const errEl    = document.getElementById("error");
const sidEl    = document.getElementById("sid");
const transcriptEl = document.getElementById("transcript"); // live transcript target
let transcriptAccum = ""; // kept for compatibility (repurposed as committed text)
let lastPartial = "";     // not used in new model, but kept to avoid large diff

// New model: committed transcript + live partial
let transcriptCommitted = "";
let currentPartial = "";

// Convert Float32Array [-1,1] audio to PCM16
function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

// Downsample Float32 audio from inputSampleRate to 16 kHz, return Int16Array
function downsampleTo16kPCM(float32Array, inSampleRate) {
  const outSampleRate = 16000;
  if (inSampleRate === outSampleRate) {
    // Fast path: just convert to PCM16
    const out = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      let s = Math.max(-1, Math.min(1, float32Array[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  const sampleRateRatio = inSampleRate / outSampleRate;
  const newLength = Math.round(float32Array.length / sampleRateRatio);
  const resultFloat = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < newLength) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < float32Array.length; i++) {
      accum += float32Array[i];
      count++;
    }
    resultFloat[offsetResult] = count > 0 ? (accum / count) : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  const out = new Int16Array(resultFloat.length);
  for (let i = 0; i < resultFloat.length; i++) {
    let s = Math.max(-1, Math.min(1, resultFloat[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// ---- Session id in URL ----
function uid() {
  return crypto.randomUUID ? crypto.randomUUID() :
    'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random()*16|0, v = c==='x'?r:(r&0x3|0x8);
      return v.toString(16);
    });
}
function ensureSessionId() {
  const url = new URL(window.location.href);
  let sid = url.searchParams.get("session_id");
  if (!sid) {
    sid = uid();
    url.searchParams.set("session_id", sid);
    window.history.replaceState({}, "", url.toString());
  }
  sidEl.textContent = sid;
  return sid;
}
const SESSION_ID = ensureSessionId();

function joinWithSpace(a, b) {
  if (!a) return b || "";
  if (!b) return a || "";
  const needsSpace = !(/[\s\n]$/.test(a)) && !(/^[\p{P}\s]/u.test(b));
  return a + (needsSpace ? " " : "") + b;
}

function renderTranscript() {
  const text = joinWithSpace(transcriptCommitted, currentPartial);
  if (transcriptEl) {
    transcriptEl.textContent = text;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }
}

// Normalize a string for overlap comparison: lowercase, remove spaces/punctuation
function normalizedChars(s) {
  const out = [];
  for (let i = 0; i < s.length; i++) {
    const ch = s[i].toLowerCase();
    if (/\p{L}|\p{N}/u.test(ch)) out.push(ch);
  }
  return out.join("");
}

// Tokenization helpers for overlap/dedup
function tokenizeWords(s) {
  if (!s) return [];
  const parts = s.trim().split(/\s+/);
  return parts.map(t => ({ text: t, norm: t.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "") }));
}
function joinTokens(tokens) {
  return tokens.map(t => t.text).join(" ");
}

// Deduplicate repeated trailing block in a string (handles stutters in partials)
function dedupeRepeatedTailString(s) {
  const tokens = tokenizeWords(s);
  let n = tokens.length;
  let changed = true;
  while (changed) {
    changed = false;
    n = tokens.length;
    for (let m = Math.floor(n / 2); m >= 1; m--) {
      let equal = true;
      for (let i = 0; i < m; i++) {
        const a = tokens[n - 2*m + i];
        const b = tokens[n - m + i];
        if (!a || !b || a.norm !== b.norm) { equal = false; break; }
      }
      if (equal) {
        tokens.splice(n - m, m);
        changed = true;
        break;
      }
    }
  }
  return joinTokens(tokens);
}

// Compute non-overlapping suffix from incoming relative to committed using token-level overlap
function computeSuffixFromOverlap(committed, incoming) {
  const cToks = tokenizeWords(committed);
  const iToks = tokenizeWords(incoming);
  const n = Math.min(cToks.length, iToks.length);
  let k = 0;
  for (let m = n; m >= 1; m--) {
    let ok = true;
    for (let j = 0; j < m; j++) {
      if (cToks[cToks.length - m + j].norm !== iToks[j].norm) { ok = false; break; }
    }
    if (ok) { k = m; break; }
  }
  return joinTokens(iToks.slice(k));
}

// ---- WebSocket connection ----
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/${encodeURIComponent(SESSION_ID)}`;
  
  websocket = new WebSocket(wsUrl);
  websocket.binaryType = "arraybuffer";
  
  websocket.onopen = () => {
    isConnected = true;
    setState({ 
      label: "WebSocket connected! Tap to start recording.", 
      dot: "", 
      micClass: "", 
      barsClass: "" 
    });
  };
  
  websocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.status === "success") {
        setState({ 
          label: `Audio saved: ${data.filename}`, 
          dot: "play", 
          micClass: "play", 
          barsClass: "" 
        });
        setTimeout(() => {
          setState({ 
            label: "Tap to start recording.", 
            dot: "", 
            micClass: "", 
            barsClass: "" 
          });
        }, 2000);

      } else if (data.status === "error") {
        setState({ 
          label: "Error saving audio.", 
          dot: "err", 
          micClass: "", 
          barsClass: "", 
          error: data.message 
        });

      } else if (data.status === "transcript") {
        const raw = (data.text || "").trim();
        const cleaned = dedupeRepeatedTailString(raw);
        const suffix = computeSuffixFromOverlap(transcriptCommitted, cleaned);
        if (data.final) {
          if (suffix) transcriptCommitted = joinWithSpace(transcriptCommitted, suffix);
          currentPartial = "";
        } else {
          currentPartial = suffix;
        }
        renderTranscript();
        console.log("[ws<-] transcript:", cleaned, "final=", !!data.final, "suffix=", JSON.stringify(suffix));
      } else if (data.status === "turn_end") {
        // Server signals explicit end-of-turn; ensure partial is cleared and UI reflects final text
        currentPartial = "";
        // Optionally, if server includes final text, ensure it's committed without duplication
        const incoming = (data.text || "").trim();
        if (incoming) {
          const suffix = computeSuffixFromOverlap(transcriptCommitted, incoming);
          if (suffix) transcriptCommitted = joinWithSpace(transcriptCommitted, suffix);
        }
        renderTranscript();
        setState({ label: "Turn ended. Tap to speak again.", dot: "", micClass: "", barsClass: "" });
        console.log("[ws<-] turn_end");
      }

    } catch (e) {
      console.error("Failed to parse WebSocket message:", e);
    }
  };
  
  websocket.onclose = () => {
    isConnected = false;
    setState({ 
      label: "WebSocket disconnected. Reconnecting...", 
      dot: "err", 
      micClass: "", 
      barsClass: "" 
    });
    setTimeout(connectWebSocket, 3000);
  };
  
  websocket.onerror = (error) => {
    console.error("WebSocket error:", error);
    setState({ 
      label: "WebSocket error.", 
      dot: "err", 
      micClass: "", 
      barsClass: "", 
      error: "Connection failed" 
    });
  };
}

// ---- UI state helpers ----
function setState({label, dot, micClass, barsClass, error=null}) {
  if (label !== undefined) statusEl.textContent = label;
  if (dot !== undefined)   dotEl.className = `dot ${dot || ""}`;
  if (micClass !== undefined)  micBtn.className = `mic ${micClass || ""}`;
  if (barsClass !== undefined) barsEl.className = `bars ${barsClass || ""}`;
  if (error !== undefined) { errEl.textContent = error || ""; }
}

// ---- Recording control with WebSocket streaming ----
micBtn.addEventListener("click", async () => {
  if (!isConnected) {
    setState({ 
      label: "WebSocket not connected.", 
      dot: "err", 
      micClass: "", 
      barsClass: "", 
      error: "Please wait for connection..." 
    });
    return;
  }
  
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Create audio pipeline for real-time PCM streaming
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    inputSampleRate = audioCtx.sampleRate;
    console.log(`[audio] input sample rate: ${inputSampleRate} Hz`);

    sourceNode = audioCtx.createMediaStreamSource(stream);
    const bufferSize = 4096; // ~85ms at 48k
    processorNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);

    bytesSentTotal = 0;
    processorNode.onaudioprocess = (e) => {
      if (!isRecording || !isConnected) return;
      try {
        const input = e.inputBuffer.getChannelData(0);
        const pcm16 = downsampleTo16kPCM(input, inputSampleRate);
        if (!pcm16 || pcm16.length === 0) return;
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(pcm16.buffer);
          bytesSentTotal += pcm16.byteLength;
          if (bytesSentTotal % (32000) === 0) { // log roughly every 1 sec of audio (16k * 2 bytes)
            console.log(`[ws->] sent total ${bytesSentTotal} bytes (~${Math.round(bytesSentTotal / 32000)}s)`);
          }
        }
      } catch (err) {
        console.error("[audio] processing error:", err);
      }
    };

    sourceNode.connect(processorNode);
    // Connect to destination at very low gain to avoid feedback; or just connect without audible output
    processorNode.connect(audioCtx.destination);

    // Reset transcript state for this recording session
    transcriptAccum = "";
    lastPartial = "";
    transcriptCommitted = "";
    currentPartial = "";
    if (transcriptEl) transcriptEl.textContent = "";
    isRecording = true;
    setState({ label: "Recording… tap to stop.", dot: "rec", micClass: "rec", barsClass: "rec" });

  } catch (e) {
    console.error("[audio] startRecording failed:", e);
    setState({ label: "Mic permission needed.", dot: "err", micClass: "", barsClass: "", error: "Allow microphone access." });
  }
}

function stopRecording() {
  // Stop legacy recorder if ever started
  if (recorder && recorder.state && recorder.state !== "inactive") recorder.stop();
  isRecording = false;

  setTimeout(() => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({"event": "end_of_audio"}));
      setState({ label: "Saving audio file...", dot: "proc", micClass: "proc", barsClass: "proc" });
      pcmBuffer = []; // clear leftover PCM
      console.log(`[ws->] end_of_audio sent. total bytes=${bytesSentTotal}`);
    } else {
      setState({ label: "WebSocket not connected.", dot: "err", micClass: "", barsClass: "", error: "Cannot save audio" });
    }
  }, 1000);

  try {
    if (sourceNode) sourceNode.disconnect();
    if (processorNode) processorNode.disconnect();
    if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
  } catch {}
  // Stop mic tracks
  try { if (recorder && recorder.stream) recorder.stream.getTracks().forEach(track => track.stop()); } catch {}
}

// ---- Initialize WebSocket connection on page load ----
document.addEventListener('DOMContentLoaded', () => {
  setState({ label: "Connecting to WebSocket...", dot: "proc", micClass: "", barsClass: "" });
  connectWebSocket();
});

// ---- Cleanup on page unload ----
window.addEventListener('beforeunload', () => {
  if (websocket) websocket.close();
  if (recorder && recorder.state !== "inactive") recorder.stop();
});
