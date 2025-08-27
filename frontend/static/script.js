// Day 23: Complete Voice Agent
// Complete conversational AI voice agent with real-time transcription, LLM responses, and TTS streaming

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
console.log("[DOM] transcriptEl found:", transcriptEl);
const conversationEl = document.getElementById("conversation"); // conversation history container
const conversationContentEl = document.getElementById("conversation-content"); // conversation content

// Debug: Check if conversation elements exist
console.log("[DOM] conversationEl:", conversationEl);
console.log("[DOM] conversationContentEl:", conversationContentEl);
const waveformCanvas = document.getElementById("waveform");
let waveformCtx = null;
let transcriptAccum = ""; // kept for compatibility
let lastPartial = "";     // kept to avoid large diff

// New model: committed transcript + live partial
let transcriptCommitted = "";
let currentPartial = "";

// Conversation history tracking
let conversationHistory = [];

// Day 21: Audio chunk accumulation
let audioChunks = [];
let isReceivingAudio = false;
let audioBuffer = []; // Buffer for smooth playback
let isPlayingAudio = false;
let ttsFallbackTimer = null;
let awaitingTtsAudio = false;

/* -------------------- Day 22: PCM16 streaming playback helpers -------------------- */
let playbackCtx = null;
let nextPlaybackTime = 0; // chain buffers end-to-end

function ensurePlaybackCtx() {
  if (!playbackCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    playbackCtx = new AC();
    nextPlaybackTime = playbackCtx.currentTime;
    console.log(`[audio] Playback context created with sample rate: ${playbackCtx.sampleRate}Hz`);
  } else if (playbackCtx.state === "suspended") {
    console.log("[audio] Resuming suspended audio context");
    playbackCtx.resume();
  }
  
  console.log(`[audio] Audio context state: ${playbackCtx.state}, sample rate: ${playbackCtx.sampleRate}Hz`);
  return playbackCtx;
}

// base64 -> Uint8Array
function base64ToUint8(b64) {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// Ensure consistent sample rate for seamless playback
function normalizeSampleRate(audioData, fromSampleRate, toSampleRate = 44100) {
  if (fromSampleRate === toSampleRate) {
    return audioData;
  }
  
  const ratio = fromSampleRate / toSampleRate;
  const newLength = Math.round(audioData.length / ratio);
  const result = new Float32Array(newLength);
  
  for (let i = 0; i < newLength; i++) {
    const srcIndex = Math.round(i * ratio);
    result[i] = audioData[srcIndex] || 0;
  }
  
  return result;
}

// Int16 -> Float32 normalized [-1, 1]
function int16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    const s = int16[i] / 0x8000;
    out[i] = Math.max(-1, Math.min(1, s));
  }
  return out;
}

/**
 * Play one PCM16 mono chunk (base64) at the given sampleRate.
 * Creates an AudioBuffer and schedules it at `nextPlaybackTime` to keep playback seamless.
 */
function playPcm16ChunkBase64(b64, sampleRate = 44100) {
  console.log("[AUDIO] playPcm16ChunkBase64 called with sampleRate:", sampleRate, "base64 length:", b64 ? b64.length : 0);
  
  const ctx = ensurePlaybackCtx();
  try { ctx.resume(); } catch {}

  const u8 = base64ToUint8(b64);
  const int16 = new Int16Array(u8.buffer, u8.byteOffset, Math.floor(u8.byteLength / 2));
  console.log("[AUDIO] Converted to Int16Array, length:", int16.length);
  
  if (!int16.length) {
    console.log("[AUDIO] Empty Int16Array, returning early");
    return;
  }

  const f32 = int16ToFloat32(int16);
  
  // Normalize sample rate to 44.1kHz for consistent playback
  const targetSampleRate = 44100;
  const normalizedF32 = normalizeSampleRate(f32, sampleRate, targetSampleRate);
  
  // Create buffer with the normalized sample rate
  const buf = ctx.createBuffer(1, normalizedF32.length, targetSampleRate);
  buf.getChannelData(0).set(normalizedF32);

  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);

  const now = ctx.currentTime;
  // Ensure proper timing with minimal gaps for seamless playback
  const startAt = Math.max(nextPlaybackTime, now + 0.001); // minimal lead-in for seamless audio
  src.start(startAt);
  
  // Calculate exact duration and set next start time using normalized sample rate
  const duration = buf.length / targetSampleRate;
  nextPlaybackTime = startAt + duration;
  
  console.log(`[audio] Playing chunk: ${normalizedF32.length} samples at ${targetSampleRate}Hz (normalized from ${sampleRate}Hz), duration: ${duration.toFixed(3)}s, next: ${nextPlaybackTime.toFixed(3)}s`);
}


function isLikelyWav(u8) {
  // "RIFF" header = 0x52 0x49 0x46 0x46, "WAVE" at bytes 8..11
  if (!u8 || u8.length < 12) {
    console.log("[WAV] Not WAV: insufficient data length");
    return false;
  }
  
  const isRiff = u8[0] === 0x52 && u8[1] === 0x49 && u8[2] === 0x46 && u8[3] === 0x46;
  const isWave = u8[8] === 0x57 && u8[9] === 0x41 && u8[10] === 0x56 && u8[11] === 0x45;
  
  console.log("[WAV] Header analysis:");
  console.log("  - RIFF header:", isRiff ? "YES" : "NO", `(${String.fromCharCode(u8[0], u8[1], u8[2], u8[3])})`);
  console.log("  - WAVE header:", isWave ? "YES" : "NO", `(${String.fromCharCode(u8[8], u8[9], u8[10], u8[11])})`);
  console.log("  - Final result:", isRiff && isWave ? "WAV DETECTED" : "NOT WAV");
  
  return isRiff && isWave;
}

async function playWavBase64(b64) {
  const ctx = ensurePlaybackCtx();
  try { await ctx.resume(); } catch {}

  const u8 = base64ToUint8(b64);
  const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);

  // decodeAudioData handles WAV/MP3/OGG, etc.
  const audioBuffer = await new Promise((resolve, reject) => {
    // Safari supports the promise form, but the callback form keeps us compatible
    const cb = (buf) => resolve(buf);
    const eb = (err) => reject(err);
    const ret = ctx.decodeAudioData(ab, cb, eb);
    if (ret && typeof ret.then === "function") {
      ret.then(resolve).catch(reject);
    }
  });

  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(ctx.destination);

  const now = ctx.currentTime;
  const startAt = Math.max(nextPlaybackTime, now + 0.005);
  src.start(startAt);

  const duration = audioBuffer.duration;
  nextPlaybackTime = startAt + duration;
}

function playAudioChunkBase64(b64, sampleRate = 44100) {
  // Always return a Promise for uniform flow
  return (async () => {
    const u8 = base64ToUint8(b64);
    console.log("[AUDIO] Audio chunk analysis:");
    console.log("  - Base64 length:", b64 ? b64.length : 0);
    console.log("  - Uint8Array length:", u8 ? u8.length : 0);
    console.log("  - First few bytes:", u8 ? Array.from(u8.slice(0, 16)).map(b => '0x' + b.toString(16).padStart(2, '0')).join(' ') : 'null');
    
    if (isLikelyWav(u8)) {
      console.log("[AUDIO] Detected WAV format - using decodeAudioData");
      // Preferred path for Murf since you requested format=WAV
      await playWavBase64(b64);
    } else {
      console.log("[AUDIO] Detected raw PCM format - using PCM16 decoder");
      // Fallback to your existing raw PCM16 path
      playPcm16ChunkBase64(b64, sampleRate);
    }
  })();
}

// Smooth audio buffer playback with uniform sample rate
function playAudioBuffer() {
  console.log("[AUDIO] playAudioBuffer called. Buffer length:", audioBuffer.length, "isPlayingAudio:", isPlayingAudio);
  
  if (audioBuffer.length === 0 || isPlayingAudio) {
    console.log("[AUDIO] Early return - buffer empty or already playing");
    return;
  }
  
  isPlayingAudio = true;
  console.log("[AUDIO] Starting audio buffer playback");
  
  // Use a consistent sample rate (44.1kHz) for all chunks
  const targetSampleRate = 44100;
  
  function playNextChunk() {
    console.log("[AUDIO] playNextChunk called. Buffer length:", audioBuffer.length);
    
    if (audioBuffer.length === 0) {
      isPlayingAudio = false;
      console.log("[AUDIO] Audio buffer playback completed");
      return;
    }
    
    const chunk = audioBuffer.shift();
    console.log("[AUDIO] Playing chunk:", chunk);
    
    try {
      // Normalize sample rate to target if different
      const sampleRate = chunk.sampleRate || targetSampleRate;
      console.log("[AUDIO] Calling playPcm16ChunkBase64 with sampleRate:", sampleRate);
      playAudioChunkBase64(chunk.base64, sampleRate)
        .then(() => setTimeout(playNextChunk, 10))
        .catch((err) => {
          console.warn("[AUDIO] playAudioChunkBase64 failed, continuing:", err);
          setTimeout(playNextChunk, 10);
        });
    } catch (err) {
      console.warn("[AUDIO] Chunk play failed:", err);
      playNextChunk(); // Continue with next chunk
    }
  }
  
  playNextChunk();
}

// Simple Web Speech API fallback
function speakWithWebSpeech(text) {
  try {
    if (!window.speechSynthesis) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0;
    utter.pitch = 1.0;
    utter.lang = 'en-US';
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  } catch {}
}
/* --------------------------------------------------------------------------- */

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
console.log("[SESSION] Session ID generated:", SESSION_ID);

function joinWithSpace(a, b) {
  if (!a) return b || "";
  if (!b) return a || "";
  const needsSpace = !(/[\s\n]$/.test(a)) && !(/^[\p{P}\s]/u.test(b));
  return a + (needsSpace ? " " : "") + b;
}

function renderTranscript() {
  const text = joinWithSpace(transcriptCommitted, currentPartial);
  console.log("[TRANSCRIPT] Rendering transcript:", { transcriptCommitted, currentPartial, combined: text });
  if (transcriptEl) {
    transcriptEl.textContent = text;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
    console.log("[TRANSCRIPT] Updated transcript element with text:", text);
  } else {
    console.log("[TRANSCRIPT] transcriptEl is null, cannot render");
  }
}

function addToConversation(role, text) {
  if (!text || !text.trim()) return;
  
  console.log(`[CONVERSATION] Adding ${role} message: "${text}"`);
  conversationHistory.push({ role, text: text.trim(), timestamp: new Date() });
  console.log(`[CONVERSATION] History length after adding: ${conversationHistory.length}`);
  
  renderConversation();
  
  // Show conversation container after first message (not waiting for 2)
  if (conversationHistory.length >= 1) {
    console.log(`[CONVERSATION] Showing conversation container. Current display: ${conversationEl.style.display}`);
    conversationEl.style.display = "block";
    console.log(`[CONVERSATION] Conversation container display set to: ${conversationEl.style.display}`);
  }
}

function renderConversation() {
  console.log(`[RENDER] Rendering conversation with ${conversationHistory.length} messages`);
  if (!conversationContentEl) {
    console.log("[RENDER] conversationContentEl is null, cannot render");
    return;
  }
  
  const html = conversationHistory.map((msg, index) => {
    const time = msg.timestamp.toLocaleTimeString();
    const isUser = msg.role === "user";
    const bgColor = isUser ? "rgba(96,165,250,0.15)" : "rgba(52,211,153,0.15)";
    const borderColor = isUser ? "rgba(96,165,250,0.4)" : "rgba(52,211,153,0.4)";
    const textColor = isUser ? "#60a5fa" : "#34d399";
    const align = isUser ? "flex-end" : "flex-start";
    const borderRadius = isUser ? "12px 12px 4px 12px" : "12px 12px 12px 4px";
    
    return `
      <div style="
        display: flex;
        justify-content: ${align};
        margin-bottom: 8px;
      ">
        <div style="
          max-width: 80%;
          padding: 10px 14px; 
          background: ${bgColor}; 
          border: 1px solid ${borderColor}; 
          border-radius: ${borderRadius};
          font-size: 14px;
          line-height: 1.4;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
          <div style="color: var(--text); margin-bottom: 2px;">${msg.text}</div>
          <div style="color: ${textColor}; font-size: 11px; opacity: 0.8;">
            ${time}
          </div>
        </div>
      </div>
    `;
  }).join("");
  
  conversationContentEl.innerHTML = html;
  conversationEl.scrollTop = conversationEl.scrollHeight;
  console.log(`[RENDER] Conversation rendered. HTML length: ${html.length}, Content element innerHTML length: ${conversationContentEl.innerHTML.length}`);
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
  console.log("[WS] Connecting to WebSocket URL:", wsUrl);
  
  websocket = new WebSocket(wsUrl);
  websocket.binaryType = "arraybuffer";
  
  websocket.onopen = () => {
    console.log("[WS] WebSocket connected successfully");
    isConnected = true;
    setState({ 
      label: "Hi there! I'm ready to chat. Just tap the microphone and start talking!", 
      dot: "", 
      micClass: "idle", 
      barsClass: "" 
    });
    if (playbackCtx) nextPlaybackTime = playbackCtx.currentTime;
    console.log("[WS] WebSocket connection established and ready");
  };
  
  websocket.onmessage = (event) => {
    console.log("[WS] Received WebSocket message:", event.data);
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

      } else if (data.status === "welcome") {
        // Handle welcome message from Orion
        console.log("[ws<-] Welcome message received:", data.text);
        console.log("[ws<-] Adding welcome message to conversation...");
        addToConversation("assistant", data.text);
        console.log("[ws<-] Welcome message added to conversation. History length:", conversationHistory.length);
        setState({ 
          label: "Orion is ready to chat!", 
          dot: "", 
          micClass: "idle", 
          barsClass: "" 
        });
        console.log("[ws<-] State updated for welcome message");

      } else if (data.status === "transcript") {
        console.log("[ws<-] Received transcript message:", data);
        const raw = (data.text || "").trim();
        const cleaned = dedupeRepeatedTailString(raw);
        const suffix = computeSuffixFromOverlap(transcriptCommitted, cleaned);
        console.log("[ws<-] Transcript processing:", { raw, cleaned, suffix, final: data.final });
        
        if (data.final) {
          if (suffix) transcriptCommitted = joinWithSpace(transcriptCommitted, suffix);
          currentPartial = "";
          console.log("[ws<-] Final transcript committed:", transcriptCommitted);
          // Add user message to conversation when final (only once)
          if (transcriptCommitted.trim() && !conversationHistory.some(msg => msg.text === transcriptCommitted.trim())) {
            addToConversation("user", transcriptCommitted);
          }
        } else {
          currentPartial = suffix;
          console.log("[ws<-] Partial transcript updated:", currentPartial);
        }
        renderTranscript();
        console.log("[ws<-] transcript:", cleaned, "final=", !!data.final, "suffix=", JSON.stringify(suffix));

      } else if (data.status === "turn_end") {
        currentPartial = "";
        const incoming = (data.text || "").trim();
        if (incoming) {
          const suffix = computeSuffixFromOverlap(transcriptAccum, incoming);
          if (suffix) transcriptAccum = joinWithSpace(transcriptAccum, suffix);
        }
        renderTranscript();
        setState({ label: "Thinking about your question...", dot: "proc", micClass: "proc", barsClass: "proc" });
        console.log("[ws<-] turn_end - waiting for LLM response");

      } else if (data.status === "audio_chunk") {
        console.log("[ws<-] AUDIO CHUNK RECEIVED:", data);
        console.log("[ws<-] Audio chunk data validation:");
        console.log("  - audio_base64 exists:", !!data.audio_base64);
        console.log("  - audio_base64 length:", data.audio_base64 ? data.audio_base64.length : 0);
        console.log("  - chunk_index:", data.chunk_index);
        console.log("  - sample_rate:", data.sample_rate);
        
        // Cancel fallback if streaming audio arrived
        if (awaitingTtsAudio && ttsFallbackTimer) {
          clearTimeout(ttsFallbackTimer);
          ttsFallbackTimer = null;
          awaitingTtsAudio = false;
          console.log("[ws<-] Cancelled TTS fallback timer");
        }
        
        if (!isReceivingAudio) {
          isReceivingAudio = true;
          audioChunks = [];
          audioBuffer = []; // Clear any old audio
          console.log("[ws<-] Starting audio stream reception");
          ensurePlaybackCtx();
        }
        
        const chunkIndex = data.chunk_index;
        const audioBase64 = data.audio_base64;
        const totalLength = data.total_length;
        const sr = Number(data.sample_rate) || 44100;
        
        console.log(`[ws<-] Processing audio chunk: index=${chunkIndex}, base64_length=${audioBase64 ? audioBase64.length : 0}, sample_rate=${sr}`);
        
        // Validate base64 data
        if (!audioBase64 || typeof audioBase64 !== 'string') {
          console.error(`[ws<-] ERROR: Invalid audio_base64 data:`, audioBase64);
          return;
        }
        
        // Check if base64 looks valid (should start with common audio prefixes)
        const isValidBase64 = /^[A-Za-z0-9+/]*={0,2}$/.test(audioBase64);
        console.log(`[ws<-] Base64 validation: ${isValidBase64 ? 'PASS' : 'FAIL'}`);
        
        // Store chunk in array (following Murf cookbook pattern)
        audioChunks[chunkIndex - 1] = audioBase64;
        console.log(`[ws<-] Received audio chunk ${chunkIndex}, len=${totalLength}, sr=${sr}Hz`);

        // Add to buffer for smooth playback (preserve original sample rate)
        audioBuffer.push({ 
          base64: audioBase64, 
          sampleRate: sr,
          chunkIndex: chunkIndex
        });
        
        console.log(`[ws<-] Added to audio buffer. Buffer length: ${audioBuffer.length}`);
        console.log(`[ws<-] Audio buffer contents:`, audioBuffer.map(chunk => ({ 
          chunkIndex: chunk.chunkIndex, 
          base64Length: chunk.base64 ? chunk.base64.length : 0,
          sampleRate: chunk.sampleRate 
        })));
        
        // Start playing if not already playing
        if (!isPlayingAudio) {
          console.log("[ws<-] Starting audio playback");
          console.log("[ws<-] Audio context state before playback:", playbackCtx ? playbackCtx.state : "null");
          playAudioBuffer();
        } else {
          console.log("[ws<-] Audio already playing, chunk queued");
        }

        // Ack
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(JSON.stringify({
            "status": "audio_chunk_ack",
            "chunk_index": chunkIndex,
            "received": true,
            "timestamp": Date.now()
          }));
          console.log(`[ws->] Sent acknowledgment for chunk ${chunkIndex}`);
        }
        
              } else if (data.status === "audio_complete") {
        const totalChunks = data.total_chunks;
        isReceivingAudio = false;
        console.log(`[ws<-] Audio streaming completed. Total chunks: ${totalChunks}`);

        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(JSON.stringify({
            "status": "audio_complete_ack",
            "total_chunks_received": audioChunks.filter(Boolean).length,
            "expected_chunks": totalChunks,
            "timestamp": Date.now()
          }));
          console.log(`[ws->] Sent completion acknowledgment`);
        }
        
        // Don't clear audio buffer immediately - let current playback finish
        console.log("[ws<-] Audio streaming completed, letting current playback finish");
        awaitingTtsAudio = false;
        
        // Add AI response to conversation history
        if (conversationHistory.length > 0 && conversationHistory[conversationHistory.length - 1].role === "user") {
          // Find the last AI response from the backend
          const lastUserMsg = conversationHistory[conversationHistory.length - 1];
          // We'll add the AI response when we get the LLM completion message
        }
        
        // Reset mic to idle (blue) after response complete
        setState({ 
          label: `What would you like to know next?`, 
          dot: "", 
          micClass: "idle", 
          barsClass: "" 
        });

        // Reset chain pointer for next response (current audio continues)
        nextPlaybackTime = playbackCtx ? playbackCtx.currentTime : 0;

        setTimeout(() => { audioChunks = []; }, 1000);
              } else if (data.status === "llm_chunk") {
        // Accumulate LLM response chunks
        const chunk = data.text || "";
        if (chunk) {
          console.log(`[ws<-] LLM chunk received: "${chunk}"`);
          // Store the chunk for later assembly
          if (!window.llmResponseChunks) {
            window.llmResponseChunks = [];
          }
          window.llmResponseChunks.push(chunk);
        }
      } else if (data.status === "llm_complete") {
        // Assemble the complete response from accumulated chunks
        let ai_response = "";
        if (window.llmResponseChunks && window.llmResponseChunks.length > 0) {
          ai_response = window.llmResponseChunks.join("");
          console.log("[ws<-] Assembled complete response from chunks:", ai_response);
          // Clear the chunks for next response
          window.llmResponseChunks = [];
        } else {
          // Fallback to the text field if no chunks were received
          ai_response = data.text || "";
        }
        
        if (ai_response && !conversationHistory.some(msg => msg.text === ai_response)) {
          addToConversation("assistant", ai_response);
          console.log("[ws<-] Added AI response to conversation history");
        }
        
        console.log("[ws<-] LLM response completed:", ai_response);
        
        // Start a short fallback timer: if no audio arrives, speak via Web Speech
        if (ai_response) {
          awaitingTtsAudio = true;
          if (ttsFallbackTimer) { try { clearTimeout(ttsFallbackTimer); } catch {} }
          ttsFallbackTimer = setTimeout(() => {
            if (awaitingTtsAudio) {
              try { speakWithWebSpeech(ai_response); } catch (e) { console.warn("[tts-fallback] speech failed", e); }
              awaitingTtsAudio = false;
            }
          }, 2000);
        }
      } else if (data.status === "audio_fallback") {
        console.log("[ws<-] Received fallback audio URL:", data.audio_url);
        // Handle fallback audio from regular Murf API
        try {
          const audio = new Audio(data.audio_url);
          audio.play();
          console.log("[ws<-] Playing fallback audio");
        } catch (error) {
          console.error("[ws<-] Failed to play fallback audio:", error);
        }
      } else if (data.status === "ready_for_next") {
        console.log("[ws<-] Ready for next recording");
        // The system is ready for the next recording session
        
        // Debug: Check conversation history state
        console.log("[DEBUG] Conversation history length:", conversationHistory.length);
        console.log("[DEBUG] Conversation container display:", conversationEl ? conversationEl.style.display : "null");
        console.log("[DEBUG] Conversation content element:", conversationContentEl);
      }

    } catch (e) {
      console.error("Failed to parse WebSocket message:", e);
    }
  };
  
  websocket.onclose = () => {
    isConnected = false;
    setState({ 
      label: "Oops! Lost connection. Let me reconnect...", 
      dot: "err", 
      micClass: "", 
      barsClass: "" 
    });
    if (playbackCtx) nextPlaybackTime = playbackCtx.currentTime;
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
  if (label !== undefined && statusEl) statusEl.textContent = label;
  if (dot !== undefined && dotEl)   dotEl.className = `dot ${dot || ""}`;
  if (micClass !== undefined && micBtn)  micBtn.className = `mic ${micClass || ""}`;
  // barsEl is optional (replaced by waveform). Guard it.
  if (barsClass !== undefined && barsEl) barsEl.className = `bars ${barsClass || ""}`;
  if (error !== undefined && errEl) { errEl.textContent = error || ""; }
}

// Test audio context functionality
document.getElementById("testAudio").addEventListener("click", () => {
  console.log("[TEST] Testing audio context...");
  const ctx = ensurePlaybackCtx();
  console.log("[TEST] Audio context state:", ctx.state);
  console.log("[TEST] Audio context sample rate:", ctx.sampleRate);
  
  // Try to create a simple test tone
  try {
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);
    
    oscillator.frequency.setValueAtTime(440, ctx.currentTime); // A4 note
    gainNode.gain.setValueAtTime(0.1, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
    
    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + 0.5);
    
    console.log("[TEST] Test tone played successfully");
  } catch (error) {
    console.error("[TEST] Failed to play test tone:", error);
  }
});

// ---- Recording control with WebSocket streaming ----
micBtn.addEventListener("click", async () => {
  // ensure playback is primed on user gesture (Safari/iOS)
  ensurePlaybackCtx();

  if (!isConnected) {
    setState({ 
      label: "Hold on, I'm still getting ready...", 
      dot: "err", 
      micClass: "", 
      barsClass: "", 
      error: "Please wait a moment..." 
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
          if (bytesSentTotal % (32000) === 0) { // log ~1s of audio (16k * 2 bytes)
            console.log(`[ws->] sent total ${bytesSentTotal} bytes (~${Math.round(bytesSentTotal / 32000)}s)`);
          }
        }
      } catch (err) {
        console.error("[audio] processing error:", err);
      }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioCtx.destination);

    // Reset transcript state for this recording session
    transcriptAccum = "";
    lastPartial = "";
    transcriptCommitted = "";
    currentPartial = "";
    if (transcriptEl) transcriptEl.textContent = "";
    isRecording = true;
    setState({ label: "I'm listening... tap to stop when you're done.", dot: "rec", micClass: "rec-green", barsClass: "rec" });

    // Show waveform and start analyser visual
    if (waveformCanvas) {
      waveformCanvas.style.display = "block";
      waveformCtx = waveformCanvas.getContext("2d");
      // Create analyser
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      const bufferLength = analyser.fftSize;
      const dataArray = new Uint8Array(bufferLength);
      sourceNode.connect(analyser);

      const draw = () => {
        if (!isRecording || !waveformCtx) return;
        requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);
        const W = waveformCanvas.width;
        const H = waveformCanvas.height;
        waveformCtx.clearRect(0, 0, W, H);
        waveformCtx.lineWidth = 2;
        waveformCtx.strokeStyle = "#60a5fa";
        waveformCtx.beginPath();
        const sliceWidth = W * 1.0 / bufferLength;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = v * H/2;
          if (i === 0) waveformCtx.moveTo(x, y);
          else waveformCtx.lineTo(x, y);
          x += sliceWidth;
        }
        waveformCtx.lineTo(W, H/2);
        waveformCtx.stroke();
      };
      draw();
    }

  } catch (e) {
    console.error("[audio] startRecording failed:", e);
    setState({ label: "I need to hear you! Please allow microphone access.", dot: "err", micClass: "", barsClass: "", error: "Click 'Allow' when prompted" });
  }
}

function stopRecording() {
  if (recorder && recorder.state && recorder.state !== "inactive") recorder.stop();
  isRecording = false;

  // Update mic to processing (red)
  setState({ label: "Got it! Let me process what you said...", dot: "proc", micClass: "proc-red", barsClass: "proc" });

  // Hide waveform
  if (waveformCanvas) {
    waveformCanvas.style.display = "none";
    waveformCtx = null;
  }

  // Send end_of_audio immediately without delay
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({"event": "end_of_audio"}));
    pcmBuffer = [];
    console.log(`[ws->] end_of_audio sent. total bytes=${bytesSentTotal}`);
  } else {
    setState({ 
      label: "Sorry, I'm having trouble connecting right now.", 
      dot: "err", 
      micClass: "", 
      barsClass: "", 
      error: "Please try again in a moment" 
    });
  }

  try {
    if (sourceNode) sourceNode.disconnect();
    if (processorNode) processorNode.disconnect();
    if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
  } catch {}
  try { if (recorder && recorder.stream) recorder.stream.getTracks().forEach(track => track.stop()); } catch {}
}

// ---- Initialize WebSocket connection on page load ----
document.addEventListener('DOMContentLoaded', () => {
      setState({ 
      label: "Setting up our conversation...", 
      dot: "proc", 
      micClass: "", 
      barsClass: "" 
    });
  connectWebSocket();
});

// ---- Cleanup on page unload ----
window.addEventListener('beforeunload', () => {
  if (websocket) websocket.close();
  if (recorder && recorder.state !== "inactive") recorder.stop();
  try { if (audioCtx && audioCtx.state !== 'closed') audioCtx.close(); } catch {}
  try { if (playbackCtx && playbackCtx.state !== 'closed') playbackCtx.close(); } catch {}
});
