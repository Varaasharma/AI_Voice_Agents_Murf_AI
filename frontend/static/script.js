// Day 12: Single-button conversational UI with animations + auto-loop
// Uses /agent/chat/{session_id}. Plays Murf audio if available, else speaks fallback text.

let recorder;
let chunks = [];
let isRecording = false;

const micBtn   = document.getElementById("micBtn");
const statusEl = document.getElementById("status");
const dotEl    = document.getElementById("dot");
const barsEl   = document.getElementById("bars");
const errEl    = document.getElementById("error");
const sidEl    = document.getElementById("sid");
const audioEl  = document.getElementById("llmAudio");

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

// ---- UI state helpers ----
function setState({label, dot, micClass, barsClass, error=null}) {
  statusEl.textContent = label;
  dotEl.className = `dot ${dot || ""}`;
  micBtn.className = `mic ${micClass || ""}`;
  barsEl.className = `bars ${barsClass || ""}`;
  if (error) { errEl.textContent = error; } else { errEl.textContent = ""; }
}

function speakFallback(text, details) {
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.onstart = () => setState({
      label: "Playing fallback (browser voice)…",
      dot: "err", micClass: "", barsClass: ""
    });
    u.onend = () => setTimeout(() => startRecording(), 280);
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  } catch {
    setState({ label: "Fallback failed.", dot: "err", micClass: "", barsClass: "", error: details || "Browser TTS unavailable." });
  }
}

// ---- Recording control (single button) ----
micBtn.addEventListener("click", async () => {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    try { recorder = new MediaRecorder(stream, { mimeType: "audio/webm" }); }
    catch { recorder = new MediaRecorder(stream); }

    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      await sendTurn(blob);
    };

    recorder.start();
    isRecording = true;
    setState({ label: "Recording… tap to stop.", dot: "rec", micClass: "rec", barsClass: "rec" });
  } catch (e) {
    setState({ label: "Mic permission needed.", dot: "err", micClass: "", barsClass: "", error: "Allow microphone access." });
  }
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") {
    recorder.stop();
  }
  isRecording = false;
  setState({ label: "Thinking… (history → LLM → TTS)", dot: "proc", micClass: "proc", barsClass: "proc" });
}

// ---- Send turn to server ----
async function sendTurn(blob) {
  const form = new FormData();
  form.append("file", blob, "turn.webm");

  try {
    const res = await fetch(`/agent/chat/${encodeURIComponent(SESSION_ID)}`, {
      method: "POST",
      body: form
    });
    const data = await res.json();

    // Case A: Murf audio URL returned
    if (data.audio_url) {
      audioEl.src = data.audio_url;
      audioEl.onended = () => setTimeout(() => startRecording(), 280);
      setState({ label: "Playing assistant reply…", dot: "play", micClass: "play", barsClass: "" });
      audioEl.play().catch(() => {}); // user has already interacted
      return;
    }

    // Case B: server asks client to speak fallback text
    if (data.speak_text) {
      speakFallback(data.speak_text, data.details);
      return;
    }

    // Case C: unexpected
    setState({ label: "No audio returned.", dot: "err", micClass: "", barsClass: "", error: "Check server logs." });

  } catch (e) {
    console.error(e);
    speakFallback("I'm having trouble connecting right now.");
  }
}