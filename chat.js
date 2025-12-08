/*// ======== Chat UI Script (final corrected) ========

const chatMessages = document.getElementById("chatMessages");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const replyAudio = document.getElementById("replyAudio");
const speakerBtn = document.getElementById("speakerBtn");
const logoutBtn = document.getElementById("logoutBtn");
const chatInputElements = [textInput, sendBtn, micBtn];

let audioEnabled = true;
let recorder = null;
let chunks = [];
let isRecording = false;

// --- Logout ---
logoutBtn.addEventListener("click", () => {
  window.location.href = "/logout"; // backend redirects to /login
});

// --- Speaker Toggle ---
speakerBtn.addEventListener("click", () => {
  audioEnabled = !audioEnabled;
  speakerBtn.classList.toggle("off", !audioEnabled);
  const icon = speakerBtn.querySelector("i");
  icon.className = audioEnabled ? "fa-solid fa-volume-up" : "fa-solid fa-volume-xmark";
});

// --- Utility ---
function setInputState(disabled) {
  chatInputElements.forEach(el => (el.disabled = disabled));
}
function safeMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(String(text || "")));
}
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- Add Message ---
function addMessage(text, sender, imageUrl = null) {
  if (!text && !imageUrl) return;
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  if (text) div.innerHTML = safeMarkdown(text);
  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    img.loading = "lazy";
    div.appendChild(img);
  }
  chatMessages.appendChild(div);
  scrollToBottom();
}

// --- Indicators ---
function showTyping() {
  if (document.getElementById("typingIndicator")) return;
  const div = document.createElement("div");
  div.className = "typing";
  div.id = "typingIndicator";
  div.textContent = "🤖 Bot is typing...";
  chatMessages.appendChild(div);
  scrollToBottom();
}
function hideTyping() {
  const indicator = document.getElementById("typingIndicator");
  if (indicator) indicator.remove();
}
function showListening() {
  if (document.getElementById("listeningIndicator")) return;
  const div = document.createElement("div");
  div.className = "listening";
  div.id = "listeningIndicator";
  div.textContent = "🎤 Listening...";
  chatMessages.appendChild(div);
  scrollToBottom();
}
function hideListening(transcript) {
  const indicator = document.getElementById("listeningIndicator");
  if (!indicator) return;
  if (transcript) {
    indicator.className = "message user";
    indicator.innerHTML = safeMarkdown(transcript);
  } else {
    indicator.remove();
  }
  scrollToBottom();
}

// --- Image Loader ---
function showImageLoader() {
  if (document.getElementById("imageLoader")) return;
  const div = document.createElement("div");
  div.className = "image-loader";
  div.id = "imageLoader";
  div.textContent = "🎨 Generating image… please wait";
  chatMessages.appendChild(div);
  scrollToBottom();
}
function hideImageLoader() {
  const loader = document.getElementById("imageLoader");
  if (loader) loader.remove();
}

// --- Bot Reply & Audio Handling ---
function addBotReply(data) {
  hideTyping();
  hideImageLoader();
  setInputState(false);

  if (data.reply || data.image_url) {
    addMessage(data.reply, "bot", data.image_url || null);
  }

  if (data.audio && audioEnabled) {
    const filenameWithQuery = data.audio.split("/").pop();
    const filename = filenameWithQuery.split("?")[0];

    let attempts = 0;
    const checkAudio = async () => {
      if (attempts++ > 30) return;
      try {
        const res = await fetch(`/audio-status/${filename}`, { cache: "no-store" });
        if (!res.ok) throw new Error("status check failed");
        const status = await res.json();
        if (status.ready) {
          replyAudio.src = data.audio;
          replyAudio.style.display = "none";
          replyAudio.play().catch(err => {
            replyAudio.controls = true;
            replyAudio.style.display = "block";
          });
        } else {
          setTimeout(checkAudio, 1000);
        }
      } catch {
        setTimeout(checkAudio, 1000);
      }
    };
    checkAudio();
  }
}

// --- Text Chat ---
async function sendMessage(question) {
  const q = String(question || "").trim();
  if (!q) return;

  addMessage(q, "user");
  textInput.value = "";
  setInputState(true);

  const isImageRequest = ["generate", "create", "draw", "illustrate", "design", "picture", "image", "diagram"]
    .some(k => q.toLowerCase().includes(k));

  if (isImageRequest) showImageLoader();
  else showTyping();

  try {
    const formData = new FormData();
    formData.append("question", q);
    formData.append("need_audio", audioEnabled ? "true" : "false");

    const res = await fetch("/chat", { method: "POST", body: formData });
    if (!res.ok) {
      let errorMsg = `Server error: ${res.status}`;
      try {
        const errData = await res.json();
        if (errData && errData.detail) errorMsg = errData.detail;
      } catch {}
      throw new Error(errorMsg);
    }

    const data = await res.json();
    addBotReply(data);
  } catch (err) {
    hideTyping();
    hideImageLoader();
    setInputState(false);
    addMessage(`❌ Error: ${err.message}`, "bot");
  }
}

// --- Event Listeners ---
sendBtn.addEventListener("click", () => sendMessage(textInput.value));
textInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage(textInput.value);
});

// --- Microphone Recording ---
micBtn.addEventListener("click", async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);
      chunks = [];

      showListening();

      recorder.ondataavailable = e => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });

        const formData = new FormData();
        formData.append("file", blob, "input.webm");
        // ✅ Always send as string, not boolean
        formData.append("need_audio", audioEnabled ? "true" : "false");

        try {
          showTyping();
          const res = await fetch("/chat-audio", { method: "POST", body: formData });
          const data = await res.json();
          hideListening(data.transcript);
          addBotReply(data);
        } catch (err) {
          hideTyping();
          hideListening(null);
          addMessage(`❌ Error: ${err.message}`, "bot");
        } finally {
          isRecording = false;
          micBtn.classList.remove("recording");
          // ✅ Reset icon back to microphone
          micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
          setInputState(false);
        }
      };

      recorder.start();
      isRecording = true;
      micBtn.classList.add("recording");
      // ✅ Show stop icon while recording
      micBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
    } catch (err) {
      hideListening(null);
      setInputState(false);
      alert("Microphone access denied or not available.");
    }
  } else {
    recorder.stop();
  }
});*/





const chatMessages = document.getElementById("chatMessages");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const replyAudio = document.getElementById("replyAudio");
const speakerBtn = document.getElementById("speakerBtn");
const logoutBtn = document.getElementById("logoutBtn");
const chatInputElements = [textInput, sendBtn, micBtn];

let audioEnabled = true;
let recorder, chunks = [], isRecording = false;

// --- Logout ---
logoutBtn.addEventListener("click", () => {
  window.location.href = "/login";  // redirect back to login page
});

// --- Speaker Toggle ---
speakerBtn.addEventListener("click", () => {
  audioEnabled = !audioEnabled;
  speakerBtn.classList.toggle("off", !audioEnabled);
  const icon = speakerBtn.querySelector("i");
  icon.className = audioEnabled ? "fa-solid fa-volume-up" : "fa-solid fa-volume-xmark";
});

// --- Utility ---
function setInputState(disabled) {
  chatInputElements.forEach(el => el.disabled = disabled);
}

// --- Add Message ---
function addMessage(text, sender, imageUrl = null) {
  if (!text && !imageUrl) return;
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  if (text) div.innerHTML = DOMPurify.sanitize(marked.parse(text));
  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    div.appendChild(img);
  }
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- Indicators ---
function showTyping() {
  if (document.getElementById("typingIndicator")) return;
  const div = document.createElement("div");
  div.className = "typing";
  div.id = "typingIndicator";
  div.textContent = "🤖 Bot is typing...";
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
function hideTyping() {
  const indicator = document.getElementById("typingIndicator");
  if (indicator) indicator.remove();
}
function showListening() {
  if (document.getElementById("listeningIndicator")) return;
  const div = document.createElement("div");
  div.className = "listening";
  div.id = "listeningIndicator";
  div.textContent = "🎤 Listening...";
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
function hideListening(transcript) {
  const indicator = document.getElementById("listeningIndicator");
  if (!indicator) return;
  if (transcript) {
    indicator.className = "message user";
    indicator.innerHTML = DOMPurify.sanitize(marked.parse(transcript));
  } else {
    indicator.remove();
  }
}

// --- Image Loader ---
function showImageLoader() {
  if (document.getElementById("imageLoader")) return;
  const div = document.createElement("div");
  div.className = "image-loader";
  div.id = "imageLoader";
  div.textContent = "🎨 Generating image… please wait";
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
function hideImageLoader() {
  const loader = document.getElementById("imageLoader");
  if (loader) loader.remove();
}

// --- Bot Reply ---
function addBotReply(data) {
  hideTyping();
  hideImageLoader();
  setInputState(false);

  if (data.reply || data.image_url) {
    addMessage(data.reply, "bot", data.image_url);
  }

  if (data.audio && audioEnabled) {
    const filenameWithQuery = data.audio.split("/").pop();
    const filename = filenameWithQuery.split("?")[0];

    let attempts = 0;
    const checkAudio = async () => {
      if (attempts++ > 30) return; // stop after ~30s
      try {
        const res = await fetch(`/audio-status/${filename}`);
        const status = await res.json();
        if (status.ready) {
          replyAudio.src = data.audio;
          replyAudio.style.display = "none"; // keep hidden
          replyAudio.play().catch(err => {
            console.warn("Autoplay blocked:", err);
            replyAudio.controls = true;
            replyAudio.style.display = "block"; // show only if autoplay fails
          });
        } else {
          setTimeout(checkAudio, 1000);
        }
      } catch {
        setTimeout(checkAudio, 1000);
      }
    };
    checkAudio();
  }
}

// --- Text Chat ---
async function sendMessage(question) {
  if (!question.trim()) return;

  addMessage(question, "user");
  textInput.value = "";
  setInputState(true);

  const isImageRequest = ["generate","create","draw","illustrate","design","picture","image","diagram"]
    .some(k => question.toLowerCase().includes(k));

  if (isImageRequest) {
    showImageLoader();
  } else {
    showTyping();
  }

  try {
    const formData = new FormData();
    formData.append("question", question);
    formData.append("need_audio", audioEnabled);

    const res = await fetch("/chat", { method: "POST", body: formData });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    addBotReply(data);
  } catch (err) {
    hideTyping();
    hideImageLoader();
    setInputState(false);
    addMessage(`❌ Error: ${err.message}`, "bot");
  }
}

// --- Event Listeners ---
sendBtn.addEventListener("click", () => sendMessage(textInput.value));
textInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage(textInput.value);
});

// --- Microphone Recording ---
micBtn.addEventListener("click", async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);
      chunks = [];

      showListening();

      recorder.ondataavailable = e => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });

        const formData = new FormData();
        formData.append("file", blob, "input.webm");
        formData.append("need_audio", audioEnabled);

        try {
          showTyping();
          const res = await fetch("/chat-audio", { method: "POST", body: formData });
          const data = await res.json();
          hideListening(data.transcript);
          addBotReply(data);
        } catch (err) {
          hideTyping();
          hideListening(null);
          addMessage(`❌ Error: ${err.message}`, "bot");
        } finally {
          isRecording = false;
          micBtn.classList.remove("recording");
          micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
          setInputState(false);
        }
      };

      recorder.start();
      isRecording = true;
      micBtn.classList.add("recording");
      micBtn.innerHTML = '<i class="fa-solid fa-stop"></i>'; // show stop icon while recording
    } catch (err) {
      hideListening(null);
      setInputState(false);
      alert("Microphone access denied or not available.");
    }
  } else {
    recorder.stop();
  }
});
