const img = document.getElementById("liveFrame");
const cameraStatus = document.getElementById("cameraStatus");
const cameraSource = document.getElementById("cameraSource");
const cameraHealth = document.getElementById("cameraHealth");
const agentContext = document.getElementById("agentContext");
const agentEvents = document.getElementById("agentEvents");
const eventCount = document.getElementById("eventCount");
const eventTable = document.getElementById("eventTable");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatLog = document.getElementById("chatLog");

const history = [];

function updateFrame() {
  img.src = "/frame?t=" + Date.now();
}

function setPill(el, text, state) {
  el.textContent = text;
  el.classList.remove("pill-success", "pill-warning");
  if (state === "ok") {
    el.style.color = "#052e16";
    el.style.background = "#86efac";
  } else if (state === "warn") {
    el.style.color = "#7c2d12";
    el.style.background = "#fdba74";
  } else {
    el.style.color = "#94a3b8";
    el.style.background = "rgba(148,163,184,0.12)";
  }
}

async function refreshCameraStatus() {
  try {
    const response = await fetch("/camera/status");
    const data = await response.json();
    cameraSource.textContent = `Fonte: ${data.source}`;
    cameraHealth.textContent = `Status: ${data.connected ? "online" : "offline"}`;
    setPill(cameraStatus, data.connected ? "camera online" : "camera offline", data.connected ? "ok" : "warn");
  } catch (err) {
    setPill(cameraStatus, "erro no status", "warn");
  }
}

function renderEvents(events) {
  eventCount.textContent = `${events.length} registros`;
  if (!events.length) {
    eventTable.innerHTML = "<tr><td colspan=\"4\">Nenhum evento registrado ainda.</td></tr>";
    return;
  }
  eventTable.innerHTML = events
    .map(
      (event) => `
      <tr>
        <td>${event.event_time}</td>
        <td><span class=\"badge\">${event.label}</span></td>
        <td>${event.confidence.toFixed(2)}</td>
        <td><a href=\"${event.image_path}\" target=\"_blank\">ver imagem</a></td>
      </tr>
    `
    )
    .join("");
}

async function refreshEvents() {
  try {
    const response = await fetch("/events");
    const data = await response.json();
    renderEvents(data);
  } catch (err) {
    // ignore
  }
}

async function refreshAgentStatus() {
  try {
    const response = await fetch("/agent/status");
    const data = await response.json();
    agentContext.textContent = data.context_preview || "Aguardando eventos...";
    agentEvents.textContent = `${data.events_in_context} eventos`;
  } catch (err) {
    agentContext.textContent = "Falha ao carregar o agente.";
  }
}

function pushBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  const meta = document.createElement("div");
  meta.className = "chat-meta";
  meta.textContent = role === "user" ? "voce" : "agente";
  const content = document.createElement("div");
  content.textContent = text;
  bubble.appendChild(meta);
  bubble.appendChild(content);
  chatLog.prepend(bubble);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  chatInput.value = "";
  pushBubble("user", question);
  history.push({ role: "user", content: question });

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    const data = await response.json();
    const answer = data.answer || "Sem resposta do agente.";
    history.push({ role: "assistant", content: answer });
    pushBubble("agent", answer);
  } catch (err) {
    pushBubble("agent", "Nao foi possivel falar com o agente agora.");
  }
});

updateFrame();
refreshCameraStatus();
refreshEvents();
refreshAgentStatus();

setInterval(updateFrame, 700);
setInterval(refreshCameraStatus, 4000);
setInterval(refreshEvents, 5000);
setInterval(refreshAgentStatus, 6000);
