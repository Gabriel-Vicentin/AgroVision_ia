import json
import time
import urllib.request
from typing import List, Dict

from services import config
from services.monitoring_agent import build_agent_messages


class OllamaError(RuntimeError):
    pass


def _request(payload: Dict) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        config.OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.OLLAMA_TIMEOUT) as response:
        raw = response.read().decode("utf-8")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError("Resposta invalida do Ollama.") from exc


def chat_with_agent(question: str, history: List[Dict], events: List[Dict]) -> str:
    messages = build_agent_messages(question, history, events)
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    response = _request(payload)
    content = response.get("message", {}).get("content")
    if not content:
        raise OllamaError("Resposta vazia do Ollama.")
    return content


def chat_with_agent_safe(question: str, history: List[Dict], events: List[Dict]) -> Dict:
    started = time.time()
    try:
        answer = chat_with_agent(question, history, events)
        return {"answer": answer, "error": None, "elapsed_ms": int((time.time() - started) * 1000)}
    except Exception as exc:  # noqa: BLE001
        return {"answer": "Ollama indisponivel no momento.", "error": str(exc), "elapsed_ms": int((time.time() - started) * 1000)}
