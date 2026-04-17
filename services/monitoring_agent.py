from dataclasses import dataclass
from collections import Counter
from typing import List, Dict

from services import config


@dataclass(frozen=True)
class AgentProfile:
    name: str
    role: str
    goal: str


AGENT_PROFILE = AgentProfile(
    name="Agente AgroVision",
    role="triagem operacional de eventos",
    goal="Analisar detecções recentes, explicar riscos e sugerir a próxima ação.",
)

MAX_HISTORY_MESSAGES = 8


def build_event_context(events: List[Dict]) -> str:
    if not events:
        return "Contexto operacional: nenhum evento recente registrado."

    labels = [event["label"] for event in events]
    counts = Counter(labels)
    most_recent = events[0]
    mean_conf = sum(event["confidence"] for event in events) / len(events)

    distribution = ", ".join(f"{label}: {count}" for label, count in counts.items())
    recent_lines = []
    for event in events[:8]:
        recent_lines.append(
            f"- {event['event_time']} | {event['label']} | {event['confidence']:.2f}"
        )

    return (
        "Contexto operacional para o agente:\n"
        f"- Eventos considerados: {len(events)}\n"
        f"- Evento mais recente: {most_recent['label']} em {most_recent['event_time']}\n"
        f"- Distribuicao recente: {distribution}\n"
        f"- Confianca media: {mean_conf:.2f}\n"
        "Eventos recentes:\n"
        + "\n".join(recent_lines)
    )


def normalize_history(history: List[Dict]) -> List[Dict]:
    filtered = []
    for message in history:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            filtered.append({"role": role, "content": content})
    return filtered[-MAX_HISTORY_MESSAGES:]


def build_agent_messages(question: str, history: List[Dict], events: List[Dict]) -> List[Dict]:
    system_prompt = (
        f"Voce e o {AGENT_PROFILE.name}, um agente de {AGENT_PROFILE.role}. "
        f"Objetivo: {AGENT_PROFILE.goal} "
        "Trate os dados como monitoramento operacional autorizado de ambiente real. "
        "Responda em portugues do Brasil, de forma direta e util. "
        "Use os eventos fornecidos como fonte principal. "
        "Nao invente dados que nao aparecem no contexto. "
        "Nao tente identificar pessoas; fale apenas sobre eventos, riscos e proximas acoes. "
        "Quando fizer sentido, organize a resposta em: leitura, risco e recomendacao."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": build_event_context(events)},
    ]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": question})
    return messages
