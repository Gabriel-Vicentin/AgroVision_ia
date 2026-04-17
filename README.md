# AgroVision AI

Monitoramento de tráfego com visão computacional (YOLO), persistência em SQLite e agente local via Ollama.

## Visão geral
Fluxo principal do sistema:

1. Uma câmera pública/autorizada fornece o vídeo.
2. O backend FastAPI abre o stream com OpenCV.
3. O YOLO detecta objetos relevantes.
4. O sistema salva eventos no SQLite.
5. O dashboard exibe vídeo, eventos e capturas.
6. O agente usa Ollama para interpretar os eventos.

## Stack
- Python 3.11+
- FastAPI + Uvicorn
- OpenCV
- Ultralytics YOLO
- SQLite
- Ollama (LLM local)

## Estrutura de pastas
```
AgroVision_ia/
	app.py
	main.py
	services/
		config.py
		event_repository.py
		video_monitor.py
		ollama_client.py
		monitoring_agent.py
		schemas.py
		capture_store.py
	templates/
		index.html
	static/
		dashboard.css
		dashboard.js
		captures/
	detections.db
	yolov8n.pt
```

## Pré-requisitos
- Python 3.11+ instalado
- Ollama instalado
- Modelo LLM baixado (ex: `llama3`)

## Instalação
Crie e ative o ambiente virtual:

```
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```
pip install -r requirements.txt
```

Instale o Ollama (Linux/macOS):

```
curl -fsSL https://ollama.com/install.sh | sh
```

Baixe o modelo:

```
ollama pull llama3
```

## Configuração (.env)
Crie um arquivo `.env` na raiz do projeto com:

```
OLLAMA_URL=http://127.0.0.1:11434/api/chat
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=120
OLLAMA_KEEP_ALIVE=30m
AGENT_EVENT_LIMIT=12

CAMERA_SOURCE=https://wzmedia.dot.ca.gov/D11/C214_SB_5_at_Via_De_San_Ysidro.stream/playlist.m3u8
CAMERA_RECONNECT_SECONDS=5
```

Outras opções úteis:

```
CONFIDENCE_THRESHOLD=0.45
TARGET_CLASSES=person,car,truck,bus,motorcycle
```

## Executando o projeto
Abra dois terminais.

Terminal 1 (Ollama):
```
ollama serve
```

Terminal 2 (API):
```
python -m uvicorn app:app --reload
```

Abra o dashboard:
```
http://127.0.0.1:8000
```

## Rotas principais
- `/health` status do serviço
- `/camera/status` status da câmera
- `/frame` frame JPEG atual
- `/video_feed` stream MJPEG
- `/events` eventos recentes
- `/agent/status` status do agente e contexto
- `/chat` conversa com o agente (POST)

## Teste rápido do agente
No painel, pergunte:
- "Leia os eventos recentes, avalie o risco e recomende a próxima ação."

## Problemas comuns
**Ollama não responde**
- Verifique se o servidor está ativo: `ollama serve`
- Teste: `curl http://127.0.0.1:11434/api/tags`

**Câmera não conecta**
- Verifique o `CAMERA_SOURCE` no `.env`
- Teste o status: `/camera/status`

**Agente responde genérico**
- Verifique se há eventos: `/events`
- Veja o contexto: `/agent/status`

## Observações éticas
Use apenas câmeras públicas oficiais ou privadas com autorização. Não utilize o sistema para identificar pessoas.