import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

from services import config
from services import event_repository
from services.monitoring_agent import AGENT_PROFILE, build_event_context
from services.ollama_client import chat_with_agent_safe
from services.schemas import ChatRequest, ChatResponse
from services.video_monitor import VideoMonitor


app = FastAPI(title="AgroVision AI")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./templates")

monitor = VideoMonitor()


@app.on_event("startup")
def startup_event():
    event_repository.init_db()
    monitor.start()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    events = event_repository.list_events(20)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "events": events,
            "agent_name": AGENT_PROFILE.name,
            "agent_role": AGENT_PROFILE.role,
            "agent_goal": AGENT_PROFILE.goal,
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "AgroVision AI"}


@app.get("/events")
def get_events():
    return JSONResponse(content=event_repository.list_events(50))


@app.get("/frame")
def get_frame():
    try:
        return Response(content=monitor.get_jpeg(), media_type="image/jpeg")
    except RuntimeError as exc:
        return JSONResponse(content={"message": str(exc)}, status_code=500)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        monitor.gen_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera/status")
def camera_status():
    return JSONResponse(content=monitor.get_status())


@app.get("/agent/status")
def agent_status():
    events = event_repository.list_events(config.AGENT_EVENT_LIMIT)
    return JSONResponse(
        content={
            "name": AGENT_PROFILE.name,
            "role": AGENT_PROFILE.role,
            "goal": AGENT_PROFILE.goal,
            "events_in_context": len(events),
            "context_preview": build_event_context(events),
        }
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    events = event_repository.list_events(config.AGENT_EVENT_LIMIT)
    result = chat_with_agent_safe(
        request.question,
        [message.model_dump() for message in request.history],
        events,
    )
    return ChatResponse(
        answer=result["answer"],
        model=config.OLLAMA_MODEL,
        events_in_context=len(events),
    )
