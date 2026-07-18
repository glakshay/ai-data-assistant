import asyncio
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import run_turn
from .guardrails import classify_intent
from .schema_cache import discover_schema
from .session import append_turn, clear_session, get_history, get_meta
from .snowflake_client import SnowflakeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Rate limiter (casual — 15 requests / minute / session) ───────────────────
# Note: in-process only; resets on restart. For prod, use Redis + sliding window.
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 15  # requests per minute per session


def _is_rate_limited(session_id: str) -> bool:
    now = time.time()
    window = [t for t in _rate_store[session_id] if now - t < 60]
    _rate_store[session_id] = window
    if len(window) >= _RATE_LIMIT:
        return True
    _rate_store[session_id].append(now)
    return False


_sf: SnowflakeClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sf
    from . import llm
    logger.info("LLM provider: %s (main=%s, fast=%s)",
                llm.active_provider(), llm.model_for("main"), llm.model_for("fast"))
    _sf = SnowflakeClient()
    logger.info("SnowflakeClient ready — discovering schema...")
    discover_schema(_sf)
    yield
    if _sf:
        _sf.close()


app = FastAPI(title="US Census Chat Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())

    async def event_stream():
        full_response: list[str] = []
        try:
            # Rate limit check
            if _is_rate_limited(session_id):
                yield "data: You're sending messages too quickly — please wait a moment and try again.\n\n"
                yield "data: [DONE]\n\n"
                return

            history = get_history(session_id)
            meta = get_meta(session_id)

            # Guardrail / intent classifier (sync Gemini call — run in thread).
            # History is passed so follow-up corrections and closings stay in context.
            intent, guard_msg = await asyncio.get_event_loop().run_in_executor(
                None, lambda: classify_intent(body.message, history)
            )
            if intent == "inappropriate":
                # Terminal: show the refusal, tell the UI to end the chat, wipe the session.
                logger.warning("Inappropriate input — ending session %s", session_id[:8])
                yield f"data: {guard_msg}\n\n"
                yield "data: [ENDCHAT]\n\n"
                yield "data: [DONE]\n\n"
                clear_session(session_id)
                return
            if intent == "blocked":  # injection / too-short — show directly, chat continues
                yield f"data: {guard_msg}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Agent — run sync generator in thread, stream chunks as SSE.
            # run_turn mutates `meta` in place (escalation + wind-down counters).
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(run_turn(history, body.message, _sf, intent, meta))
            )

            for chunk in chunks:
                full_response.append(chunk)
                if chunk.startswith("\n[FOLLOWUPS]"):
                    # Send followups as a special SSE event (no newline escaping)
                    yield f"data: {chunk.strip()}\n\n"
                else:
                    escaped = chunk.replace("\n", "\\n")
                    yield f"data: {escaped}\n\n"

            yield "data: [DONE]\n\n"

            # Save turn (strip followups from stored history)
            answer = "".join(
                c for c in full_response if not c.startswith("\n[FOLLOWUPS]")
            )
            if answer:
                append_turn(session_id, body.message, answer)

        except Exception as e:
            logger.exception("Unhandled error in /chat")
            yield f"data: [ERROR] {e}\n\n"
            yield "data: [DONE]\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.set_cookie("session_id", session_id, httponly=True, samesite="lax", max_age=86400)
    return response


@app.post("/reset")
async def reset(request: Request):
    sid = request.cookies.get("session_id")
    if sid:
        clear_session(sid)
    return {"status": "cleared"}
