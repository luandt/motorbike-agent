"""Live 3-way chat demo: buyer and seller in two browser tabs, the agent in between.

  python run.py web                      # http://localhost:8000
  python run.py web --llm openai         # any provider (see agent/llm.py)

Flow for one message:
  tab (buyer|seller) --ws--> server --> both tabs see the message (contact details
  masked for the other party) --> Agent.handle() on the agent thread --> each agent
  reply goes only to the person it addresses --> both tabs get the agent's view
  (facts, risks, actions, state).

Agent.handle() is synchronous and its SQLite store is tied to one thread, so all
agent work runs on a single worker thread. That also processes messages strictly
one at a time: a buyer and a seller typing at the same moment can't overwrite
each other's state update.
"""
from __future__ import annotations

import asyncio
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import pii
from agent.memory import Store
from agent.models import Message, Sender, State, Tracked
from agent.pipeline import Agent, TurnResult
from agent.reducer import slot_coverage

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
ROLES = ("buyer", "seller")


class Config:
    mode = "auto"
    llm: Optional[str] = None
    db = ROOT / "runs/web.db"


# --------------------------------------------------------------------------- #
# The agent lives on one thread
# --------------------------------------------------------------------------- #
class AgentWorker:
    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent")
        self.agent: Optional[Agent] = None
        self.store: Optional[Store] = None

    def _boot(self) -> Agent:
        if self.agent is None:
            if Config.db.exists():
                Config.db.unlink()
            self.store = Store(Config.db)
            self.agent = Agent(self.store, mode=Config.mode, llm=Config.llm)
        return self.agent

    async def run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)

    # ---- functions below run on the agent thread ----
    def info(self) -> dict:
        a = self._boot()
        return {"mode": a.mode, "llm": a.llm_name}

    def seller_listings(self, seller: Optional[str]) -> list[dict]:
        return self._boot().tools.find_seller_listings(seller) if seller else []

    def all_listings(self) -> list[dict]:
        return self._boot().tools.listings

    def register(self, conv: str, seller: Optional[str]) -> None:
        self._boot().meta[conv] = {"buyer_id": f"b_{conv}", "seller_id": seller}

    def handle(self, conv: str, role: str, text: str) -> tuple[TurnResult, dict]:
        a = self._boot()
        st = self.store.load_state(conv)
        i = (st.msg_count if st else 0) + 1
        msg = Message(conversation_id=conv, msg_id=f"{conv}#{i}", index=i, sender=Sender(role), text=text,
                      timestamp=datetime.now(timezone.utc))
        t = a.handle(msg)
        return t, state_view(t.state)

    def state(self, conv: str) -> Optional[dict]:
        self._boot()
        st = self.store.load_state(conv)
        return state_view(st) if st else None

    def events(self, conv: str, n: int = 30) -> list[dict]:
        self._boot()
        return self.store.events(conv)[-n:]


def _money(v: Any) -> Any:
    return f"{v / 1e6:g}tr" if isinstance(v, int) and v >= 1_000_000 else v


def state_view(s: State) -> dict:
    sel = s.listing_context.selected
    n = s.negotiation
    return {
        "stage": s.lead_stage.value,
        "coverage": round(slot_coverage(s), 2),
        "constraints": [{"field": k, "value": _money(t.value), "source": t.source_msg}
                        for k, t in s.constraints if isinstance(t, Tracked)],
        "listing": None if not sel else {
            "id": sel.listing_id, "model": sel.model, "year": sel.year, "ask": _money(sel.ask_price),
            "floor": _money(sel.seller_floor), "paperwork": sel.paperwork_status, "odo": sel.odo_km},
        "candidates": [f"{c.listing_id} {c.model} {c.year} {_money(c.ask_price)}" for c in s.listing_context.candidates],
        "negotiation": {"gap": n.gap_pct, "buyer_offer": _money(n.buyer_offer), "seller_stance": n.seller_stance},
        "risks": [{"type": r.type.value, "severity": r.severity, "status": r.status, "note": r.note,
                   "listing_id": r.listing_id, "applies": s.applies(r)} for r in s.risks],
        "open_questions": [f"{q.field} → {q.ask}" for q in s.open_questions],
        "channel": s.channel_id,
        "appointment": None if s.appointment.status == "none" else {
            "status": s.appointment.status, "time": s.appointment.proposed_time, "place": s.appointment.proposed_place},
        "summary": s.summary,
    }


def turn_view(t: TurnResult) -> dict:
    ex = t.extraction
    return {
        "msg_id": t.message.msg_id,
        "intents": [i.value for i in ex.intents] if ex else [],
        "facts": [{"field": f.field, "value": _money(f.value), "conf": f.confidence} for f in (ex.facts if ex else [])],
        "risks": [{"type": r.type.value, "severity": r.severity, "note": r.note} for r in (ex.risks if ex else [])],
        "actions": [{"name": a.name, "detail": a.args.get("field") if a.name == "ask_clarification" else
                     {k: v for k, v in a.args.items() if v not in (None, False, [])}, "reason": a.reason}
                    for a in t.actions],
        "blocked": {k: v for k, v in t.blocked.items() if k != "share_contact"},
        "signals": t.signals,
    }


# --------------------------------------------------------------------------- #
# Rooms: who is connected, and the history a late-joining tab needs
# --------------------------------------------------------------------------- #
class Room:
    def __init__(self, conv: str, seller: Optional[str]):
        self.conv, self.seller = conv, seller
        self.sockets: dict[WebSocket, str] = {}
        self.history: list[dict] = []        # each item has "audience": all | buyer | seller

    def view_for(self, item: dict, role: str) -> Optional[dict]:
        if item["audience"] not in ("all", role):
            return None
        out = {k: v for k, v in item.items() if k not in ("audience", "text_masked")}
        # the platform masks contact details before delivering a message to the other party
        if item["type"] == "message" and item["from"] != role:
            out["text"] = item["text_masked"]
        return out

    async def publish(self, item: dict) -> None:
        if item["type"] != "busy":           # transient indicators aren't replayed to late joiners
            self.history.append(item)
        dead = []
        for ws, role in self.sockets.items():
            view = self.view_for(item, role)
            if view is None:
                continue
            try:
                await ws.send_json(view)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.pop(ws, None)

    async def presence(self) -> None:
        online = sorted(set(self.sockets.values()))
        for ws in list(self.sockets):
            try:
                await ws.send_json({"type": "presence", "online": online})
            except Exception:
                self.sockets.pop(ws, None)


worker = AgentWorker()
rooms: dict[str, Room] = {}
app = FastAPI(title="Motorbike marketplace agent - live demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class NewConversation(BaseModel):
    seller: Optional[str] = None


@app.get("/")
def lobby():
    return FileResponse(STATIC / "index.html")


@app.get("/chat")
def chat_page():
    return FileResponse(STATIC / "chat.html")


@app.get("/api/info")
async def info():
    return await worker.run(worker.info)


@app.get("/api/sellers")
async def sellers():
    listings = await worker.run(worker.all_listings)
    by_seller: dict[str, list] = {}
    for l in listings:
        by_seller.setdefault(l["seller_id"], []).append(
            f'{l["model"]} {l["year"]} · {_money(l["price"])} · {l["location"]} · {l["paperwork_status"]}')
    return [{"id": k, "listings": v} for k, v in sorted(by_seller.items())]


@app.post("/api/conversations")
async def create_conversation(body: NewConversation):
    conv = "w" + secrets.token_hex(3)
    seller = body.seller or None
    await worker.run(worker.register, conv, seller)
    rooms[conv] = Room(conv, seller)
    return {"conv": conv, "seller": seller}


@app.get("/api/conversations/{conv}/state")
async def get_state(conv: str):
    if conv not in rooms:
        raise HTTPException(404, "unknown conversation")
    return await worker.run(worker.state, conv)


@app.get("/api/conversations/{conv}/events")
async def get_events(conv: str):
    if conv not in rooms:
        raise HTTPException(404, "unknown conversation")
    return await worker.run(worker.events, conv)


@app.websocket("/ws/{conv}/{role}")
async def chat_socket(ws: WebSocket, conv: str, role: str):
    await ws.accept()
    room = rooms.get(conv)
    if room is None or role not in ROLES:
        await ws.send_json({"type": "error", "text": "Unknown conversation or role. Start one from the lobby."})
        await ws.close()
        return
    room.sockets[ws] = role
    await ws.send_json({"type": "hello", "conv": conv, "role": role, "seller": room.seller,
                        "listings": await worker.run(worker.seller_listings, room.seller),
                        **(await worker.run(worker.info))})
    for item in room.history:
        if (view := room.view_for(item, role)) is not None:
            await ws.send_json(view)
    await room.presence()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            text = str(data.get("text", "")).strip()[:1000]
            if not text:
                continue
            masked, _ = pii.redact(text)
            ts = datetime.now(timezone.utc).isoformat()
            await room.publish({"type": "message", "audience": "all", "from": role, "text": text,
                                "text_masked": masked, "ts": ts})
            await room.publish({"type": "busy", "audience": "all", "on": True})
            try:
                turn, state = await worker.run(worker.handle, conv, role, text)
            except Exception as e:   # never leave the tabs hanging on an agent error
                await room.publish({"type": "busy", "audience": "all", "on": False})
                await room.publish({"type": "error", "audience": "all", "text": f"agent error: {e}"})
                continue
            for r in turn.replies:
                await room.publish({"type": "agent", "audience": r.to, "to": r.to, "text": r.text,
                                    "msg_id": turn.message.msg_id, "ts": ts})
            await room.publish({"type": "turn", "audience": "all", "from": role, **turn_view(turn)})
            await room.publish({"type": "state", "audience": "all", "state": state})
            await room.publish({"type": "busy", "audience": "all", "on": False})
    except WebSocketDisconnect:
        pass
    finally:
        room.sockets.pop(ws, None)
        await room.presence()


def serve(host: str = "127.0.0.1", port: int = 8000, mode: str = "auto", llm: Optional[str] = None) -> None:
    import uvicorn
    Config.mode, Config.llm = mode, llm
    uvicorn.run(app, host=host, port=port, log_level="warning")
