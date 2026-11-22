"""Three memory layers, one SQLite file.

1. Raw log        - `events`: every message, tool call, state diff, escalation, feedback.
                    Append-only. Never loaded whole into a prompt.
2. Structured     - `states` (per lead), `buyer_profiles` (per buyer, across leads),
                    `listing_facts` (per listing, shared by every future buyer).
3. Rolling summary- `State.summary`, rebuilt from state (not from the old summary)
                    on stage change or every SUMMARY_EVERY messages.

Retrieval is keyed (conversation_id, buyer_id, listing_id), not vector search.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .models import Event, EventType, LeadStage, State
from .reducer import slot_coverage

SUMMARY_EVERY = 8
RECENT_K = 20


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT,
            ts TEXT, at_msg TEXT, type TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS states(conversation_id TEXT PRIMARY KEY, version INTEGER, json TEXT);
        CREATE TABLE IF NOT EXISTS state_snapshots(conversation_id TEXT, version INTEGER, at_msg TEXT,
            json TEXT, actions TEXT, PRIMARY KEY(conversation_id, version));
        CREATE TABLE IF NOT EXISTS buyer_profiles(buyer_id TEXT PRIMARY KEY, json TEXT);
        CREATE TABLE IF NOT EXISTS listing_facts(listing_id TEXT, field TEXT, value TEXT, source TEXT,
            PRIMARY KEY(listing_id, field));
        CREATE TABLE IF NOT EXISTS outcomes(conversation_id TEXT, outcome TEXT, note TEXT, ts TEXT);
        """)

    # ---------------- raw log ---------------- #
    def append_event(self, e: Event) -> None:
        self.db.execute("INSERT INTO events(conversation_id, ts, at_msg, type, payload) VALUES(?,?,?,?,?)",
                        (e.conversation_id, e.ts.isoformat(), e.at_msg, e.type.value,
                         json.dumps(e.payload, ensure_ascii=False, default=str)))
        self.db.commit()

    def events(self, conversation_id: str, types: Optional[list[EventType]] = None) -> list[dict]:
        rows = self.db.execute("SELECT ts, at_msg, type, payload FROM events WHERE conversation_id=? ORDER BY id",
                               (conversation_id,)).fetchall()
        out = [{"ts": r[0], "at_msg": r[1], "type": r[2], "payload": json.loads(r[3])} for r in rows]
        if types:
            names = {t.value for t in types}
            out = [e for e in out if e["type"] in names]
        return out

    def recent_messages(self, conversation_id: str, k: int = RECENT_K) -> list[dict]:
        out = []
        for e in self.events(conversation_id, [EventType.USER_MESSAGE, EventType.AGENT_ACTION]):
            p = e["payload"]
            if e["type"] == "USER_MESSAGE":
                out.append({"from": p["sender"], "id": p["msg_id"], "text": p["text"]})
            else:
                for r in p.get("replies", []):
                    out.append({"from": "agent", "to": r["to"], "text": r["text"]})
        return out[-k:]

    # ---------------- structured ---------------- #
    def load_state(self, conversation_id: str) -> Optional[State]:
        row = self.db.execute("SELECT json FROM states WHERE conversation_id=?", (conversation_id,)).fetchone()
        return State.model_validate_json(row[0]) if row else None

    def save_state(self, state: State, expected_version: int, at_msg: str, actions: list) -> None:
        """Optimistic concurrency: refuse to overwrite a state someone else changed."""
        row = self.db.execute("SELECT version FROM states WHERE conversation_id=?", (state.conversation_id,)).fetchone()
        current = row[0] if row else 0
        if current != expected_version:
            raise RuntimeError(f"state version conflict for {state.conversation_id}: "
                               f"expected {expected_version}, found {current}")
        state.version = expected_version + 1
        js = state.model_dump_json()
        self.db.execute("INSERT OR REPLACE INTO states VALUES(?,?,?)", (state.conversation_id, state.version, js))
        self.db.execute("INSERT OR REPLACE INTO state_snapshots VALUES(?,?,?,?,?)",
                        (state.conversation_id, state.version, at_msg, js,
                         json.dumps([a.model_dump() for a in actions], ensure_ascii=False, default=str)))
        self.db.commit()

    def events_of_type(self, etype: EventType) -> list[dict]:
        """One event type across all conversations (e.g. every QUALITY_SIGNAL, for the error-analysis queue)."""
        rows = self.db.execute("SELECT conversation_id, ts, at_msg, payload FROM events WHERE type=? ORDER BY id",
                               (etype.value,)).fetchall()
        return [{"conversation_id": r[0], "ts": r[1], "at_msg": r[2], "payload": json.loads(r[3])} for r in rows]

    def snapshots(self, conversation_id: Optional[str] = None) -> list[dict]:
        q = "SELECT conversation_id, version, at_msg, json, actions FROM state_snapshots"
        rows = self.db.execute(q + (" WHERE conversation_id=?" if conversation_id else "") + " ORDER BY conversation_id, version",
                               (conversation_id,) if conversation_id else ()).fetchall()
        return [{"conversation_id": r[0], "version": r[1], "at_msg": r[2], "state": json.loads(r[3]),
                 "actions": json.loads(r[4])} for r in rows]

    def upsert_buyer_profile(self, buyer_id: str, profile: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO buyer_profiles VALUES(?,?)",
                        (buyer_id, json.dumps(profile, ensure_ascii=False, default=str)))
        self.db.commit()

    def buyer_profile(self, buyer_id: Optional[str]) -> dict:
        if not buyer_id:
            return {}
        row = self.db.execute("SELECT json FROM buyer_profiles WHERE buyer_id=?", (buyer_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def upsert_listing_fact(self, listing_id: str, field: str, value, source: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO listing_facts VALUES(?,?,?,?)",
                        (listing_id, field, json.dumps(value, ensure_ascii=False), source))
        self.db.commit()

    def listing_facts(self, listing_id: Optional[str]) -> dict:
        if not listing_id:
            return {}
        rows = self.db.execute("SELECT field, value, source FROM listing_facts WHERE listing_id=?", (listing_id,)).fetchall()
        return {r[0]: {"value": json.loads(r[1]), "source": r[2]} for r in rows}

    # ---------------- outcomes (feedback) ---------------- #
    def record_outcome(self, conversation_id: str, outcome: str, note: str, ts: str) -> None:
        self.db.execute("INSERT INTO outcomes VALUES(?,?,?,?)", (conversation_id, outcome, note, ts))
        self.db.commit()

    def outcomes(self) -> dict[str, str]:
        rows = self.db.execute("SELECT conversation_id, outcome FROM outcomes ORDER BY rowid").fetchall()
        return {r[0]: r[1] for r in rows}   # latest outcome wins


# --------------------------------------------------------------------------- #
# Memory policy: what gets promoted, what goes into context, how it's compacted
# --------------------------------------------------------------------------- #
BUYER_PROFILE_FIELDS = ["budget_max", "budget_flexibility", "location", "vehicle_type", "brands", "year_min",
                        "odo_max", "odo_pref", "urgency"]


def promote(state: State, store: Store) -> None:
    """Copy durable facts out of the lead into buyer- and listing-level memory."""
    buyer = state.participants.get("buyer_id")
    if buyer:
        profile = store.buyer_profile(buyer)
        for f in BUYER_PROFILE_FIELDS:
            t = getattr(state.constraints, f)
            if t is not None and t.confidence >= 0.6:
                profile[f] = {"value": t.value, "source": f"{state.conversation_id}:{t.source_msg}"}
        if state.constraints.deal_breakers:
            profile["deal_breakers"] = state.constraints.deal_breakers
        store.upsert_buyer_profile(buyer, profile)
    sel = state.listing_context.selected
    if sel and sel.listing_id:
        if sel.paperwork_status:
            store.upsert_listing_fact(sel.listing_id, "paperwork_status", sel.paperwork_status, state.conversation_id)
        if sel.seller_claims:
            store.upsert_listing_fact(sel.listing_id, "seller_claims", sel.seller_claims, state.conversation_id)
        if sel.seller_floor:
            store.upsert_listing_fact(sel.listing_id, "seller_floor", sel.seller_floor, state.conversation_id)
        if state.negotiation.seller_stance:
            store.upsert_listing_fact(sel.listing_id, "seller_stance", state.negotiation.seller_stance, state.conversation_id)


def compact_state(state: State) -> dict:
    """The part of state worth putting in a prompt (no history, no tool payloads)."""
    c = {k: (v.value if hasattr(v, "value") else v)
         for k, v in state.constraints.__dict__.items() if v not in (None, [])}
    sel = state.listing_context.selected
    return {
        "stage": state.lead_stage.value,
        "constraints": c,
        "selected_listing": sel.model_dump(exclude_none=True, exclude={"source_msg"}) if sel else None,
        "candidates": [f"{x.listing_id} {x.model} {x.year} {x.ask_price:,}đ {x.district or ''} seller={x.seller_id}"
                       for x in state.listing_context.candidates],
        "earlier_bikes": [f"{x.listing_id} {x.model} {x.year} {x.district or ''} seller={x.seller_id}"
                          for x in state.listing_context.previous if x.listing_id],
        "seller_in_this_chat": state.participants.get("seller_id"),
        "negotiation": state.negotiation.model_dump(exclude_none=True),
        "risks": [f"{r.type.value}/{r.severity}/{r.status}" for r in state.active_risks()],
        "consent": state.consent.model_dump(exclude_none=True),
        "channel_id": state.channel_id,
        "appointment": state.appointment.model_dump(exclude_none=True),
        "slot_coverage": slot_coverage(state),
    }


def build_context(state: State, store: Store) -> dict:
    sel = state.listing_context.selected
    return {
        "state": compact_state(state),
        "summary": state.summary,
        "recent_messages": store.recent_messages(state.conversation_id),
        "listing_memory": store.listing_facts(sel.listing_id if sel else None),
        "buyer_profile": store.buyer_profile(state.participants.get("buyer_id")),
    }


def summarize(state: State) -> str:
    """Rebuilt from structured state each time, so it can't drift."""
    parts = []
    c = state.constraints
    want = []
    if c.vehicle_type: want.append(c.vehicle_type.value)
    if c.brands: want.append("/".join(c.brands.value))
    if c.year_min: want.append(f"{c.year_min.value}+")
    if c.location: want.append(f"in {c.location.value}")
    if c.budget_max: want.append(f"max {c.budget_max.value/1e6:g}tr ({c.budget_flexibility.value if c.budget_flexibility else '?'})")
    if c.odo_pref: want.append("low odo")
    if want:
        parts.append("Buyer wants " + ", ".join(want) + ".")
    sel = state.listing_context.selected
    if sel and (sel.model or sel.listing_id):
        s = f"Discussing {sel.model or '?'} {sel.year or ''} ({sel.listing_id or 'unidentified'})"
        if sel.ask_price: s += f" at {sel.ask_price/1e6:g}tr"
        if sel.paperwork_status and sel.paperwork_status != "clean": s += f", paperwork {sel.paperwork_status}"
        parts.append(s + ".")
    if state.negotiation.gap_pct and state.negotiation.gap_pct > 0:
        parts.append(f"Price gap {state.negotiation.gap_pct:.0%}, seller {state.negotiation.seller_stance or 'unknown'}.")
    open_r = [f"{r.type.value}({r.status})" for r in state.active_risks() if r.status != "resolved"]
    if open_r:
        parts.append("Risks: " + ", ".join(open_r) + ".")
    if state.channel_id:
        parts.append(f"Channel {state.channel_id} open.")
    if state.appointment.status == "booked":
        parts.append(f"Viewing booked {state.appointment.proposed_time} at {state.appointment.proposed_place}.")
    parts.append(f"Stage {state.lead_stage.value}.")
    return " ".join(parts)


def maybe_refresh_summary(state: State, material_change: bool) -> bool:
    if material_change or state.msg_count - state.last_summary_at >= SUMMARY_EVERY or not state.summary:
        state.summary = summarize(state)
        state.last_summary_at = state.msg_count
        return True
    return False
