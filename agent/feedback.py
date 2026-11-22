"""Minimal feedback loop.

1. Collect outcome signals per lead (booked / showed_up / closed / dropped / bad_reply).
2. Every turn already stores (state snapshot, actions) in `state_snapshots`.
   Joining the two gives (context, state, action, outcome) tuples.
3. Positive tuples become few-shot examples for the LLM planner, grouped by
   situation (stage + open risks), so the planner learns which move worked
   for which objection. Negative tuples are written out for error analysis.
4. Quality signals (agent/signals.py) flag bad turns during the chat, without waiting
   for an outcome. A flagged turn joins the error-analysis queue and is never used as
   a few-shot example, even if the lead was booked in the end.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .memory import Store
from .models import Event, EventType

POSITIVE = {"booked", "showed_up", "closed"}
NEGATIVE = {"dropped", "bad_reply"}


def record_outcome(store: Store, conversation_id: str, outcome: str, note: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    store.record_outcome(conversation_id, outcome, note, ts)
    store.append_event(Event(conversation_id=conversation_id, type=EventType.FEEDBACK,
                             payload={"outcome": outcome, "note": note}))


def situation(state: dict) -> str:
    risks = sorted(r["type"] for r in state.get("risks", []) if r["status"] != "resolved")
    return f'{state["lead_stage"]}|{",".join(risks) or "no_risk"}'


def export_tuples(store: Store) -> list[dict]:
    outcomes = store.outcomes()
    rows = []
    for snap in store.snapshots():
        if not snap["actions"]:
            continue
        st = snap["state"]
        rows.append({
            "conversation_id": snap["conversation_id"],
            "at_msg": snap["at_msg"],
            "situation": situation(st),
            "summary": st.get("summary", ""),
            "actions": [{"name": a["name"], "field": a["args"].get("field"), "reason": a["reason"]}
                        for a in snap["actions"]],
            "outcome": outcomes.get(snap["conversation_id"]),
        })
    return rows


def export_flags(store: Store) -> list[dict]:
    """The error-analysis queue: one row per flagged turn, with what the agent did and said there.

    Buyer-side signals (a correction, a repeated question) blame the agent turn BEFORE them."""
    acts = {(e["conversation_id"], e["at_msg"]): e["payload"] for e in store.events_of_type(EventType.AGENT_ACTION)}
    msgs = {(e["conversation_id"], e["at_msg"]): e["payload"] for e in store.events_of_type(EventType.USER_MESSAGE)}
    rows: dict[tuple, dict] = {}
    for e in store.events_of_type(EventType.QUALITY_SIGNAL):
        p = e["payload"]
        bad = p.get("blames") if p["side"] == "buyer" else e["at_msg"]
        if not bad:
            continue
        key = (e["conversation_id"], bad)
        row = rows.setdefault(key, {
            "conversation_id": key[0], "bad_turn": bad,
            "message": (msgs.get(key) or {}).get("text"),
            "actions": [{"name": a["name"], "field": a["args"].get("field")} for a in (acts.get(key) or {}).get("actions", [])],
            "replies": (acts.get(key) or {}).get("replies", []),
            "signals": []})
        row["signals"].append({"type": p["type"], "severity": p["severity"], "seen_at": e["at_msg"], "note": p["note"]})
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(rows.values(), key=lambda r: (min(order[s["severity"]] for s in r["signals"]), r["conversation_id"]))


def mine_fewshot(store: Store, out_path: Path, per_situation: int = 1) -> list[dict]:
    tuples = export_tuples(store)
    flagged = {(f["conversation_id"], f["bad_turn"]) for f in export_flags(store)}
    by_sit: dict[str, list[dict]] = {}
    for t in tuples:
        if (t["conversation_id"], t["at_msg"]) in flagged:
            continue                       # a booked lead can still contain a bad turn: never teach it
        if t["outcome"] in POSITIVE and any(a["name"] != "respond_only" for a in t["actions"]):
            by_sit.setdefault(t["situation"], []).append(t)
    examples = [x for rows in by_sit.values() for x in rows[-per_situation:]]
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    negatives = [t for t in tuples if t["outcome"] in NEGATIVE] + \
                [{**f, "source": "quality_signal"} for f in export_flags(store)]
    out_path.with_name("negative_examples.json").write_text(
        json.dumps(negatives, ensure_ascii=False, indent=2), encoding="utf-8")
    return examples
