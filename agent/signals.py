"""Quality signals read from the conversation itself, so a bad turn is flagged in minutes,
not when the lead finally shows up as "dropped".

Buyer side (the buyer's reaction says the PREVIOUS agent turn went wrong):
  BUYER_CORRECTION  "đang nói vision 2020 mà", "ý mình là...", "không phải xe đó"
  BUYER_REPEAT      the buyer asks the same thing again ("chủ có giảm ko" after "hỏi chủ có giảm được ko")
Agent side (this turn's reply is weak on its face):
  AGENT_REPEAT      the same reply sent to the same person again
  EMPTY_ACK         "Dạ mình đã ghi nhận" to a message that asked or said something
  WRONG_BIKE        the reply names a bike that is not the one being discussed

None of these block a reply except WRONG_BIKE in the LLM responder (see responder.LLMResponder).
They are logged as QUALITY_SIGNAL events and feed the error-analysis queue (feedback.export_flags).
"""
from __future__ import annotations

import re
from typing import Optional

from . import normalize as N
from .models import ListingRef, State

EMPTY_ACK_TEXT = "Dạ mình đã ghi nhận ạ."
CORRECTION = ["đang nói", "mình đang nói", "em đang nói", "ý mình là", "ý em là", "ý là", "không phải xe",
              "ko phải xe", "không phải chiếc", "chứ không phải", "nhầm rồi", "nhầm xe", "mình hỏi xe",
              "mình hỏi chiếc", "đã nói", "nói rồi", "hỏi rồi", "là chiếc", "chiếc kia", "xe kia"]
STOP = {"a", "ah", "a?", "ạ", "nhe", "nha", "nhé", "voi", "với", "thi", "thì", "ma", "mà", "the", "thế", "vay", "vậy"}
WINDOW = 3            # look back this many messages from the same person


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", N.fold(text)) if w not in {N.fold(s) for s in STOP}}


def _contained(a: str, b: str, ratio: float) -> bool:
    """Most of a's words appear in b (short follow-ups like "chủ có giảm ko" repeat a longer question)."""
    ta, tb = _tokens(a), _tokens(b)
    return len(ta) >= 3 and len(ta & tb) / len(ta) >= ratio


def repeats(a: str, b: str, ratio: float = 0.85) -> bool:
    """`a` says again what `b` said: the same text, or most of a's word pairs appear in b. Word pairs, not single
    words: "Người bán đồng ý giờ 15h00" is news even when each word was in a longer earlier message."""
    if a.strip() == b.strip():
        return True
    wa, wb = re.findall(r"\w+", N.fold(a)), re.findall(r"\w+", N.fold(b))
    pa, pb = set(zip(wa, wa[1:])), set(zip(wb, wb[1:]))
    return len(pa) >= 3 and len(pa & pb) / len(pa) >= ratio


def _label(x: ListingRef) -> Optional[str]:
    return f"{x.model} {x.year}" if x and x.model and x.year else None


def allowed_bikes(state: State, executed_names: set[str], news: Optional[list[dict]]) -> set[str]:
    """Bikes a reply may name this turn: the selected one, ones in news (a seller's new price),
    and fresh search results."""
    lc = state.listing_context
    ids = {n["listing_id"] for n in news or [] if n.get("listing_id")}
    if lc.selected and lc.selected.listing_id:
        ids.add(lc.selected.listing_id)
    if "search_listings" in executed_names:
        ids |= {c.listing_id for c in lc.candidates if c.listing_id}
    return ids


def wrong_bike(texts: list[str], state: State, allowed_ids: set[str]) -> list[str]:
    """Bikes named in the reply that are neither selected nor allowed this turn (news, fresh search results)."""
    lc = state.listing_context
    sel = lc.selected
    known = [x for x in [*lc.previous, *lc.candidates] if x and x.listing_id]
    sel_label = _label(sel) if sel else None
    out = []
    for x in known:
        lab = _label(x)
        if not lab or x.listing_id in allowed_ids or lab == sel_label:
            continue
        if any(N.fold(lab) in N.fold(t) for t in texts):
            out.append(f"{lab} ({x.listing_id})")
    return out


def detect(state: State, sender: str, text: str, intents: list[str], replies: list, history: list[dict],
           allowed_ids: set[str], blames: Optional[str]) -> list[dict]:
    """history: earlier messages, oldest first, as memory.recent_messages() returns them (current one excluded)."""
    out: list[dict] = []

    def add(kind: str, side: str, severity: str, note: str, at: Optional[str] = None) -> None:
        out.append({"type": kind, "side": side, "severity": severity, "note": note, "blames": at})

    if sender == "buyer":
        if N.has_any(text, CORRECTION):
            add("BUYER_CORRECTION", "buyer", "medium", f"buyer corrects the agent: {text!r}", blames)
        earlier = [h["text"] for h in history if h.get("from") == "buyer"][-WINDOW:]
        same = next((e for e in reversed(earlier) if _contained(text, e, 0.75)), None)
        if same:
            add("BUYER_REPEAT", "buyer", "medium", f"asked again: {text!r} (before: {same!r})", blames)

    for r in replies:
        before = [h["text"] for h in history if h.get("from") == "agent" and h.get("to") == r.to][-WINDOW:]
        if r.text.strip() == EMPTY_ACK_TEXT:
            if sender == "buyer" and (intents or "?" in text or len(_tokens(text)) >= 3):
                add("EMPTY_ACK", "agent", "medium", f"no answer to {text!r}")
            continue
        if any(repeats(r.text, b) for b in before):
            add("AGENT_REPEAT", "agent", "low", f"same reply to the {r.to} again: {r.text[:80]!r}")

    names = wrong_bike([r.text for r in replies if r.to == "buyer"], state, allowed_ids)
    if names:
        sel = state.listing_context.selected
        add("WRONG_BIKE", "agent", "high",
            f"reply names {', '.join(names)} while discussing {_label(sel) if sel else 'no bike'}")
    return out
