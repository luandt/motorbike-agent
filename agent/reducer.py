"""Deterministic state reducer.

Input: current state + proposed facts (from extractor or tool results).
Output: new state + a diff (logged as STATE_UPDATE).

All business rules about *what the state means* live here, in plain code:
superseding old values, price-gap maths, risk lifecycle, stage transitions,
open questions. That makes them unit-testable and identical across LLM/rule modes.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from .normalize import ASKED_YET_RE, exact_place, find_location
from .pii import redact
from .models import (Extraction, Fact, Intent, LeadStage, ListingRef, Message, OpenQuestion,
                     Risk, RiskSignal, RiskType, State, Tracked)

PRUNE_CONFIDENCE = 0.6
PRUNE_AFTER_MSGS = 4
BIG_GAP = 0.15
HIGH_ODO_PER_YEAR = 15_000
SLOT_WEIGHTS = {"budget_max": 3, "location": 2, "vehicle_type": 1, "brands": 1,
                "year_min": 1, "odo": 1, "urgency": 1}
PAPERWORK_SPECIFIC = {"pending_original_docs"}
MAX_GAP_TO_MEET = 0.10        # don't arrange a viewing when the price is >10% apart (policy gates the bridge on it)
IMPLAUSIBLE_DROP = 0.5          # reject an ask price or buyer offer below 50% of the known ask
IMPLIED_CONFIDENCE = 0.6        # facts inferred from a listing, not said by anyone
# seller questions whose answer is passed to the buyer word for word (viewing answers have their own news)
FORWARD_FIELDS = ("relay_question", "condition_history", "seller_floor_price", "relay_offer")
BIKE_RISKS = {RiskType.PAPERWORK_PENDING, RiskType.STOLEN_RISK, RiskType.HIGH_ODO}   # about one bike, not the deal


def new_state(conversation_id: str, participants: dict) -> State:
    return State(conversation_id=conversation_id, participants=participants)


# --------------------------------------------------------------------------- #
# Applying facts
# --------------------------------------------------------------------------- #
def _set_constraint(state: State, name: str, fact: Fact, index: int, diffs: list) -> None:
    if name == "deal_breakers":
        if fact.value not in state.constraints.deal_breakers:
            state.constraints.deal_breakers.append(fact.value)
            diffs.append({"field": "constraints.deal_breakers", "old": None, "new": fact.value, "source": fact.source_msg})
        return
    if name == "location" and isinstance(fact.value, str):
        # the LLM writes what the buyer typed ("SG"); listings say "HCM"
        fact = fact.model_copy(update={"value": find_location(fact.value) or fact.value})
    cur: Optional[Tracked] = getattr(state.constraints, name)
    if cur is not None and cur.value == fact.value:
        cur.confidence = max(cur.confidence, fact.confidence)   # re-confirmed
        cur.set_at_index = index
        if str(cur.source_msg or "").startswith("implied:") and fact.speaker in ("buyer", "seller"):
            # inferred from the listing, now said by a person: it is theirs ("sg" after the Gò Vấp bike was found)
            cur.source_msg, cur.speaker = fact.source_msg, fact.speaker
            diffs.append({"field": f"constraints.{name}", "old": cur.value, "new": cur.value, "source": fact.source_msg})
        return
    history = []
    if cur is not None:
        history = cur.history + [{"value": cur.value, "source_msg": cur.source_msg}]
    setattr(state.constraints, name, Tracked(value=fact.value, source_msg=fact.source_msg, speaker=fact.speaker,
                                             confidence=fact.confidence, set_at_index=index, history=history))
    diffs.append({"field": f"constraints.{name}", "old": cur.value if cur else None,
                  "new": fact.value, "source": fact.source_msg})


def room_seller(state: State) -> Optional[str]:
    """The seller taking part in this chat (None when the buyer is searching on their own)."""
    return state.participants.get("seller_id")


def in_room(state: State, ref: Optional[ListingRef]) -> bool:
    """True when the bike belongs to the seller in this chat, so the seller here can answer about it."""
    room = room_seller(state)
    return ref is None or not ref.seller_id or not room or ref.seller_id == room


def room_listing(state: State) -> Optional[ListingRef]:
    """The bike the seller in this chat is selling: the selected one, or the one the buyer moved away from."""
    lc = state.listing_context
    if in_room(state, lc.selected):
        return lc.selected
    return next((p for p in reversed(lc.previous) if in_room(state, p)), None)


def _ensure_selected(state: State, fact: Fact) -> ListingRef:
    lc = state.listing_context
    if lc.selected is None:
        lc.selected = ListingRef(seller_id=state.participants.get("seller_id"), source_msg=fact.source_msg)
    return lc.selected


def _select_candidate(state: State, model: str, fact: Fact, diffs: list) -> None:
    lc = state.listing_context
    match = next((c for c in lc.candidates if c.model == model), None)
    if match is None:
        lc.mentioned_model = model
        return
    _select(state, match, fact, diffs)


def _select_listing(state: State, listing_id: str, fact: Fact, diffs: list) -> None:
    """Switch to a bike by id: one of the shown candidates, or a bike discussed earlier."""
    lc = state.listing_context
    if lc.selected and lc.selected.listing_id == listing_id:
        return
    match = next((c for c in lc.candidates + lc.previous if c.listing_id == listing_id), None)
    if match is not None:
        _select(state, match, fact, diffs)


def _select(state: State, match: ListingRef, fact: Fact, diffs: list) -> None:
    lc = state.listing_context
    old = lc.selected
    lc.previous = [p for p in lc.previous if p.listing_id != match.listing_id]
    if old is not None:
        lc.previous.append(old)
    lc.selected = match.model_copy(deep=True)
    lc.selected.source_msg = fact.source_msg
    lc.candidates = [c for c in lc.candidates if c.listing_id != match.listing_id]
    # a new bike means a new negotiation, channel and appointment
    state.negotiation.seller_stance = None
    state.negotiation.seller_firm_turns = 0
    state.anchors = {k: v for k, v in state.anchors.items() if not k.startswith("negotiation.")}
    state.channel_id = None
    state.appointment.status = "none"
    diffs.append({"field": "listing_context.selected", "old": old.listing_id if old else None,
                  "new": match.listing_id, "source": fact.source_msg})


def _set_listing(state: State, name: str, fact: Fact, diffs: list) -> None:
    lc = state.listing_context
    if name == "mentioned_model":
        lc.mentioned_model = fact.value
        diffs.append({"field": "listing_context.mentioned_model", "old": None, "new": fact.value, "source": fact.source_msg})
        return
    if name == "select_model":
        _select_candidate(state, fact.value, fact, diffs)
        return
    if name == "select_listing":
        _select_listing(state, str(fact.value), fact, diffs)
        return
    sel = None
    if fact.speaker == "seller" and not in_room(state, lc.selected):
        # the buyer moved to another seller's bike; what THIS seller says is still about their own bike
        sel = room_listing(state)
    sel = sel or _ensure_selected(state, fact)
    if name == "model" and sel.model and sel.model != fact.value and sel is lc.selected:
        lc.previous.append(sel)          # seller now talks about a different bike
        sel = lc.selected = ListingRef(seller_id=state.participants.get("seller_id"), source_msg=fact.source_msg)
    if name == "seller_claims":
        if fact.value not in sel.seller_claims:
            sel.seller_claims.append(fact.value)
        return
    if name == "paperwork_status" and sel.paperwork_status in PAPERWORK_SPECIFIC:
        return                            # don't overwrite a specific status with a vaguer one
    if name == "ask_price" and sel.ask_price and fact.value < sel.ask_price * IMPLAUSIBLE_DROP:
        diffs.append({"field": "listing.ask_price", "old": sel.ask_price, "new": sel.ask_price,
                      "source": f"rejected:implausible {fact.value:,} from {fact.source_msg}"})
        return                            # a 90% price cut in one message is almost always a misread
    old = getattr(sel, name)
    if old != fact.value:
        setattr(sel, name, fact.value)
        diffs.append({"field": f"listing.{name}", "old": old, "new": fact.value, "source": fact.source_msg,
                      "listing_id": sel.listing_id})


def _add_risk(state: State, sig: RiskSignal, diffs: list) -> None:
    existing = state.risk(sig.type)
    if existing is None:
        state.risks.append(Risk(type=sig.type, severity=sig.severity, evidence=[sig.evidence], note=sig.note,
                                listing_id=sig.listing_id))
        diffs.append({"field": "risks", "old": None, "new": f"{sig.type.value}({sig.severity})", "source": sig.evidence})
        return
    if sig.evidence not in existing.evidence:
        existing.evidence.append(sig.evidence)
    rank = {"low": 0, "medium": 1, "high": 2}
    if rank[sig.severity] > rank[existing.severity]:
        diffs.append({"field": "risks", "old": f"{sig.type.value}({existing.severity})",
                      "new": f"{sig.type.value}({sig.severity})", "source": sig.evidence})
        existing.severity, existing.note = sig.severity, sig.note
        existing.status = "open"          # e.g. buyer offered a deposit, now the seller asks for one: escalate
        return
    if existing.status == "resolved":
        existing.status = "open"
        diffs.append({"field": "risks", "old": f"{sig.type.value}:resolved", "new": "open", "source": sig.evidence})


def concrete_time(value: Any) -> bool:
    """An ISO datetime with an hour. "tomorrow afternoon" (LLM-mode live chat) can't be booked."""
    try:
        return "T" in str(value) and datetime.fromisoformat(str(value)) is not None
    except ValueError:
        return False


def _set_appointment(state: State, name: str, fact: Fact, diffs: list) -> None:
    """Time and place are each proposed by one side and agreed by the other. Live web chat: the seller's
    "hẹn khách ở quận 1 được ko" was booked without the buyer, and the buyer's later "3h" was "confirmed"
    without the seller."""
    ap, who = state.appointment, fact.speaker
    if name == "agree":
        # the side that agreed answered its question: what is still missing from it can be asked now. The other
        # side's question stays asked (Phase 5: the seller's "ok 3h" re-sent the buyer the Q1 question unanswered)
        q = "appointment_confirm" if who == "seller" else "appointment_ok"
        state.asked_fields.pop(q, None)
        state.asked_counts.pop(q, None)
        for slot in ("time", "place"):
            if getattr(ap, f"proposed_{slot}") and getattr(ap, f"{slot}_by") not in (None, who) \
                    and not getattr(ap, f"{slot}_agreed"):
                setattr(ap, f"{slot}_agreed", True)
                diffs.append({"field": f"appointment.{slot}_agreed", "old": False, "new": True, "source": fact.source_msg})
        return
    if name == "proposed_time" and not concrete_time(fact.value):
        diffs.append({"field": fact.field, "old": ap.proposed_time, "new": f"rejected: {fact.value!r} has no hour",
                      "source": "rejected:" + fact.source_msg})
        return
    slot = "time" if name == "proposed_time" else "place"
    old = getattr(ap, name)
    if old == fact.value:
        if getattr(ap, f"{slot}_by") not in (None, who) and not getattr(ap, f"{slot}_agreed"):
            setattr(ap, f"{slot}_agreed", True)       # the other side repeats it: agreed
            diffs.append({"field": f"appointment.{slot}_agreed", "old": False, "new": True, "source": fact.source_msg})
        return
    # the seller gives the street address inside an area the buyer already agreed to ("Q1" -> "12 Lê Lợi"): still agreed
    keep = slot == "place" and who == "seller" and ap.place_agreed and not exact_place(old) and exact_place(fact.value)
    if keep and old and old.lower() not in str(fact.value).lower():
        fact = fact.model_copy(update={"value": f"{fact.value}, {old}"})     # "12 Lê Lợi, Quận 1"
    setattr(ap, name, fact.value)
    setattr(ap, f"{slot}_by", who)
    setattr(ap, f"{slot}_agreed", keep)
    ap.source_msg = fact.source_msg
    q = "appointment_ok" if who == "seller" else "appointment_confirm"   # the other side is asked afresh
    state.asked_fields.pop(q, None)
    state.asked_counts.pop(q, None)
    if ap.status == "booked":                                 # a changed booking must be agreed again
        diffs.append({"field": "appointment.status", "old": "booked", "new": "proposed", "source": fact.source_msg})
        ap.status, ap.booking_id = "proposed", None
    elif ap.status == "none":
        ap.status = "proposed"
    diffs.append({"field": fact.field, "old": old, "new": fact.value, "source": fact.source_msg})


def apply_fact(state: State, fact: Fact, index: int, diffs: list) -> None:
    group, _, name = fact.field.partition(".")
    if group == "constraints":
        _set_constraint(state, name, fact, index, diffs)
    elif group == "listing":
        _set_listing(state, name, fact, diffs)
    elif group == "negotiation":
        sel = state.listing_context.selected
        if name == "buyer_offer" and sel and sel.ask_price and fact.value < sel.ask_price * IMPLAUSIBLE_DROP:
            diffs.append({"field": fact.field, "old": state.negotiation.buyer_offer, "new": state.negotiation.buyer_offer,
                          "source": f"rejected:implausible {fact.value:,} vs ask {sel.ask_price:,} from {fact.source_msg}"})
            return                        # "5tr" on a 32tr bike is a deposit or a misread, not an offer to relay
        old = getattr(state.negotiation, name)
        if old != fact.value:
            setattr(state.negotiation, name, fact.value)
            diffs.append({"field": fact.field, "old": old, "new": fact.value, "source": fact.source_msg})
    elif group == "appointment":
        _set_appointment(state, name, fact, diffs)
    elif group == "consent":
        old = getattr(state.consent, name)
        if old != fact.value:
            setattr(state.consent, name, fact.value)
            diffs.append({"field": fact.field, "old": old, "new": fact.value, "source": fact.source_msg})


# Facts whose source message is pinned into the prompt while the fact is live (memory.pinned_refs).
# Constraints, the selected bike and risks carry their source already (Tracked.source_msg, ListingRef.source_msg,
# Risk.evidence); these do not.
ANCHOR_FIELDS = {"listing.ask_price", "listing.seller_floor", "listing.paperwork_status",
                 "negotiation.seller_stance", "negotiation.buyer_offer",
                 "consent.buyer_bridge", "consent.buyer_contact_share"}


def _record_anchors(state: State, diffs: list[dict]) -> None:
    for d in diffs:
        src = str(d.get("source") or "")
        if d["field"] in ANCHOR_FIELDS and "#" in src and not src.startswith(("rejected", "tool", "implied")):
            key = d["field"] + (f"@{d['listing_id']}" if d.get("listing_id") else "")
            state.anchors[key] = src


def apply_extraction(state: State, ex: Extraction, msg: Message) -> tuple[State, list[dict]]:
    state = state.model_copy(deep=True)
    diffs: list[dict] = []
    state.msg_count += 1
    state.turn_intents = [i.value for i in ex.intents]
    for it in ex.intents:
        if it.value not in state.seen_intents:
            state.seen_intents.append(it.value)
    if Intent.FIRM_PRICE in ex.intents:
        state.negotiation.seller_firm_turns += 1
    # select_model first, so that later listing facts in the same message attach to the new bike
    for fact in sorted(ex.facts, key=lambda f: f.field != "listing.select_model"):
        apply_fact(state, fact, msg.index, diffs)
    # a viewing question waits on one person: once they answer about the viewing (a time, a place, a yes), the next
    # one to them (usually asking for something new) may go out; until then it is not sent again (Phase 5 live
    # runs: the same question every turn). "mấy h?" is not an answer: the normal cooldown applies.
    mine = {"seller": "appointment_confirm", "buyer": "appointment_ok"}.get(msg.sender.value)
    if mine and any(f.field.startswith("appointment.") for f in ex.facts):
        state.asked_fields.pop(mine, None)
    if msg.sender.value == "buyer" and Intent.ASK_SELLER in ex.intents and not ASKED_YET_RE.search(msg.text):
        # "hỏi chủ xe đi" names no question: it is the buyer's previous message (live chat: "Người mua hỏi: “đi”")
        q = relay_text(redact(msg.text)[0]) or (state.prev_buyer or {}).get("text")
        if q:
            set_relay(state, q, msg.msg_id)
    side = msg.sender.value
    if side == "seller":
        # the seller writing after we sent them a question is their answer; the buyer must hear it (live chat:
        # "chắc 2 tuần là có", "được" never reached the buyer, who was then told "chưa có cập nhật")
        pending = [f for f in FORWARD_FIELDS if state.asked_fields.get(f, -1) >= state.last_from.get("seller", -1)
                   and state.asked_fields.get(f, -1) < state.msg_count]
        if pending:
            state.seller_reply = {"text": redact(msg.text)[0], "msg": msg.msg_id, "index": state.msg_count,
                                  "about": state.relay["text"] if "relay_question" in pending and state.relay else None}
    if side in ("buyer", "seller"):
        state.last_from[side] = state.msg_count
    if side == "buyer":
        state.prev_buyer = {"text": redact(msg.text)[0], "msg": msg.msg_id}
    sel = state.listing_context.selected
    if msg.sender.value == "buyer" and sel and sel.listing_id and not state.consent.buyer_bridge \
            and {Intent.REQUEST_VIEWING, Intent.SCHEDULE} & set(ex.intents):
        # a day or time for a bike already picked ("mai hoặc mốt thì đc", "chiều mai 4h") is a yes to viewing it.
        # Live web chat: the LLM read REQUEST_VIEWING, but only "xem xe" / "kết nối" set consent, so the bridge
        # stayed blocked and the seller was never asked about the time.
        apply_fact(state, Fact(field="consent.buyer_bridge", value=True, source_msg=msg.msg_id, speaker="buyer",
                               confidence=0.85), msg.index, diffs)
    _record_anchors(state, diffs)
    for sig in ex.risks:
        if sig.type in BIKE_RISKS and sig.listing_id is None:
            # "giấy tờ đang chờ" is about the speaker's bike: the seller's own, or the one the buyer is on
            ref = room_listing(state) if msg.sender.value == "seller" else state.listing_context.selected
            sig = sig.model_copy(update={"listing_id": ref.listing_id if ref else None})
        _add_risk(state, sig, diffs)
    if Intent.DROP in ex.intents:
        state.handed_off = True
    diffs += prune(state)
    diffs += recompute(state, msg)
    return state, diffs


# --------------------------------------------------------------------------- #
# Tool results -> state
# --------------------------------------------------------------------------- #
def _listing_from_tool(d: dict) -> ListingRef:
    return ListingRef(listing_id=d["listing_id"], seller_id=d.get("seller_id"), brand=d.get("brand"),
                      model=d.get("model"), year=d.get("year"), odo_km=d.get("odo_km"),
                      ask_price=d.get("price"), location=d.get("location"), district=d.get("district"),
                      paperwork_status=d.get("paperwork_status"), source_msg="tool")


def apply_tool_result(state: State, tool: str, args: dict, result: Any, msg: Message) -> list[dict]:
    diffs: list[dict] = []
    src = f"tool:{tool}"
    lc = state.listing_context
    if tool == "search_listings" and isinstance(result, list):
        # the buyer saw results for these requirements and went on: they are not guesses to prune any more
        for name in ("budget_max", "location", "year_min", "brands", "vehicle_type"):
            t = getattr(state.constraints, name, None)
            if isinstance(t, Tracked) and t.confidence < PRUNE_CONFIDENCE:
                t.confidence = PRUNE_CONFIDENCE
        sel_id = lc.selected.listing_id if lc.selected else None
        before = [c.listing_id for c in lc.candidates]
        lc.candidates = [_listing_from_tool(r) for r in result if r["listing_id"] != sel_id]
        now = [c.listing_id for c in lc.candidates]
        lc.candidates_unchanged = bool(before) and before == now
        lc.candidates_narrowed = bool(now) and not lc.candidates_unchanged and set(now) < set(before)
        if lc.candidates:
            lc.candidates_at = state.msg_count
            state.anchors["candidates_shown"] = msg.msg_id
        diffs.append({"field": "listing_context.candidates", "old": None,
                      "new": [c.listing_id for c in lc.candidates], "source": src})
    elif tool == "find_seller_listings" and isinstance(result, list):
        lc.seller_listing_options = [f'{r["model"]} {r["year"]} ({r["listing_id"]})' for r in result]
    elif tool == "get_listing_detail" and isinstance(result, dict) and "listing_id" in result:
        sel = lc.selected if lc.selected and lc.selected.listing_id == result["listing_id"] else None
        if sel is None:
            sel = lc.selected = _listing_from_tool(result)
        for name, key in [("brand", "brand"), ("model", "model"), ("year", "year"), ("odo_km", "odo_km"),
                          ("ask_price", "price"), ("location", "location"), ("district", "district"),
                          ("seller_id", "seller_id")]:
            if getattr(sel, name) is None and result.get(key) is not None:
                setattr(sel, name, result[key])
                diffs.append({"field": f"listing.{name}", "old": None, "new": result[key], "source": src})
        status = result.get("paperwork_status")
        if status and sel.paperwork_status not in PAPERWORK_SPECIFIC:
            sel.paperwork_status = status
            diffs.append({"field": "listing.paperwork_status", "old": None, "new": status, "source": src})
        if status and status != "clean":
            _add_risk(state, RiskSignal(type=RiskType.PAPERWORK_PENDING, severity="high",
                                        evidence=f"{src}:{result['listing_id']}", listing_id=result["listing_id"],
                                        note=f"listing record says paperwork_status={status}"), diffs)
        if result["listing_id"] not in lc.detail_fetched:
            lc.detail_fetched.append(result["listing_id"])
        if state.constraints.location is None and result.get("location"):
            # the buyer is asking about a bike in this city, so don't ask "where are you?"
            state.constraints.location = Tracked(value=result["location"], source_msg=f"implied:{result['listing_id']}",
                                                 speaker="inferred", confidence=IMPLIED_CONFIDENCE,
                                                 set_at_index=state.msg_count)
            diffs.append({"field": "constraints.location", "old": None, "new": result["location"],
                          "source": f"implied:{result['listing_id']}"})
    elif tool == "create_chat_bridge" and isinstance(result, dict) and result.get("channel_id"):
        state.channel_id = result["channel_id"]
        diffs.append({"field": "channel_id", "old": None, "new": state.channel_id, "source": src})
    elif tool == "book_appointment" and isinstance(result, dict):
        state.appointment.status = "booked" if result.get("status") == "booked" else "failed"
        state.appointment.booking_id = result.get("booking_id")
        diffs.append({"field": "appointment.status", "old": "proposed", "new": state.appointment.status, "source": src})
    elif tool == "escalate_to_human" and isinstance(result, dict):
        for r in state.active_risks():
            if r.type.value in args.get("risk_types", []) and r.status == "open":
                r.status = "escalated"
                diffs.append({"field": "risks", "old": f"{r.type.value}:open", "new": "escalated", "source": src})
    elif tool == "handoff_to_human" and isinstance(result, dict):
        state.handed_off = True
        diffs.append({"field": "handed_off", "old": False, "new": True, "source": src})
    diffs += recompute(state, msg)
    return diffs


# --------------------------------------------------------------------------- #
# Derived fields
# --------------------------------------------------------------------------- #
def prune(state: State) -> list[dict]:
    """Drop low-confidence constraints nobody confirmed within a few messages."""
    diffs = []
    for name in type(state.constraints).model_fields:
        t = getattr(state.constraints, name)
        if isinstance(t, Tracked) and t.confidence < PRUNE_CONFIDENCE \
                and state.msg_count - t.set_at_index >= PRUNE_AFTER_MSGS:
            setattr(state.constraints, name, None)
            diffs.append({"field": f"constraints.{name}", "old": t.value, "new": None, "source": "prune:unconfirmed"})
    return diffs


def slot_coverage(state: State) -> float:
    score = 0.0
    for slot, w in SLOT_WEIGHTS.items():
        if slot == "odo":
            if state.cval("odo_max") is not None:
                score += w
            elif state.cval("odo_pref") is not None:
                score += w * 0.5
        elif state.cval(slot) is not None:
            score += w
    return round(score / sum(SLOT_WEIGHTS.values()), 3)


def _resolve(state: State, rtype: RiskType, diffs: list, why: str) -> None:
    r = state.risk(rtype)
    if r and r.status == "open":
        r.status = "resolved"
        diffs.append({"field": "risks", "old": f"{rtype.value}:open", "new": "resolved", "source": why})


def recompute(state: State, msg: Message) -> list[dict]:
    diffs: list[dict] = []
    sel = state.listing_context.selected
    neg = state.negotiation

    # price gap
    budget = state.cval("budget_max")
    ask = sel.ask_price if sel else None
    if budget and ask:
        neg.gap_vnd = ask - budget
        neg.gap_pct = round((ask - budget) / budget, 3)
        if neg.gap_pct > 0:
            sev = "medium" if neg.gap_pct > BIG_GAP else "low"
            r = state.risk(RiskType.PRICE_GAP)
            if r is None or r.status == "resolved":
                _add_risk(state, RiskSignal(type=RiskType.PRICE_GAP, severity=sev, evidence=msg.msg_id,
                                            note=f"ask {ask:,} vs budget {budget:,} ({neg.gap_pct:.0%})"), diffs)
            else:
                r.severity, r.note = sev, f"ask {ask:,} vs budget {budget:,} ({neg.gap_pct:.0%})"
                if msg.msg_id not in r.evidence:
                    r.evidence.append(msg.msg_id)
        else:
            _resolve(state, RiskType.PRICE_GAP, diffs, "gap closed")
    else:
        neg.gap_vnd = neg.gap_pct = None

    # high odo for its age
    if sel and sel.odo_km and sel.year:
        age = max(1, msg.timestamp.year - sel.year)
        odo_max = state.cval("odo_max")
        if sel.odo_km / age > HIGH_ODO_PER_YEAR or (odo_max and sel.odo_km > odo_max):
            _add_risk(state, RiskSignal(type=RiskType.HIGH_ODO, severity="low", evidence=msg.msg_id,
                                        note=f"{sel.odo_km:,} km over {age} years", listing_id=sel.listing_id), diffs)

    # explicit consent closes the contact request (a human does the exchange)
    if state.consent.buyer_contact_share is True:
        _resolve(state, RiskType.PII_REQUEST, diffs, "buyer consented")

    # "chiếc vision có thể xong giấy tờ 1-2 tuần ko?": the listing says the papers are pending but not when; only the
    # seller knows, so the question goes to them (live chat: "chưa có thông tin xác nhận" three times, nothing asked)
    if msg.sender.value == "buyer" and Intent.ASK_PAPERWORK.value in state.turn_intents and sel and sel.listing_id \
            and sel.paperwork_status in PAPERWORK_SPECIFIC and in_room(state, sel) and room_seller(state) \
            and not (state.relay and state.relay["msg"] == msg.msg_id):
        set_relay(state, relay_text(redact(msg.text)[0]) or redact(msg.text)[0], msg.msg_id)

    # stage
    target = _target_stage(state)
    if target != state.lead_stage and _transition_allowed(state.lead_stage, target):
        state.stage_history.append({"from": state.lead_stage.value, "to": target.value,
                                    "at": msg.msg_id, "reason": _stage_reason(state, target)})
        diffs.append({"field": "lead_stage", "old": state.lead_stage.value, "new": target.value, "source": msg.msg_id})
        state.lead_stage = target

    state.open_questions = build_open_questions(state)
    return diffs


def _target_stage(state: State) -> LeadStage:
    sel = state.listing_context.selected
    if state.handed_off:
        return LeadStage.DROPPED
    if state.appointment.status == "booked":
        return LeadStage.APPOINTMENT
    if sel and sel.ask_price and (state.negotiation.gap_pct or 0) > 0:
        return LeadStage.NEGOTIATION
    if (sel and sel.listing_id) or state.listing_context.candidates:
        return LeadStage.MATCHING
    return LeadStage.DISCOVERY


def _transition_allowed(cur: LeadStage, target: LeadStage) -> bool:
    if cur in (LeadStage.DROPPED,):
        return False
    if cur in (LeadStage.APPOINTMENT, LeadStage.CLOSING):
        return target in (LeadStage.CLOSING, LeadStage.DROPPED)
    return True   # NEGOTIATION -> MATCHING is allowed: buyer pivots to another bike


def _stage_reason(state: State, target: LeadStage) -> str:
    return {
        LeadStage.DROPPED: "handed off or buyer dropped",
        LeadStage.APPOINTMENT: "appointment booked",
        LeadStage.NEGOTIATION: f"ask price above budget by {state.negotiation.gap_pct:.0%}" if state.negotiation.gap_pct else "price gap",
        LeadStage.MATCHING: "listing selected or candidates found",
        LeadStage.DISCOVERY: "collecting requirements",
    }.get(target, "")


def viewing_on(state: State) -> bool:
    """The buyer wants to view the selected bike and nothing stops a viewing (the same checks as the bridge)."""
    sel, neg = state.listing_context.selected, state.negotiation
    if state.channel_id:
        return True
    if not (state.consent.buyer_bridge and sel and sel.listing_id):
        return False
    high = any(r.severity == "high" and r.status != "resolved" for r in state.active_risks())
    return not high and (neg.gap_pct or 0) <= MAX_GAP_TO_MEET


def _index(msg_id: Optional[str]) -> int:
    try:
        return int(str(msg_id).rsplit("#", 1)[1])
    except (IndexError, ValueError):
        return -1


FILLER = {"đi", "đi ạ", "đi nha", "đi nhé", "đi bạn", "nhé", "nha", "giúp", "giúp mình", "giúp em", "thử", "xem",
          "thử đi", "giùm", "dùm", "ạ", "với", "giúp với", "thử xem"}


def relay_text(text: str) -> Optional[str]:
    """"hỏi chủ xe thử địa chỉ nào, đường nào" -> "địa chỉ nào, đường nào"; "hỏi chủ xe đi" -> None (no question in it)."""
    t = re.sub(r"(?i)^\s*(ý\s+(là|mình là)\s+)?(bạn\s+|ad\s+|em\s+)?(đã\s+)?hỏi\s+(thử\s+|giúp\s+|giùm\s+|dùm\s+|lại\s+)?"
               r"(chủ xe|chủ|người bán|bên bán|anh bán|chị bán)?\s*(thử|giúp|giùm|dùm|xem)?\s*", "", text).strip(" ,.?!")
    return None if t.lower() in FILLER or len(t) < 3 else t


def set_relay(state: State, text: str, msg_id: str) -> None:
    """A new question for the seller. The same question asked again while unanswered is not sent twice."""
    cur = state.relay
    if cur and cur["text"].strip(" ?").lower() == text.strip(" ?").lower() \
            and not (state.seller_reply and state.seller_reply["index"] > _index(cur["msg"])):
        return
    state.relay = {"text": text, "msg": msg_id}
    state.asked_counts.pop("relay_question", None)          # a new question: the repeat limit starts again


def waiting_on(state: State, side: str) -> list[str]:
    """Parts of the viewing ("time", "place") that `side` still has to agree to or give."""
    ap, other = state.appointment, "seller" if side == "buyer" else "buyer"
    out = [s for s in ("time", "place")
           if getattr(ap, f"proposed_{s}") and getattr(ap, f"{s}_by") == other and not getattr(ap, f"{s}_agreed")]
    if side == "seller" and ap.proposed_place is None:
        out.append("place")                  # the seller's bike: the seller says where
    elif side == "seller" and not exact_place(ap.proposed_place):
        out.append("address")                # "Q1" is an area: the buyer needs a street address
    return out


def needs_agreement(state: State) -> bool:
    """Booking needs both sides when the bike's seller is in this chat."""
    return bool(room_seller(state)) and in_room(state, state.listing_context.selected)


def build_open_questions(state: State) -> list[OpenQuestion]:
    qs: list[OpenQuestion] = []
    lc, neg = state.listing_context, state.negotiation
    sel = lc.selected
    wants_view = any(i in state.seen_intents for i in (Intent.REQUEST_VIEWING.value, Intent.REQUEST_CONNECT.value))
    pii = state.risk(RiskType.PII_REQUEST)
    paper = state.risk(RiskType.PAPERWORK_PENDING)

    if pii and pii.status != "resolved" and state.consent.buyer_contact_share is None:
        qs.append(OpenQuestion(field="contact_consent", ask="buyer", priority=0,
                               reason="seller asked for buyer phone; need explicit consent"))
    if wants_view and not (sel and sel.listing_id):
        qs.append(OpenQuestion(field="listing_id", ask="buyer", priority=0,
                               reason="buyer wants to view but the bike is not identified"))
    if paper and paper.status != "resolved" and state.cval("urgency") is None:
        qs.append(OpenQuestion(field="urgency", ask="buyer", priority=1,
                               reason="paperwork pending; does buyer need transfer now?"))
    here = in_room(state, sel)          # can the seller in this chat answer about the selected bike?
    asks_discount = Intent.ASK_DISCOUNT.value in state.turn_intents
    if sel and sel.listing_id and not here and sel.listing_id in lc.detail_fetched \
            and not state.channel_id and not state.consent.buyer_bridge:
        qs.append(OpenQuestion(field="other_seller", ask="buyer", priority=0 if asks_discount else 1,
                               reason=f"{sel.listing_id} belongs to {sel.seller_id}, not the seller in this chat"))
    seller_said_no = neg.seller_firm_turns > 0 and "seller_floor_price" in state.asked_fields
    # over budget: ask the lowest price once; again only when the buyer asks for a discount (live chat: asked again
    # three messages later as "Người mua hỏi ... bớt thêm", while the buyer was asking about the papers)
    if here and sel and sel.seller_floor is None and not seller_said_no \
            and ((neg.gap_pct and neg.gap_pct > 0 and "seller_floor_price" not in state.asked_fields) or asks_discount):
        qs.append(OpenQuestion(field="seller_floor_price", ask="seller", priority=0 if asks_discount else 1,
                               reason="buyer asks for a discount" if asks_discount
                               else "need seller's lowest price to judge the gap"))
    if here and neg.buyer_offer and sel and sel.seller_floor is None and "relay_offer" not in state.asked_fields:
        qs.append(OpenQuestion(field="relay_offer", ask="seller", priority=1,
                               reason=f"buyer offered {neg.buyer_offer:,}; relay to seller"))
    reference = neg.buyer_offer or state.cval("budget_max")
    if sel and sel.seller_floor and reference and not state.consent.buyer_bridge \
            and (neg.buyer_offer is None or sel.seller_floor > neg.buyer_offer):
        qs.append(OpenQuestion(field="accept_floor", ask="buyer", priority=0 if asks_discount else 1,
                               reason=f"seller floor {sel.seller_floor:,} vs "
                                      f"{'offer' if neg.buyer_offer else 'budget'} {reference:,}"))
    if here and Intent.ASK_CONDITION.value in state.seen_intents and "condition_history" not in state.asked_fields:
        qs.append(OpenQuestion(field="condition_history", ask="seller", priority=1,
                               reason="buyer asked about flood / accident / fines; the listing record does not say"))
    ap = state.appointment
    if viewing_on(state) and ap.status != "booked":
        if ap.proposed_time is None:
            qs.append(OpenQuestion(field="appointment_time", ask="buyer", priority=1, reason="buyer wants to view, no hour yet"))
        if needs_agreement(state) and ap.proposed_time and (w := waiting_on(state, "seller")):
            qs.append(OpenQuestion(field="appointment_confirm", ask="seller", priority=1,
                                   reason=f"seller has to agree / give: {', '.join(w)}"))
        if needs_agreement(state) and (w := waiting_on(state, "buyer")):
            qs.append(OpenQuestion(field="appointment_ok", ask="buyer", priority=1,
                                   reason=f"buyer has to agree to the seller's {', '.join(w)}"))
    if state.relay and here and room_seller(state) \
            and state.asked_fields.get("relay_question", -1) < _index(state.relay["msg"]):
        qs.append(OpenQuestion(field="relay_question", ask="seller", priority=1,
                               reason=f"buyer asks the seller: {state.relay['text']}"))
    if neg.gap_pct and neg.gap_pct > BIG_GAP and lc.candidates and "choose_alternative" not in state.asked_fields:
        qs.append(OpenQuestion(field="choose_alternative", ask="buyer", priority=2,
                               reason="gap too large; offer alternatives"))
    if state.cval("budget_max") is None:
        qs.append(OpenQuestion(field="budget", ask="buyer", priority=2, reason="needed to search or judge price"))
    if state.cval("location") is None:
        qs.append(OpenQuestion(field="location", ask="buyer", priority=3, reason="needed to search"))
    picked = (sel and sel.listing_id) or lc.mentioned_model    # "bạn ưu tiên hãng nào" makes no sense after that
    if Intent.SEARCH.value in state.seen_intents and not picked \
            and (state.cval("brands") is None or state.cval("year_min") is None):
        qs.append(OpenQuestion(field="brands_year", ask="buyer", priority=3, reason="narrows the search"))
    if state.cval("odo_pref") == "low" and state.cval("odo_max") is None:
        qs.append(OpenQuestion(field="odo_max", ask="buyer", priority=5, reason="'odo thấp' needs a number"))
    return sorted(qs, key=lambda q: q.priority)
