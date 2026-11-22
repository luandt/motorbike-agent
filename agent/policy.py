"""Policy layer: risk escalation rules and stage gates.

`gates(state)` returns which actions are ALLOWED right now and why the others
are BLOCKED. The planner (rule or LLM) may only pick from `allowed`.
This is the "LLM proposes, code decides" boundary.
"""
from __future__ import annotations

from .models import RiskType, State
from .reducer import BIG_GAP, slot_coverage

ESCALATE_TYPES = {RiskType.PAPERWORK_PENDING, RiskType.DEPOSIT_REQUEST, RiskType.STOLEN_RISK}
MAX_GAP_TO_MEET = 0.10        # don't arrange a viewing when the price is >10% apart
SEARCH_MIN_COVERAGE = 0.6     # search once budget + location + type (or equivalent) are known; enforced in the gate

ACTION_CATALOG = {
    "ask_clarification": "Ask the buyer or seller one question from open_questions.",
    "search_listings": "Search listings matching the buyer's constraints (or alternatives when the price gap is large).",
    "get_listing_detail": "Fetch the full record of the selected listing (price, odo, paperwork status).",
    "create_chat_bridge": "Open a platform chat channel between buyer and the listing's seller.",
    "book_appointment": "Book a viewing in the open channel at the proposed time and place.",
    "escalate_to_human": "Send the case to a human operator because of a high-severity risk.",
    "handoff_to_human": "Hand the lead to a human (e.g. buyer consented to direct contact).",
    "respond_only": "Reply without calling any tool.",
}


def needs_escalation(state: State) -> list:
    return [r for r in state.active_risks()
            if r.type in ESCALATE_TYPES and r.severity == "high" and r.status == "open"]


def blocking_risks(state: State) -> list:
    """High-severity risks that are not resolved (escalated still blocks progress)."""
    return [r for r in state.active_risks() if r.severity == "high" and r.status != "resolved"]


def gates(state: State) -> tuple[dict[str, str], dict[str, str]]:
    allowed: dict[str, str] = {"respond_only": "always allowed"}
    blocked: dict[str, str] = {}
    lc, neg = state.listing_context, state.negotiation
    sel = lc.selected
    high = blocking_risks(state)
    high_txt = ", ".join(f"{r.type.value}({r.status})" for r in high)

    def gate(name: str, ok: bool, why_ok: str, why_not: str) -> None:
        (allowed if ok else blocked)[name] = why_ok if ok else why_not

    if state.handed_off:
        blocked.update({n: "lead handed off / dropped" for n in ACTION_CATALOG if n != "respond_only"})
        return allowed, blocked

    gate("ask_clarification", bool(state.open_questions),
         f"{len(state.open_questions)} open question(s)", "no open questions")

    esc = needs_escalation(state)
    gate("escalate_to_human", bool(esc), "open high risk: " + ", ".join(r.type.value for r in esc),
         "no unescalated high-severity risk")

    has_budget = state.cval("budget_max") is not None
    has_scope = any(state.cval(s) is not None for s in ("location", "vehicle_type", "brands")) or bool(lc.mentioned_model)
    cov = slot_coverage(state)
    alternatives = bool(sel and sel.ask_price) and (neg.gap_pct or 0) > BIG_GAP   # bike too expensive: look around
    search_block = (
        "need budget and one of location / vehicle type / brand" if not (has_budget and has_scope)
        else f"coverage {cov:.0%} < {SEARCH_MIN_COVERAGE:.0%}: ask more first" if cov < SEARCH_MIN_COVERAGE and not alternatives
        else None
    )
    gate("search_listings", search_block is None,
         f"coverage={cov}" + (f", gap {neg.gap_pct:.0%} > {BIG_GAP:.0%}: alternatives" if alternatives else ""),
         search_block or "")

    gate("get_listing_detail", bool(sel and sel.listing_id and sel.listing_id not in lc.detail_fetched),
         "selected listing not verified yet",
         "no identified listing" if not (sel and sel.listing_id) else "already fetched")

    bridge_block = (
        "no identified listing" if not (sel and sel.listing_id)
        else "channel already open" if state.channel_id
        else "verify listing detail first" if sel.listing_id not in lc.detail_fetched
        else "buyer has not asked to view / connect" if not state.consent.buyer_bridge
        else f"blocked by {high_txt}" if high
        else "seller refuses platform (DISINTERMEDIATION)" if (r := state.risk(RiskType.DISINTERMEDIATION)) and r.status == "open" and r.severity != "low"
        else f"price gap {neg.gap_pct:.0%} > {MAX_GAP_TO_MEET:.0%}" if (neg.gap_pct or 0) > MAX_GAP_TO_MEET
        else None
    )
    gate("create_chat_bridge", bridge_block is None, "listing verified, buyer wants to view, no blocking risk",
         bridge_block or "")

    ap = state.appointment
    book_block = (
        "no channel" if not state.channel_id
        else "already booked" if ap.status == "booked"
        else "no time/place agreed" if not (ap.proposed_time and ap.proposed_place)
        else f"blocked by {high_txt}" if high
        else f"price gap {neg.gap_pct:.0%}" if (neg.gap_pct or 0) > MAX_GAP_TO_MEET
        else None
    )
    gate("book_appointment", book_block is None, "channel open, time and place proposed", book_block or "")

    gate("handoff_to_human", state.consent.buyer_contact_share is True,
         "buyer consented to direct contact", "buyer has not consented to share contact")
    blocked["share_contact"] = "never done by the agent; only a human after explicit buyer consent"
    return allowed, blocked
