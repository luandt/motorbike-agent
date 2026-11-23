"""Data model: messages, extracted facts, per-lead state, actions and events.

The State is the single source of truth the planner reads. Every fact in it
keeps provenance (which message or tool produced it), so we can debug,
measure hallucination (a fact with no source) and supersede old values.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
class Sender(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"
    AGENT = "agent"


class Message(BaseModel):
    conversation_id: str
    msg_id: str                      # "c1#4" = 4th message of c1 (1-based)
    index: int
    timestamp: datetime
    sender: Sender
    text: str


# --------------------------------------------------------------------------- #
# Extraction output (what the extractor reads out of ONE message)
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    GREETING = "GREETING"
    SEARCH = "SEARCH"
    ASK_INFO = "ASK_INFO"
    ASK_PAPERWORK = "ASK_PAPERWORK"
    PAPERWORK_CONCERN = "PAPERWORK_CONCERN"
    OFFER_PRICE = "OFFER_PRICE"
    OBJECT_PRICE = "OBJECT_PRICE"
    FIRM_PRICE = "FIRM_PRICE"
    REFERENCE_LISTING = "REFERENCE_LISTING"
    SELECT_LISTING = "SELECT_LISTING"
    REQUEST_VIEWING = "REQUEST_VIEWING"
    REQUEST_CONNECT = "REQUEST_CONNECT"
    REJECT_INTERMEDIARY = "REJECT_INTERMEDIARY"
    REQUEST_CONTACT = "REQUEST_CONTACT"
    SHARE_CONTACT = "SHARE_CONTACT"        # seller posts their own phone / Zalo
    ASK_CONDITION = "ASK_CONDITION"        # flood damage, accidents, traffic fines
    ASK_SELLER = "ASK_SELLER"              # "hỏi chủ xe thử ...": the buyer wants a question passed to the seller
    ASK_DISCOUNT = "ASK_DISCOUNT"          # buyer asks whether the price can come down (no number)
    CONSENT_SHARE_CONTACT = "CONSENT_SHARE_CONTACT"
    SCHEDULE = "SCHEDULE"
    DROP = "DROP"


class RiskType(str, Enum):
    PRICE_GAP = "PRICE_GAP"
    PAPERWORK_PENDING = "PAPERWORK_PENDING"
    DISINTERMEDIATION = "DISINTERMEDIATION"
    PII_REQUEST = "PII_REQUEST"
    HIGH_ODO = "HIGH_ODO"
    DEPOSIT_REQUEST = "DEPOSIT_REQUEST"
    STOLEN_RISK = "STOLEN_RISK"


Severity = Literal["low", "medium", "high"]


class Fact(BaseModel):
    field: str                       # canonical path, e.g. "constraints.budget_max"
    value: Any
    source_msg: str                  # msg id or "tool:<name>"
    speaker: str                     # buyer | seller | tool
    confidence: float = 1.0
    raw: Optional[str] = None        # the text span it came from


class RiskSignal(BaseModel):
    type: RiskType
    severity: Severity
    evidence: str
    note: str = ""
    listing_id: Optional[str] = None       # set for risks about one bike (papers), None = whole deal


class Extraction(BaseModel):
    msg_id: str
    intents: list[Intent] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    risks: list[RiskSignal] = Field(default_factory=list)
    extractor: str = "rules"         # rules | llm | hybrid


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class LeadStage(str, Enum):
    DISCOVERY = "DISCOVERY"
    MATCHING = "MATCHING"
    NEGOTIATION = "NEGOTIATION"
    APPOINTMENT = "APPOINTMENT"
    CLOSING = "CLOSING"
    DROPPED = "DROPPED"


class Tracked(BaseModel):
    """A value with provenance. Old values go to `history` when superseded."""
    value: Any
    source_msg: Optional[str] = None
    speaker: Optional[str] = None
    confidence: float = 1.0
    set_at_index: int = 0
    history: list[dict] = Field(default_factory=list)


class Constraints(BaseModel):
    budget_max: Optional[Tracked] = None
    budget_flexibility: Optional[Tracked] = None   # firm | approx
    location: Optional[Tracked] = None
    vehicle_type: Optional[Tracked] = None         # scooter | underbone | manual_clutch
    brands: Optional[Tracked] = None               # list[str]
    year_min: Optional[Tracked] = None
    odo_max: Optional[Tracked] = None
    odo_pref: Optional[Tracked] = None             # "low"
    condition_pref: Optional[Tracked] = None
    urgency: Optional[Tracked] = None              # transfer_now | can_wait
    deal_breakers: list[str] = Field(default_factory=list)


class ListingRef(BaseModel):
    listing_id: Optional[str] = None
    seller_id: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    odo_km: Optional[int] = None
    ask_price: Optional[int] = None
    seller_floor: Optional[int] = None
    location: Optional[str] = None
    district: Optional[str] = None
    paperwork_status: Optional[str] = None
    seller_claims: list[str] = Field(default_factory=list)
    source_msg: Optional[str] = None


class ListingContext(BaseModel):
    selected: Optional[ListingRef] = None
    previous: list[ListingRef] = Field(default_factory=list)
    candidates: list[ListingRef] = Field(default_factory=list)
    mentioned_model: Optional[str] = None
    seller_listing_options: list[str] = Field(default_factory=list)  # when ambiguous
    detail_fetched: list[str] = Field(default_factory=list)
    # after a search with no results: which single filter, if dropped, would return bikes
    empty_search_hints: list[dict] = Field(default_factory=list)
    candidates_at: Optional[int] = None    # msg_count when the candidates were last shown
    candidates_unchanged: bool = False     # the latest search returned the bikes already shown
    candidates_narrowed: bool = False      # ...or only some of them ("honda" after "có xe nào giới thiệu")


class Negotiation(BaseModel):
    gap_vnd: Optional[int] = None
    gap_pct: Optional[float] = None
    seller_stance: Optional[str] = None            # firm | flexible
    buyer_stance: Optional[str] = None
    buyer_offer: Optional[int] = None              # counter-offer, NOT the budget
    seller_firm_turns: int = 0


class OpenQuestion(BaseModel):
    field: str
    ask: Literal["buyer", "seller"]
    priority: int
    reason: str = ""


class Risk(BaseModel):
    type: RiskType
    severity: Severity
    evidence: list[str] = Field(default_factory=list)
    status: Literal["open", "escalated", "resolved"] = "open"
    note: str = ""
    listing_id: Optional[str] = None       # a bike's risk only counts while that bike is selected


class Consent(BaseModel):
    buyer_bridge: Optional[bool] = None            # buyer wants platform chat with seller
    buyer_contact_share: Optional[bool] = None     # buyer agrees to share phone number


class Appointment(BaseModel):
    proposed_time: Optional[str] = None
    proposed_place: Optional[str] = None
    status: Literal["none", "proposed", "booked", "failed"] = "none"
    booking_id: Optional[str] = None
    source_msg: Optional[str] = None
    # each part is proposed by one side and must be agreed by the other before booking
    time_by: Optional[str] = None         # buyer | seller
    place_by: Optional[str] = None
    time_agreed: bool = False
    place_agreed: bool = False


class Action(BaseModel):
    name: str                         # tool name or virtual action (ask_clarification, respond_only)
    args: dict[str, Any] = Field(default_factory=dict)
    target: Optional[Literal["buyer", "seller", "both"]] = None
    reason: str = ""


class ToolRecord(BaseModel):
    tool: str
    args_hash: str
    status: str
    at_msg: str
    summary: str = ""


class State(BaseModel):
    conversation_id: str
    participants: dict[str, Optional[str]] = Field(default_factory=dict)
    lead_stage: LeadStage = LeadStage.DISCOVERY
    stage_history: list[dict] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    listing_context: ListingContext = Field(default_factory=ListingContext)
    negotiation: Negotiation = Field(default_factory=Negotiation)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    consent: Consent = Field(default_factory=Consent)
    channel_id: Optional[str] = None
    appointment: Appointment = Field(default_factory=Appointment)
    handed_off: bool = False
    seen_intents: list[str] = Field(default_factory=list)
    turn_intents: list[str] = Field(default_factory=list)      # intents of the latest message only
    relay: Optional[dict] = None      # {"text", "msg"}: the buyer's latest question for the seller (redacted)
    prev_buyer: Optional[dict] = None     # {"text", "msg"}: the buyer's previous message ("hỏi chủ xe đi" points at it)
    seller_reply: Optional[dict] = None   # {"text", "msg", "index", "about"}: the seller's answer to a question we sent
    last_from: dict[str, int] = Field(default_factory=dict)    # side -> msg_count of their latest message
    # "field" or "field@listing_id" -> id of the message that set a live price / consent / paperwork fact,
    # plus "candidates_shown" -> the turn whose agent reply listed the current candidates (see memory.pinned_refs)
    anchors: dict[str, str] = Field(default_factory=dict)
    asked_fields: dict[str, int] = Field(default_factory=dict)   # question field -> msg_count when asked
    asked_counts: dict[str, int] = Field(default_factory=dict)   # question field -> times asked
    next_best_action: list[Action] = Field(default_factory=list)
    tool_history: list[ToolRecord] = Field(default_factory=list)
    summary: str = ""
    msg_count: int = 0
    last_summary_at: int = 0
    version: int = 0
    updated_at: Optional[datetime] = None

    # ---- helpers ---------------------------------------------------------- #
    def applies(self, r: Risk) -> bool:
        """A bike's risk (pending papers on L002) stops counting when the buyer moves to another bike."""
        sel = self.listing_context.selected
        return r.listing_id is None or (sel is not None and sel.listing_id == r.listing_id)

    def active_risks(self) -> list[Risk]:
        return [r for r in self.risks if self.applies(r)]

    def risk(self, rtype: RiskType) -> Optional[Risk]:
        return next((r for r in self.active_risks() if r.type == rtype), None)

    def open_risks(self, min_severity: Severity = "low") -> list[Risk]:
        order = {"low": 0, "medium": 1, "high": 2}
        return [r for r in self.active_risks()
                if r.status != "resolved" and order[r.severity] >= order[min_severity]]

    def cval(self, name: str) -> Any:
        t = getattr(self.constraints, name)
        return t.value if isinstance(t, Tracked) else None


# --------------------------------------------------------------------------- #
# Events (append-only log)
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    USER_MESSAGE = "USER_MESSAGE"
    AGENT_ACTION = "AGENT_ACTION"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    STATE_UPDATE = "STATE_UPDATE"
    ESCALATION = "ESCALATION"
    HANDOFF = "HANDOFF"
    FEEDBACK = "FEEDBACK"
    QUALITY_SIGNAL = "QUALITY_SIGNAL"      # a flagged turn (agent/signals.py), queued for error analysis


class Event(BaseModel):
    conversation_id: str
    type: EventType
    payload: dict[str, Any]
    at_msg: Optional[str] = None
    ts: datetime = Field(default_factory=utcnow)
