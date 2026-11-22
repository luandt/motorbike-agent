"""Choose the next best action(s) from the ALLOWED set.

RulePlanner - priority list, deterministic, the baseline.
LLMPlanner  - Claude picks WHAT to do (action + which question) from the allowed
              set; code fills in HOW (tool arguments come from state, never from
              the model, so arguments cannot be hallucinated). Invalid choices
              fall back to the rule planner; policy overrides always apply.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .llm import LLMUnavailable, StructuredLLM
from .models import Action, RiskType, State
from .policy import ACTION_CATALOG, SEARCH_MIN_COVERAGE
from .reducer import BIG_GAP, slot_coverage
from .tools import args_hash

MAX_ACTIONS = 2
RECENTLY_ASKED = 2   # don't repeat a (non-blocking) question asked within this many messages
MAX_REPEATS = 2      # a blocking question is asked at most this many times


# --------------------------------------------------------------------------- #
# Argument builders: code, not the LLM, decides tool arguments
# --------------------------------------------------------------------------- #
def search_args(state: State) -> dict:
    budget = state.cval("budget_max")
    firm = state.cval("budget_flexibility") == "firm"
    lo, hi = (0.85, 1.05) if firm else (0.8, 1.1)
    year_min = state.cval("year_min")
    return {
        "brand": state.cval("brands"),
        "model": None if state.cval("brands") else state.listing_context.mentioned_model,
        "year_range": [year_min, None] if year_min else None,
        "price_range": [int(round(budget * lo, -5)), int(round(budget * hi, -5))] if budget else None,
        "location": state.cval("location"),
        "vehicle_type": state.cval("vehicle_type"),
        "keywords": None,
        "paperwork_clean_only": "paperwork_issue" in state.constraints.deal_breakers
                                or state.risk(RiskType.PAPERWORK_PENDING) is not None,
    }


def build_action(name: str, state: State, field: Optional[str] = None, reason: str = "") -> Action:
    sel = state.listing_context.selected
    p = state.participants
    if name == "search_listings":
        return Action(name=name, args=search_args(state), target="buyer", reason=reason)
    if name == "get_listing_detail":
        return Action(name=name, args={"listing_id": sel.listing_id}, reason=reason)
    if name == "create_chat_bridge":
        return Action(name=name, args={"buyer_id": p.get("buyer_id"), "seller_id": sel.seller_id,
                                       "listing_id": sel.listing_id}, target="both", reason=reason)
    if name == "book_appointment":
        ap = state.appointment
        return Action(name=name, args={"channel_id": state.channel_id, "time": ap.proposed_time,
                                       "place": ap.proposed_place}, target="both", reason=reason)
    if name == "escalate_to_human":
        risks = [r for r in state.active_risks() if r.severity == "high" and r.status == "open"]
        return Action(name=name, args={"conversation_id": state.conversation_id,
                                       "reason": "; ".join(r.note for r in risks),
                                       "severity": "high", "risk_types": [r.type.value for r in risks]},
                      target="buyer", reason=reason)
    if name == "handoff_to_human":
        return Action(name=name, args={"conversation_id": state.conversation_id,
                                       "reason": "buyer consented to direct contact"}, target="both", reason=reason)
    if name == "ask_clarification":
        q = next((q for q in state.open_questions if q.field == field), None)
        return Action(name=name, args={"field": field}, target=q.ask if q else "buyer",
                      reason=reason or (q.reason if q else ""))
    return Action(name="respond_only", reason=reason)


def already_ran(state: State, action: Action) -> bool:
    h = args_hash(action.name, action.args)
    return any(t.args_hash == h for t in state.tool_history)


def askable(state: State, field: str) -> bool:
    q = next((q for q in state.open_questions if q.field == field), None)
    if q is None:
        return False
    if state.asked_counts.get(field, 0) >= MAX_REPEATS:
        return False
    last = state.asked_fields.get(field)
    return q.priority == 0 or last is None or state.msg_count - last > RECENTLY_ASKED


# --------------------------------------------------------------------------- #
class RulePlanner:
    name = "rules"

    def choose(self, state: State, allowed: dict, blocked: dict, context: Optional[dict] = None) -> list[Action]:
        acts: list[Action] = []

        def add(name: str, field: Optional[str] = None, reason: str = "") -> None:
            if len(acts) >= MAX_ACTIONS or name not in allowed:
                return
            if name == "ask_clarification" and (not field or not askable(state, field)
                                                 or any(a.args.get("field") == field for a in acts)):
                return
            a = build_action(name, state, field, reason)
            if name != "ask_clarification" and (any(x.name == name for x in acts) or already_ran(state, a)):
                return
            acts.append(a)

        lc, neg = state.listing_context, state.negotiation
        sel = lc.selected
        qs = state.open_questions
        gap = neg.gap_pct or 0

        add("escalate_to_human", reason="high-severity risk must go to a human before anything else")
        add("handoff_to_human", reason="buyer consented to direct contact; a human handles the exchange")
        for q in qs:
            if q.priority == 0:
                add("ask_clarification", q.field, q.reason)
        add("get_listing_detail", reason="verify the listing record (price, paperwork) before acting on it")
        add("ask_clarification", "other_seller", "the bike belongs to a seller who is not in this chat")
        add("ask_clarification", "condition_history", "buyer asked about the bike's history; only the seller knows")
        if gap > BIG_GAP:
            add("search_listings", reason=f"gap {gap:.0%} with this bike; find alternatives in budget")
            add("ask_clarification", "seller_floor_price", "ask seller for lowest price")
            add("ask_clarification", "choose_alternative", "steer buyer to in-budget options")
        elif gap > 0:
            add("ask_clarification", "seller_floor_price", f"small gap {gap:.0%}; ask seller for lowest price")
        add("ask_clarification", "relay_offer", "pass the buyer's counter-offer to the seller")
        add("ask_clarification", "accept_floor", "seller named a floor; ask buyer to accept and view")
        add("book_appointment", reason="time and place proposed in an open channel")
        add("create_chat_bridge", reason="listing verified and buyer wants to view")
        add("ask_clarification", "appointment_time", "buyer wants to view, need a time")
        add("ask_clarification", "appointment_confirm", "the seller agrees to the buyer's time / gives the place")
        add("ask_clarification", "appointment_ok", "the buyer agrees to the seller's time / place")
        if not any(a.name == "ask_clarification" and a.target == "seller" for a in acts):
            # "hỏi chủ xe thử địa chỉ nào": pass it on, unless a seller question this turn already covers it
            add("ask_clarification", "relay_question", "the buyer asked to check something with the seller")
        add("ask_clarification", "urgency", "paperwork pending; check if buyer can wait")
        if not (sel and sel.listing_id) and (slot_coverage(state) >= SEARCH_MIN_COVERAGE
                                             or "brands_year" in state.asked_fields):   # asked once: show bikes
            # first search, or the buyer narrowed the request after one ("ưu tiên Honda, đời 2019"):
            # already_ran() skips it when the arguments have not changed
            add("search_listings", reason=f"enough requirements (coverage {slot_coverage(state):.0%})"
                if not lc.candidates else "requirements changed since the last search")
        blocking_pending = any(q.priority == 0 for q in qs)
        if not acts and not blocking_pending:   # don't drift to minor questions while a blocking one is open
            for q in qs:
                add("ask_clarification", q.field, q.reason)
                if acts:
                    break
        if not acts and not lc.candidates and not (sel and sel.listing_id):
            add("search_listings", reason="nothing to ask; search with what we know")
        if not acts:
            acts.append(Action(name="respond_only", reason="nothing to do but acknowledge"))
        return acts


# --------------------------------------------------------------------------- #
class PlannedAction(BaseModel):
    name: Literal[tuple(ACTION_CATALOG)]  # type: ignore[valid-type]
    question_field: Optional[str] = Field(None, description="For ask_clarification: one field from open_questions")
    reason: str


class PlannerOutput(BaseModel):
    actions: list[PlannedAction] = Field(description="1 or 2 actions, most important first")


PLAN_SYSTEM = """You are the decision step of a marketplace agent that helps a buyer and a seller close a used-motorbike deal in Vietnam.
Pick the next best 1-2 actions. You may ONLY choose actions listed under ALLOWED. Arguments are filled in by code.
Principles:
- Safety first: high-severity risks go to escalate_to_human; never move a deal forward past an open high risk.
- Never share personal contact details; ask the buyer for consent instead.
- Price gap above 15% with a firm seller: offer in-budget alternatives instead of pushing either side.
- Ask at most one question per person, the one that unblocks the most.
- Prefer actions that move the lead toward a booked viewing."""


class LLMPlanner:
    name = "llm"

    def __init__(self, llm: StructuredLLM, fewshot_path: Optional[Path] = None):
        self.llm = llm
        self.rules = RulePlanner()
        self.fewshot_path = fewshot_path
        self.fallbacks = 0
        self.overrides = 0          # LLM choices replaced by code (skipped work, missing consent question)

    def _fewshot(self) -> list:
        if self.fewshot_path and self.fewshot_path.exists():
            return json.loads(self.fewshot_path.read_text(encoding="utf-8"))[:6]
        return []

    def choose(self, state: State, allowed: dict, blocked: dict, context: Optional[dict] = None) -> list[Action]:
        user = json.dumps({
            "context": context or {},
            "open_questions": [q.model_dump() for q in state.open_questions],
            "ALLOWED": allowed,
            "BLOCKED": blocked,
            "examples_that_led_to_bookings": self._fewshot(),
        }, ensure_ascii=False, default=str)
        try:
            out: PlannerOutput = self.llm.parse(PLAN_SYSTEM, user, PlannerOutput)
        except LLMUnavailable as e:
            self.fallbacks += 1
            print(f"[planner] LLM unavailable ({e}); using rule planner")
            return self.rules.choose(state, allowed, blocked, context)

        acts: list[Action] = []
        q_fields = {q.field for q in state.open_questions}
        for p in out.actions[:MAX_ACTIONS]:
            if p.name not in allowed:
                continue                                   # LLM proposed something code forbids
            if p.name == "ask_clarification" and p.question_field not in q_fields:
                continue
            a = build_action(p.name, state, p.question_field, p.reason)
            if p.name != "ask_clarification" and already_ran(state, a):
                continue
            acts.append(a)
        if not acts:
            self.fallbacks += 1
            return self.rules.choose(state, allowed, blocked, context)
        # the LLM chooses HOW to move the lead, not WHETHER: "just reply" is not allowed while the rule
        # planner sees a tool to run or a question to ask (in the first OpenAI eval this was most misses)
        if all(a.name == "respond_only" for a in acts):
            rule_acts = self.rules.choose(state, allowed, blocked, context)
            if any(a.name != "respond_only" for a in rule_acts):
                self.overrides += 1
                return rule_acts
        # privacy: once the seller asks for the buyer's number, the buyer must be asked before anything else
        if askable(state, "contact_consent") and not any(a.args.get("field") == "contact_consent" for a in acts):
            self.overrides += 1
            acts.insert(0, build_action("ask_clarification", state, "contact_consent",
                                        "policy override: seller asked for the buyer's contact"))
        # policy override: escalation is not optional
        if "escalate_to_human" in allowed and all(a.name != "escalate_to_human" for a in acts):
            acts.insert(0, build_action("escalate_to_human", state, reason="policy override: open high risk"))
        return acts[:MAX_ACTIONS + 1]
