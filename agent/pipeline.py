"""The agent loop for one incoming message.

  message -> log -> extract -> reduce -> resolve listing
          -> [policy gates -> plan -> execute tools -> reduce results] (<= MAX_STEPS)
          -> reply -> log -> promote memory -> save state
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import normalize as N
from .extractor import HybridExtractor, RuleExtractor
from .llm import make_llm
from .memory import Store, build_context, maybe_refresh_summary, pinned_messages, promote
from .models import Action, Event, EventType, Extraction, Message, Sender, State, ToolRecord
from .planner import LLMPlanner, RulePlanner
from .policy import gates
from .reducer import apply_extraction, apply_tool_result, new_state
from . import pii
from . import signals as S
from .responder import LLMResponder, Reply, TemplateResponder
from .tools import MockMarketplace, args_hash

ROOT = Path(__file__).resolve().parent.parent
MAX_STEPS = 3
# after these tools the state changes enough that we plan again in the same turn
REPLAN_AFTER = {"get_listing_detail", "search_listings", "create_chat_bridge"}
VIRTUAL = {"ask_clarification", "respond_only"}
# a diff on these refreshes the rolling summary
MATERIAL = {"constraints", "listing", "listing_context", "risks", "lead_stage", "channel_id", "appointment", "negotiation"}


@dataclass
class TurnResult:
    message: Message
    extraction: Optional[Extraction] = None
    diffs: list[dict] = field(default_factory=list)
    allowed: dict = field(default_factory=dict)
    blocked: dict = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
    state: Optional[State] = None
    legacy: bool = False
    signals: list[dict] = field(default_factory=list)


class Agent:
    def __init__(self, store: Store, mode: str = "rules", llm: Any = None,
                 listings_path: Path = ROOT / "data/listings.json",
                 conversations_path: Path = ROOT / "data/conversations.json",
                 fewshot_path: Path = ROOT / "data/fewshot.json"):
        self.store = store
        self.tools = MockMarketplace(listings_path, store)
        self.meta = json.loads(Path(conversations_path).read_text(encoding="utf-8"))
        # llm: a "provider:model" spec (see agent/llm.py), or any object with parse(system, user, Schema)
        llm = llm if hasattr(llm, "parse") else make_llm(mode, llm)
        self.llm = llm
        self.llm_name = getattr(llm, "name", "custom") if llm else "-"
        self.mode = "llm" if llm else "rules"
        self.extractor = HybridExtractor(llm) if llm else RuleExtractor()
        self.planner = LLMPlanner(llm, fewshot_path) if llm else RulePlanner()
        self.responder = LLMResponder(llm) if llm else TemplateResponder()

    def _log(self, conv: str, etype: EventType, payload: dict, at_msg: Optional[str]) -> None:
        self.tools.log_event(conv, Event(conversation_id=conv, type=etype, payload=payload, at_msg=at_msg))

    # ------------------------------------------------------------------ #
    def handle(self, msg: Message) -> TurnResult:
        conv = msg.conversation_id
        state = self.store.load_state(conv) or new_state(conv, self.meta.get(conv, {}))
        base_version = state.version
        # raw log keeps the message but never a phone number (PII minimisation)
        redacted, pii_kinds = pii.redact(msg.text)
        self._log(conv, EventType.USER_MESSAGE, {"msg_id": msg.msg_id, "sender": msg.sender.value,
                                                 "text": redacted, "pii_redacted": pii_kinds,
                                                 "ts": msg.timestamp.isoformat()}, msg.msg_id)

        # Messages written by the old agent in the replay log are history, not input.
        if msg.sender == Sender.AGENT:
            state.msg_count += 1
            self._log(conv, EventType.AGENT_ACTION, {"legacy": True, "replies": [{"to": "both", "text": msg.text}]}, msg.msg_id)
            self.store.save_state(state, base_version, msg.msg_id, [])
            return TurnResult(message=msg, state=state, legacy=True)

        context = build_context(state, self.store)
        to_sender = [m for m in context["recent_messages"] if m["from"] == "agent" and m.get("to") == msg.sender.value]
        context["agent_last_to_sender"] = to_sender[-1]["text"] if to_sender else None
        ex = self.extractor.extract(msg, state, context)
        old_stage = state.lead_stage
        old_bike = state.listing_context.selected.listing_id if state.listing_context.selected else None
        risks_before = {r.type.value: list(r.evidence) for r in state.risks}
        state, diffs = apply_extraction(state, ex, msg)
        diffs += self._resolve_listing(state, ex, msg)

        executed: list[tuple[Action, Any]] = []
        allowed, blocked = {}, {}
        for step in range(MAX_STEPS):
            allowed, blocked = gates(state)
            actions = self.planner.choose(state, allowed, blocked, build_context(state, self.store))
            done = {(x.name, x.args.get("field")) for x, _ in executed}
            actions = [a for a in actions if (a.name, a.args.get("field")) not in done
                       and not (a.name == "respond_only" and executed)]
            ran_info = False
            for a in actions:
                if a.name in VIRTUAL:
                    if a.name == "ask_clarification":
                        f = a.args["field"]
                        state.asked_fields[f] = state.msg_count
                        state.asked_counts[f] = state.asked_counts.get(f, 0) + 1
                    executed.append((a, None))
                    continue
                result = self._run_tool(state, a, msg, diffs)
                executed.append((a, result))
                ran_info |= a.name in REPLAN_AFTER
            if not ran_info or all(a.name in VIRTUAL for a in actions):
                break

        new_risks = self._new_risk_types(ex, risks_before)
        news = self._news(state, diffs)
        sent: dict[str, list[str]] = {}
        pinned = pinned_messages(state, self.store, self.store.recent_messages(conv))
        for h in self.store.recent_messages(conv, k=12):
            if h.get("from") == "agent" and h.get("to") in ("buyer", "seller"):
                sent.setdefault(h["to"], []).append(h["text"])
        replies = self.responder.write(state, executed, new_risks, [i.value for i in ex.intents], news, sent, pinned,
                                       {"from": msg.sender.value, "id": msg.msg_id, "text": redacted,
                                        "bike_changed": bool(old_bike) and old_bike != (
                                            state.listing_context.selected.listing_id if state.listing_context.selected else None)})
        flags = self._signals(state, msg, ex, replies, executed, news)
        material = state.lead_stage != old_stage or any(
            d["field"].split(".")[0] in MATERIAL for d in diffs)
        if maybe_refresh_summary(state, material):
            diffs.append({"field": "summary", "old": None, "new": state.summary, "source": "memory:refresh"})
        self._log(conv, EventType.STATE_UPDATE, {"extractor": ex.extractor, "intents": [i.value for i in ex.intents],
                                                 "diffs": diffs}, msg.msg_id)
        state.next_best_action = [a for a, _ in executed]
        self._log(conv, EventType.AGENT_ACTION, {
            "actions": [a.model_dump() for a, _ in executed],
            "allowed": list(allowed), "blocked": blocked,
            "replies": [r.model_dump() for r in replies],
        }, msg.msg_id)
        promote(state, self.store)
        self.store.save_state(state, base_version, msg.msg_id, [a for a, _ in executed])
        return TurnResult(message=msg, extraction=ex, diffs=diffs, allowed=allowed, blocked=blocked,
                          actions=[a for a, _ in executed], tool_results=[r for _, r in executed],
                          replies=replies, state=state, signals=flags)

    def _signals(self, state: State, msg: Message, ex: Extraction, replies: list, executed: list, news: list) -> list[dict]:
        """Flag this turn (or the previous one) from what the chat itself shows; log each flag."""
        conv = msg.conversation_id
        history = self.store.recent_messages(conv, k=12)[:-1]          # the last item is this message
        prev = [e for e in self.store.events(conv, [EventType.AGENT_ACTION]) if not e["payload"].get("legacy")]
        flags = S.detect(state, msg.sender.value, msg.text, [i.value for i in ex.intents], replies, history,
                         S.allowed_bikes(state, {a.name for a, _ in executed}, news),
                         prev[-1]["at_msg"] if prev else None)
        for f in flags:
            self._log(conv, EventType.QUALITY_SIGNAL, {**f, "at_msg": msg.msg_id,
                                                       "selected": state.listing_context.selected.listing_id
                                                       if state.listing_context.selected else None}, msg.msg_id)
        return flags

    # ------------------------------------------------------------------ #
    def _run_tool(self, state: State, a: Action, msg: Message, diffs: list) -> Any:
        result, cached = self.tools.call(state.conversation_id, a.name, a.args, msg.msg_id)
        status = "error" if isinstance(result, dict) and "error" in result else ("cached" if cached else "ok")
        state.tool_history.append(ToolRecord(tool=a.name, args_hash=args_hash(a.name, a.args), status=status,
                                             at_msg=msg.msg_id, summary=_short(result)))
        if a.name == "escalate_to_human":
            self._log(state.conversation_id, EventType.ESCALATION, {"args": a.args, "result": result}, msg.msg_id)
        if a.name == "handoff_to_human":
            self._log(state.conversation_id, EventType.HANDOFF, {"args": a.args, "result": result}, msg.msg_id)
        if status != "error":
            diffs += apply_tool_result(state, a.name, a.args, result, msg)
        if a.name == "search_listings":
            lc = state.listing_context
            seen = tuple(x.listing_id for x in [lc.selected, *lc.previous] if x and x.listing_id)
            hints = self.tools.relax_empty_search(a.args, exclude=seen) if result == [] else []
            state.listing_context.empty_search_hints = hints
            if hints:
                diffs.append({"field": "listing_context.empty_search_hints", "old": None,
                              "new": [h["field"] for h in hints], "source": "tool:search_listings"})
        return result

    def _resolve_listing(self, state: State, ex: Extraction, msg: Message) -> list[dict]:
        """Work out which listing a 3-way chat is about, using the seller's listings."""
        lc = state.listing_context
        sel = lc.selected
        if sel and sel.listing_id:
            return []
        seller = state.participants.get("seller_id")
        model = (sel.model if sel else None) or lc.mentioned_model
        seller_describes = msg.sender == Sender.SELLER and any(f.field.startswith("listing.") for f in ex.facts)
        # "xe này bao lâu có giấy tờ" in a chat opened from the seller's bike is about that bike (live chat:
        # "mình chưa có thông tin" and a budget question, while the listing says the papers are pending)
        talks_about_bike = model or seller_describes or N.THIS_BIKE_RE.search(msg.text) or any(
            i.value in ("REQUEST_VIEWING", "REQUEST_CONNECT", "ASK_PAPERWORK", "ASK_DISCOUNT", "ASK_CONDITION",
                        "REFERENCE_LISTING") for i in ex.intents)
        if not seller or not talks_about_bike:
            return []
        a = Action(name="find_seller_listings", args={"seller_id": seller}, reason="resolve which bike this chat is about")
        diffs: list[dict] = []
        options = self._run_tool(state, a, msg, diffs)
        matches = [o for o in options if model and o["model"] == model] if isinstance(options, list) else []
        if not matches and isinstance(options, list) and len(options) == 1 and (not model or seller_describes):
            matches = options
        if len(matches) == 1:
            m = matches[0]
            if lc.selected is None:
                from .models import ListingRef
                lc.selected = ListingRef(seller_id=seller, source_msg=msg.msg_id)
            lc.selected.listing_id, lc.selected.model, lc.selected.brand = m["listing_id"], m["model"], m["brand"]
            lc.selected.year = lc.selected.year or m["year"]
            lc.selected.ask_price = lc.selected.ask_price or m["price"]
            diffs.append({"field": "listing.listing_id", "old": None, "new": m["listing_id"], "source": "tool:find_seller_listings"})
            from .reducer import recompute
            diffs += recompute(state, msg)
        return diffs

    @staticmethod
    def _news(state: State, diffs: list[dict]) -> list[dict]:
        """Things the other party said this turn that the buyer must hear (e.g. a new lowest price)."""
        lc = state.listing_context
        out = []
        for d in diffs:
            if d["field"] == "constraints.urgency" and d.get("new"):
                out.append({"type": "urgency", "value": d["new"]})
            if d["field"] == "listing.seller_floor" and d.get("new"):
                ref = next((x for x in [lc.selected, *lc.previous] if x and x.listing_id == d.get("listing_id")),
                           lc.selected)
                bike = f"{ref.model} {ref.year}" if ref and ref.model else "này"
                out.append({"type": "seller_floor", "listing_id": d.get("listing_id"), "bike": bike,
                            "floor": d["new"], "ask": ref.ask_price if ref else None})
            if d["field"] in ("appointment.time_agreed", "appointment.place_agreed") and d.get("new") \
                    and state.appointment.status != "booked":
                slot = d["field"].split(".")[1].split("_")[0]
                ap = state.appointment
                out.append({"type": "appointment_agreed", "slot": slot, "by": "seller" if getattr(ap, f"{slot}_by") == "buyer" else "buyer",
                            "value": getattr(ap, f"proposed_{slot}")})
        return out

    @staticmethod
    def _new_risk_types(ex: Extraction, before: dict) -> set[str]:
        return {r.type.value for r in ex.risks if ex.msg_id not in before.get(r.type.value, [])}


def _short(result: Any) -> str:
    if isinstance(result, list):
        return f"{len(result)} result(s): " + ", ".join(r.get("listing_id", "?") for r in result[:5])
    if isinstance(result, dict):
        return ", ".join(f"{k}={v}" for k, v in list(result.items())[:3])
    return str(result)[:80]


def load_messages(paths: list[Path], conversation_id: Optional[str] = None) -> list[Message]:
    msgs = N.load_jsonl(paths)
    return [m for m in msgs if conversation_id is None or m.conversation_id == conversation_id]
