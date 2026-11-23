"""Checks an LLM-written reply against what code decided this turn, before anything is sent.

The LLM writes freely from a MUST_SAY list (the template sentences, each tagged with who it is for). Code then checks:
  1. everyone with a MUST_SAY item gets a reply           (live chat: the planned seller follow-up was dropped)
  2. each item's numbers, times, bikes and addresses appear in the reply to that person, and the seller's own
     words when a buyer item passes on the seller's answer      (paid run w6#12: "vẫn là thông tin đã báo ở trên")
  3. no amount in a reply that state, tool results, the message or MUST_SAY don't contain (invented numbers)
  4. "đã hỏi người bán" only when a question really went to the seller   (live chat: "đã hỏi chủ xe rồi")
  5. no internal words ("kênh chat" when the seller is in the chat, "tool")
  6. no reply that repeats a recent message to the same person (signals.AGENT_REPEAT)
plus the promise guard and the wrong-bike guard that already existed.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from . import normalize as N
from . import signals as S
from .models import State

RECENT_ASK = 3          # "đã hỏi người bán" may refer to a seller question sent within this many messages
SELLER_FIELDS = {"seller_floor_price", "relay_offer", "condition_history", "appointment_confirm", "relay_question"}
ASKED_SELLER_RE = re.compile(r"(?i)(đã|vừa|mới)\s+(hỏi|nhắn|báo)\s+(lại\s+|thêm\s+)?(người bán|chủ xe|bên bán|chủ|anh bán|chị bán)")
INTERNAL = ["tool", "công cụ", "policy", "hệ thống chưa cho phép"]
_DOT_MONEY = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{3}){2,})\s*(?:đ|đồng|vnd|vnđ)?(?!\d)", re.I)
SELLER_WORDS_RE = re.compile(r"trả lời(?: về \u201c[^\u201d]*\u201d)?:?\s*\u201c([^\u201d]+)\u201d")
_THOUSAND = re.compile(r"(?i)(?<![\d.,])(\d{2,3})\s*(?:k|nghìn|ngàn)(?!\w)")


def money(text: str) -> set[int]:
    """Amounts in a reply: "24,5 triệu", "24.5tr", "24.500.000đ", "500 nghìn"."""
    out = {m.low for m in N.parse_money(text)} | {m.high for m in N.parse_money(text)}
    out |= {int(m.group(1).replace(".", "")) for m in _DOT_MONEY.finditer(text)}
    out |= {int(m.group(1)) * 1000 for m in _THOUSAND.finditer(text)
            if not re.match(r"\s*km", text[m.end():])}                  # "8k km" is odo
    return {v for v in out if v >= 100_000}


def hours(text: str) -> set[int]:
    """Clock hours named in a text; "4h chiều" also counts as 16."""
    out = set()
    for m in N.TIME_RE.finditer(text):
        h = int(m.group(1))
        out.add(h)
        if h < 12:
            out.add(h + 12)
    return out


def _ints(obj: Any) -> Iterable[int]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _ints(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _ints(v)


def allowed_money(state: State, items: list[dict], tool_results: list, message: Optional[str]) -> set[int]:
    lc, neg, c = state.listing_context, state.negotiation, state.constraints
    bikes = [x for x in [lc.selected, *lc.candidates, *lc.previous] if x]
    vals = {v for x in bikes for v in (x.ask_price, x.seller_floor) if v}
    vals |= {v for v in (neg.buyer_offer, neg.gap_vnd, c.budget_max.value if c.budget_max else None) if v}
    vals |= {v for v in _ints(tool_results) if v >= 100_000}
    vals |= {v for it in items for v in money(it["text"])}
    if message:
        vals |= money(message)
    budget = c.budget_max.value if c.budget_max else None
    base = list(vals)
    for a in base:                       # differences people say: "cao hơn ngân sách 500 nghìn", "giảm 500k"
        for b in base + ([budget] if budget else []):
            if a and b and a != b:
                vals.add(abs(a - b))
    return vals


def anchors(text: str) -> dict:
    return {"money": money(text), "hours": {int(m.group(1)) for m in N.TIME_RE.finditer(text)},
            "address": N.find_address(text), "models": {m[1] for m in N.find_models(text)},
            "words": [m.group(1) for m in SELLER_WORDS_RE.finditer(text)]}


def _bare(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", N.fold(text)).split())


def check(replies: list, items: list[dict], state: State, executed: list, tool_results: list,
          message: Optional[str], sent: Optional[dict] = None) -> list[str]:
    problems: list[str] = []
    by_to: dict[str, str] = {}
    for r in replies:
        by_to[r.to] = (by_to.get(r.to, "") + " " + r.text).strip()
    for to in {it["to"] for it in items}:
        if not by_to.get(to) and not by_to.get("both"):
            problems.append(f"no message to the {to}, but MUST_SAY has items for them")
    for it in items:
        text = (by_to.get(it["to"], "") + " " + by_to.get("both", "")).strip()
        if not text:
            continue
        a = anchors(it["text"])
        said_money, said_hours, folded = money(text), hours(text), N.fold(text)
        miss = [f"{v:,}đ" for v in a["money"] if v not in said_money]
        miss += [f"{h}h" for h in a["hours"] if h not in said_hours]
        if a["address"] and N.fold(a["address"]) not in folded:
            miss.append(a["address"])
        miss += [m for m in a["models"] if N.fold(m) not in folded]
        if it["to"] == "buyer":                     # the seller's answer is passed on in their words, not "như trên"
            miss += [f"\u201c{w}\u201d" for w in a["words"] if _bare(w) not in _bare(text)]
        if miss:
            problems.append(f"to the {it['to']}: leaves out {', '.join(miss)} from {it['text']!r}")
    ok_money = allowed_money(state, items, tool_results, message)
    for to, text in by_to.items():
        new = sorted(v for v in money(text) if v not in ok_money)
        if new:
            problems.append(f"to the {to}: amounts not in FACTS / TOOL_RESULTS / MUST_SAY: {', '.join(f'{v:,}đ' for v in new)}")
    asked_seller_now = any(a.name == "ask_clarification" and a.target == "seller" for a in executed)
    last = max((state.asked_fields[f] for f in SELLER_FIELDS if f in state.asked_fields), default=None)
    asked_seller_before = last is not None and state.msg_count - last <= RECENT_ASK   # not "rồi" about anything, ever
    buyer_text = by_to.get("buyer", "") + " " + by_to.get("both", "")
    if ASKED_SELLER_RE.search(buyer_text) and not (asked_seller_now or asked_seller_before):
        problems.append("tells the buyer the seller was asked, but no question was sent to the seller")
    for r in replies:                     # Phase 5 live runs: most quality flags were the same message sent again
        before = (sent or {}).get(r.to, [])[-S.WINDOW:]
        if any(S.repeats(r.text, b) for b in before):
            problems.append(f"to the {r.to}: repeats a message already in ALREADY_SENT; say only what is new, "
                            "and if you are still waiting on someone, say that in one short new sentence")
    room = state.participants.get("seller_id")
    sel = state.listing_context.selected
    seller_here = bool(sel and sel.seller_id and room and sel.seller_id == room)
    internal = INTERNAL + (["kênh chat"] if seller_here else [])
    for to, text in by_to.items():
        hit = [w for w in internal if N.fold(w) in N.fold(text)]
        if hit:
            problems.append(f"to the {to}: internal words {hit}")
    return problems
