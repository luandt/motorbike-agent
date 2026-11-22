"""Mock internal marketplace APIs.

The four tools from the brief (search_listings, get_listing_detail,
create_chat_bridge, book_appointment) plus log_event, and two I added:
  escalate_to_human  - so the agent can *actually* "check the paperwork" (c2)
  handoff_to_human   - for consented direct contact (c3)
and one internal lookup, find_seller_listings, used to resolve which bike a
3-way chat is about. Every call is idempotent by (conversation, tool, args).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from .models import Event, EventType

TOOL_SCHEMAS = {
    "search_listings": {"brand": "list[str]|None", "model": "str|None", "year_range": "[min,max]",
                        "price_range": "[min,max] VND", "location": "str", "vehicle_type": "str",
                        "keywords": "list[str]", "paperwork_clean_only": "bool"},
    "get_listing_detail": {"listing_id": "str"},
    "find_seller_listings": {"seller_id": "str"},
    "create_chat_bridge": {"buyer_id": "str", "seller_id": "str", "listing_id": "str"},
    "book_appointment": {"channel_id": "str", "time": "ISO datetime", "place": "str"},
    "escalate_to_human": {"conversation_id": "str", "reason": "str", "severity": "low|medium|high", "risk_types": "list[str]"},
    "handoff_to_human": {"conversation_id": "str", "reason": "str"},
    "log_event": {"conversation_id": "str", "event": "Event"},
}


def args_hash(tool: str, args: dict) -> str:
    blob = json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class MockMarketplace:
    def __init__(self, listings_path: str | Path, store=None):
        self.listings: list[dict] = json.loads(Path(listings_path).read_text(encoding="utf-8"))
        self.by_id = {l["listing_id"]: l for l in self.listings}
        self.store = store
        self._cache: dict[str, Any] = {}

    # ---------------- brief tools ---------------- #
    def search_listings(self, brand=None, model=None, year_range=None, price_range=None, location=None,
                        vehicle_type=None, keywords=None, paperwork_clean_only=False, limit: int = 4) -> list[dict]:
        def ok(l: dict) -> bool:
            if brand and l["brand"] not in brand:
                return False
            if model and l["model"].lower() != str(model).lower():
                return False
            if year_range:
                lo, hi = year_range
                if (lo and l["year"] < lo) or (hi and l["year"] > hi):
                    return False
            if price_range:
                lo, hi = price_range
                if (lo and l["price"] < lo) or (hi and l["price"] > hi):
                    return False
            if location and l["location"] != location:
                return False
            if vehicle_type and l["vehicle_type"] != vehicle_type:
                return False
            if paperwork_clean_only and l["paperwork_status"] != "clean":
                return False
            if keywords and not any(k.lower() in json.dumps(l, ensure_ascii=False).lower() for k in keywords):
                return False
            return True

        mid = sum(price_range) / 2 if price_range and all(price_range) else 0
        hits = sorted(filter(ok, self.listings),
                      key=lambda l: (l["paperwork_status"] != "clean", abs(l["price"] - mid), l["odo_km"]))
        return [{k: l[k] for k in ("listing_id", "seller_id", "brand", "model", "year", "odo_km", "price",
                                   "location", "district", "paperwork_status")} for l in hits[:limit]]

    RELAXABLE = ("location", "price_range", "year_range", "brand", "model", "vehicle_type")

    def relax_empty_search(self, args: dict, exclude: tuple = ()) -> list[dict]:
        """For a search with no results: drop one filter at a time and report which ones would
        return bikes (a facet count, like a real marketplace's "0 results here, 5 nearby").
        Paperwork filters are never relaxed."""
        hints = []
        for key in self.RELAXABLE:
            if not args.get(key):
                continue
            hits = [h for h in self.search_listings(**{**args, key: None, "limit": 50})
                    if h["listing_id"] not in exclude]     # the bike already being discussed is no "alternative"
            if hits:
                hints.append({"field": key, "count": len(hits),
                              "locations": sorted({h["location"] for h in hits}),
                              "min_price": min(h["price"] for h in hits),
                              "max_price": max(h["price"] for h in hits),
                              "min_year": min(h["year"] for h in hits)})
        return hints

    def get_listing_detail(self, listing_id: str) -> dict:
        l = self.by_id.get(listing_id)
        return dict(l) if l else {"error": "not_found", "listing_id": listing_id}

    def create_chat_bridge(self, buyer_id: str, seller_id: str, listing_id: str) -> dict:
        if not (buyer_id and seller_id and listing_id):
            return {"error": "missing_participant"}
        return {"channel_id": "ch_" + args_hash("bridge", {"b": buyer_id, "s": seller_id, "l": listing_id})[:8]}

    def book_appointment(self, channel_id: str, time: str, place: str) -> dict:
        if not (channel_id and time and place):
            return {"status": "failed", "error": "missing_field"}
        return {"status": "booked", "booking_id": "bk_" + args_hash("book", {"c": channel_id, "t": time})[:8],
                "time": time, "place": place}

    def log_event(self, conversation_id: str, event: Event) -> None:
        if self.store is not None:
            self.store.append_event(event)

    # ---------------- added tools ---------------- #
    def find_seller_listings(self, seller_id: str) -> list[dict]:
        return [{k: l[k] for k in ("listing_id", "brand", "model", "year", "price")}
                for l in self.listings if l["seller_id"] == seller_id]

    def escalate_to_human(self, conversation_id: str, reason: str, severity: str, risk_types: list[str]) -> dict:
        return {"ticket_id": "esc_" + args_hash("esc", {"c": conversation_id, "r": risk_types})[:8],
                "queue": "trust_and_safety", "sla_minutes": 30}

    def handoff_to_human(self, conversation_id: str, reason: str) -> dict:
        return {"handoff_id": "ho_" + args_hash("ho", {"c": conversation_id})[:8], "queue": "sales_ops"}

    # ---------------- executor ---------------- #
    def call(self, conversation_id: str, tool: str, args: dict, at_msg: Optional[str] = None) -> tuple[Any, bool]:
        """Run a tool with logging + idempotency. Returns (result, was_cached)."""
        key = f"{conversation_id}:{args_hash(tool, args)}"
        self.log_event(conversation_id, Event(conversation_id=conversation_id, type=EventType.TOOL_CALL,
                                              at_msg=at_msg, payload={"tool": tool, "args": args}))
        cached = key in self._cache
        if not cached:
            fn = getattr(self, tool)
            try:
                self._cache[key] = fn(**args)
            except Exception as e:  # a tool failure must not crash the turn
                self._cache[key] = {"error": f"{type(e).__name__}: {e}"}
        result = self._cache[key]
        self.log_event(conversation_id, Event(conversation_id=conversation_id, type=EventType.TOOL_RESULT,
                                              at_msg=at_msg, payload={"tool": tool, "cached": cached, "result": result}))
        return result, cached
