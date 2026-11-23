"""Extract intents, facts and risk signals from ONE message.

Three implementations share one interface:
  RuleExtractor   - regex + phrase lists. Offline, deterministic, the test baseline.
  LLMExtractor    - Claude with a Pydantic output schema (structured output).
  HybridExtractor - rules for numbers (exact), LLM for meaning; falls back to
                    rules if the LLM is unavailable or returns invalid output.

The extractor only *proposes* facts. The reducer decides what enters state.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from . import normalize as N
from . import pii
from .llm import LLMUnavailable, StructuredLLM
from .models import Extraction, Fact, Intent, ListingRef, Message, RiskSignal, RiskType, Sender, State
from .reducer import room_listing

FACT_FIELDS = [
    "constraints.budget_max", "constraints.budget_flexibility", "constraints.location",
    "constraints.vehicle_type", "constraints.brands", "constraints.year_min",
    "constraints.odo_max", "constraints.odo_pref", "constraints.condition_pref",
    "constraints.urgency", "constraints.deal_breakers",
    "listing.mentioned_model", "listing.select_model", "listing.select_listing", "listing.brand", "listing.model",
    "listing.year", "listing.odo_km", "listing.ask_price", "listing.seller_floor",
    "listing.paperwork_status", "listing.seller_claims",
    "negotiation.seller_stance", "negotiation.buyer_stance", "negotiation.buyer_offer",
    "appointment.proposed_time", "appointment.proposed_place", "appointment.agree",
    "consent.buyer_bridge", "consent.buyer_contact_share",
]
INT_FIELDS = {"negotiation.buyer_offer", "constraints.budget_max", "constraints.year_min", "constraints.odo_max",
              "listing.year", "listing.odo_km", "listing.ask_price", "listing.seller_floor"}
BOOL_FIELDS = {"consent.buyer_bridge", "consent.buyer_contact_share", "appointment.agree"}

OBJECTION = ["cao quá", "đắt", "mắc", "cao", "chát"]
BUYER_OFFER = ["bớt", "được không", "trả", "lấy giá", "để lại", "fix"]
DEPOSIT = ["cọc", "đặt cọc", "giữ xe"]
FLOOR = ["bớt còn", "giảm còn", "thấp nhất", "để lại", "giá cuối"]
CAN_WAIT = ["không gấp", "ko gấp", "đợi được", "chờ được", "từ từ", "đợi vài tuần", "chờ vài tuần"]
# the buyer needs the bike (and its papers) soon: answers "sang tên ngay hay chờ vài tuần?"
NEED_SOON = ["gấp", "cần ngay", "lấy ngay", "liền", "trong tuần", "tuần này", "hôm nay", "ngày mai", "sang tên ngay"]
BUDGET_FIRM = ["tối đa", "chỉ mua", "chỉ có", "max", "không quá", "nhiều nhất"]
BUDGET_APPROX = ["tầm", "khoảng", "khoản", "cỡ", "budget", "ngân sách", "dưới", "trong tầm"]
SELLER_FIRM = ["khó giảm", "không bớt", "không giảm", "ko bớt", "ko giảm", "k bớt", "hết bớt", "giá hợp lý",
               "giá chót", "giá cuối", "là tối đa", "fix", "không thương lượng", "giảm sâu"]
SELLER_FLEX = ["có bớt", "thương lượng được", "bớt chút", "fix nhẹ", "có thương lượng"]
PAPER_WORDS = ["giấy tờ", "giấy", "hồ sơ", "cà vẹt", "sang tên"]
PAPER_BAD = ["chờ", "chưa", "đang rút", "rút hồ sơ", "mất", "giấy tay", "không chính chủ"]
INTERMEDIARY = ["trung gian", "môi giới", "bên thứ ba", "bán trực tiếp", "không qua"]
CONTACT = ["số điện thoại", "sđt", "sdt", "số đt", "zalo", "số phone"]
# the seller offers THEIR OWN contact (vs. asking for the buyer's)
OWN_CONTACT = ["số mình", "sđt mình", "zalo mình", "số tôi", "zalo tôi", "số anh", "zalo anh", "số chị",
               "add zalo", "kết bạn zalo", "nhắn zalo", "gọi mình", "gọi cho mình", "gọi trực tiếp",
               "gọi tôi", "liên hệ mình", "ib mình", "inbox mình"]
ASK_BUYER_CONTACT = ["người mua", "xin số", "cho mình số", "cho xin", "số của bạn", "số của em", "số bạn",
                     "số em", "gửi số", "cho số", "để lại số"]
# paying before seeing the bike, or paying outside the platform
PREPAY = ["trước", "chuyển", "ck", "chuyển khoản", "stk", "tài khoản", "giữ xe"]
CONDITION = ["thủy kích", "thuỷ kích", "ngập nước", "ngập", "tai nạn", "đâm đụng", "đụng", "phạt nguội",
             "làm máy", "bể máy", "đại tu", "sơn lại", "tua công tơ", "tua odo"]
# the buyer asks whether the price can come down, without naming a number
ASK_DISCOUNT = ["có giảm", "giảm thêm", "giảm được", "giảm giá", "có bớt", "bớt thêm", "bớt được", "bớt giá",
                "fix được", "fix thêm", "thương lượng"]
# the buyer accepts / picks something just offered
ACCEPT = ["được", "ok", "oki", "okie", "oke", "okay", "chốt", "lấy", "ưng", "chọn", "thích", "muốn xem", "cũng được"]
YES = ACCEPT + ["có", "ừ", "đồng ý", "kết nối", "nhờ", "giúp"]
RECENT_CANDIDATES = 2   # "cũng okie" picks the bike shown within this many messages
THIS_BIKE = ["xe này", "chiếc này", "con này", "xe đó của anh", "xe của anh", "xe của chị"]


def accepts(text: str) -> bool:
    """A yes / pick, ignoring the "được" inside a question like "giảm được không"."""
    f = N.fold(text)
    for p in ASK_DISCOUNT + ["được không", "được ko", "được k", "được hông"]:
        f = re.sub(r"(?<!\w)" + re.escape(N.fold(p)) + r"(?!\w)", " ", f)
    return N.has_any(f, ACCEPT) and not N.NEGATION_RE.search(f)
CLAIMS = ["máy zin", "zin", "giữ kỹ", "chính chủ", "bảo dưỡng", "không đâm đụng"]


def match_listing(text: str, state: State) -> Optional[str]:
    """Which shown bike does the buyer mean? Returns a listing_id to switch to, or None.

    Two bikes often share a model ("Vision"), so model alone never decides: year, district and price do.
    A bare "cũng okie" picks the only bike shown in the last couple of messages."""
    lc = state.listing_context
    pool = [c for c in lc.candidates + lc.previous if c.listing_id]
    if not pool:
        return None
    f = N.fold(text)
    # a year picks a bike only when it points at one ("chiếc 2022", "vision 2020"), never in a wish
    # like "đời 2020 trở lên"; a price only when it is not a budget ("tầm 25tr")
    floor_year = N.parse_year_min(text)
    pointing = r"(?:chiec|xe|con|cai|" + "|".join(re.escape(N.fold(m)) for m in N.MODELS) + r")\s+(?:\w+\s+)?"
    years = {y for y in N.parse_years(text) if y != floor_year and re.search(pointing + str(y), f)}
    prices = {m.low for m in N.parse_money(text)} if not N.has_any(text, BUDGET_APPROX + BUDGET_FIRM) else set()

    def score(c: ListingRef) -> int:
        s = 0
        if c.year and c.year in years:
            s += 2
        if c.district and re.search(r"(?<!\w)" + re.escape(N.fold(c.district)) + r"(?!\w)", f):
            s += 2
        if c.ask_price and c.ask_price in prices:
            s += 2
        if s and c.model and N.fold(c.model) in f:
            s += 1
        return s

    scored = sorted(((score(c), c.listing_id) for c in pool), reverse=True)
    best, best_id = scored[0]
    if best >= 2 and (len(scored) == 1 or scored[1][0] < best):
        sel = lc.selected
        if sel and sel.listing_id and score(sel) >= best:
            return None                       # the details fit the bike already being discussed
        return best_id
    shown_recently = lc.candidates_at is not None and state.msg_count - lc.candidates_at <= RECENT_CANDIDATES
    if len(lc.candidates) == 1 and shown_recently and not years and not prices and not N.find_models(text) \
            and accepts(text) and not N.has_any(text, THIS_BIKE):
        return lc.candidates[0].listing_id
    return None


DAY_WORDS = re.compile(r"(?<!\w)(mai|mốt|thứ|chủ nhật|hôm nay|nay|ngày)(?!\w)", re.I)


def same_day(when: str, text: str, state: State) -> str:
    """A bare hour while a time is on the table ("ok 3h" after "3h chiều mai") keeps that day and half of the day,
    so the other side's "ok 3h" reads as agreeing, not as 3am today."""
    cur = state.appointment.proposed_time
    if not cur or DAY_WORDS.search(text):
        return when
    try:
        from datetime import datetime
        c, w = datetime.fromisoformat(cur), datetime.fromisoformat(when)
    except ValueError:
        return when
    hour = w.hour + 12 if w.hour < 12 <= c.hour and not re.search(r"(?i)sáng", text) else w.hour
    return c.replace(hour=hour, minute=w.minute).isoformat()


class RuleExtractor:
    name = "rules"

    def extract(self, msg: Message, state: State, context: Optional[dict] = None) -> Extraction:
        ex = Extraction(msg_id=msg.msg_id, extractor=self.name)
        if msg.sender == Sender.AGENT:
            return ex
        if msg.sender == Sender.BUYER:
            self._buyer(msg, state, ex)
        else:
            self._seller(msg, state, ex)
        self._common(msg, ex)
        ex.intents = list(dict.fromkeys(ex.intents))
        seen, facts = set(), []
        for f in ex.facts:
            key = (f.field, json.dumps(f.value, ensure_ascii=False))
            if key not in seen:
                seen.add(key)
                facts.append(f)
        ex.facts = facts
        return ex

    # ------------------------------------------------------------------ #
    def _add(self, ex: Extraction, msg: Message, field: str, value: Any, conf: float = 0.95, raw: str = ""):
        ex.facts.append(Fact(field=field, value=value, source_msg=msg.msg_id,
                             speaker=msg.sender.value, confidence=conf, raw=raw or msg.text))

    def _buyer(self, msg: Message, state: State, ex: Extraction) -> None:
        t = msg.text
        if N.has_any(t, ["chào", "hello", "hi"]):
            ex.intents.append(Intent.GREETING)
        if N.has_any(t, ["tìm", "kiếm", "cần mua", "muốn mua", "giới thiệu", "gợi ý", "có xe nào", "xe nào ok"]):
            ex.intents.append(Intent.SEARCH)
        if vt := N.find_vehicle_type(t):
            self._add(ex, msg, "constraints.vehicle_type", vt)
        if loc := N.find_location(t):
            self._add(ex, msg, "constraints.location", loc)

        models = N.find_models(t)
        picked = match_listing(t, state)
        if picked:
            ex.intents.append(Intent.SELECT_LISTING)
            self._add(ex, msg, "listing.select_listing", picked, 0.9)
        if models and not picked:
            model = models[0][1]
            candidate_models = {c.model for c in state.listing_context.candidates}
            if model in candidate_models and N.has_any(t, ACCEPT):
                ex.intents.append(Intent.SELECT_LISTING)
                self._add(ex, msg, "listing.select_model", model)
            else:
                ex.intents.append(Intent.REFERENCE_LISTING)
                self._add(ex, msg, "listing.mentioned_model", model)
        elif not models:
            wanted, excluded = N.find_brands_polarity(t)
            if wanted:                    # "không lấy Honda nữa, thích Yamaha" -> Yamaha only
                self._add(ex, msg, "constraints.brands", wanted)
            elif excluded and (cur := state.cval("brands")):
                remaining = [b for b in cur if b not in excluded]
                if remaining and remaining != cur:
                    self._add(ex, msg, "constraints.brands", remaining, 0.9)

        if (y := N.parse_year_min(t)) is not None:
            self._add(ex, msg, "constraints.year_min", y)
        if "odo" in N.fold(t) and N.has_any(t, ["dưới", "tối đa", "không quá"]):
            odo = N.parse_odo(re.sub(r"(?i)(dưới|tối đa|không quá)", "", t))
            if odo:
                self._add(ex, msg, "constraints.odo_max", odo)
        if N.has_any(t, ["odo thấp", "ít đi", "odo ít", "đi ít"]):
            self._add(ex, msg, "constraints.odo_pref", "low", 0.9)
        if N.has_any(t, ["xe đẹp", "đẹp"]):
            self._add(ex, msg, "constraints.condition_pref", "good_looking", 0.7)

        for cl in N.clauses(t):
            if N.has_any(cl, DEPOSIT):
                # "chuyển khoản cọc trước 5 triệu": a deposit, never a price offer or a budget
                if N.has_any(cl, PREPAY):
                    ex.risks.append(RiskSignal(type=RiskType.DEPOSIT_REQUEST, severity="medium", evidence=msg.msg_id,
                                               note="buyer offers to pay a deposit before viewing"))
                continue
            is_offer = N.has_any(cl, BUYER_OFFER)
            ms = N.parse_money(cl, allow_bare=is_offer)
            if not ms:
                continue
            if N.has_any(cl, OBJECTION):
                ex.intents.append(Intent.OBJECT_PRICE)
                continue
            if Intent.SELECT_LISTING in ex.intents and N.find_models(cl):
                continue  # price of the listing they are picking, not a budget
            lc = state.listing_context
            known = {x.ask_price for x in [lc.selected, *lc.candidates, *lc.previous] if x and x.ask_price}
            if not is_offer and any(m.low in known for m in ms) \
                    and (Intent.SELECT_LISTING in ex.intents or models or N.parse_years(t)
                         or not N.has_any(cl, BUDGET_FIRM + BUDGET_APPROX)):
                # "chiếc vision 2020, 21t mà" / "xe ở gò vấp 21 triệu": the price of a bike they point at, not a budget
                continue
            if is_offer:   # "25 được không?" is a counter-offer, never a new budget
                ex.intents.append(Intent.OFFER_PRICE)
                self._add(ex, msg, "negotiation.buyer_offer", min(m.low for m in ms), 0.85, cl)
                continue
            firm, approx = N.has_any(cl, BUDGET_FIRM), N.has_any(cl, BUDGET_APPROX)
            value = max(m.high for m in ms)
            self._add(ex, msg, "constraints.budget_max", value, 0.95 if (firm or approx) else 0.55, cl)
            if firm or approx:
                self._add(ex, msg, "constraints.budget_flexibility", "firm" if firm else "approx", 0.9, cl)
            if firm:
                self._add(ex, msg, "negotiation.buyer_stance", "firm", 0.8, cl)

        if N.has_any(t, PAPER_WORDS):
            if "?" in t or N.has_any(t, ["sao", "thế nào", "như nào", "ra sao"]):
                ex.intents.append(Intent.ASK_PAPERWORK)
            if N.has_any(t, ["ngại", "lo", "sợ", "rủi ro"]):
                ex.intents.append(Intent.PAPERWORK_CONCERN)
                self._add(ex, msg, "constraints.deal_breakers", "paperwork_issue", 0.8)

        just_asked = state.msg_count - state.asked_fields.get("urgency", -99) <= 2
        if N.has_any(t, CAN_WAIT):
            self._add(ex, msg, "constraints.urgency", "can_wait", 0.9)
        elif N.has_any(t, ["gấp", "liền", "ngay"]) and N.has_any(t, ["sang tên"]):
            self._add(ex, msg, "constraints.urgency", "transfer_now", 0.9)
        elif N.has_any(t, NEED_SOON) and (just_asked or N.has_any(t, ["cần", "muốn", "lấy", "mua"])):
            # "cần trong tuần này" right after "sang tên ngay hay chờ vài tuần?"
            self._add(ex, msg, "constraints.urgency", "transfer_now", 0.9 if just_asked else 0.8)
        if N.has_any(t, ["xem xe", "coi xe", "đi xem", "muốn xem", "xem thử"]):
            ex.intents.append(Intent.REQUEST_VIEWING)
            self._add(ex, msg, "consent.buyer_bridge", True, 0.9)
        if N.has_any(t, ["tự liên hệ"]):
            ex.intents.append(Intent.ASK_INFO)
        elif N.has_any(t, ["kết nối", "liên hệ", "gặp người bán"]):
            ex.intents.append(Intent.REQUEST_CONNECT)
            self._add(ex, msg, "consent.buyer_bridge", True, 0.9)
        if N.has_any(t, CONDITION):
            ex.intents.append(Intent.ASK_CONDITION)
        if re.search(r"(?i)hỏi\s+(thử\s+|giúp\s+|giùm\s+|dùm\s+|lại\s+)?(chủ xe|chủ|người bán|bên bán)", t) \
                and not N.ASKED_YET_RE.search(t):
            ex.intents.append(Intent.ASK_SELLER)
        if N.has_any(t, ASK_DISCOUNT) and Intent.OFFER_PRICE not in ex.intents:
            ex.intents.append(Intent.ASK_DISCOUNT)
        # "có" / "ok" right after the agent offered to connect with another seller's bike
        if state.asked_fields.get("other_seller") == state.msg_count and (accepts(t) or N.has_any(t, YES[len(ACCEPT):])) \
                and not N.has_any(t, ["không", "ko", "thôi", "chưa"]):
            ex.intents.append(Intent.REQUEST_CONNECT)
            self._add(ex, msg, "consent.buyer_bridge", True, 0.85)
        if "?" in t and Intent.ASK_PAPERWORK not in ex.intents:
            ex.intents.append(Intent.ASK_INFO)

        if N.has_any(t, ["không muốn cho số", "đừng cho số", "không cho số"]):
            self._add(ex, msg, "consent.buyer_contact_share", False)
        elif N.has_any(t, ["cho số", "gửi số", "đưa số", "cho sđt", "đồng ý chia sẻ", "chia sẻ số"]):
            ex.intents.append(Intent.CONSENT_SHARE_CONTACT)
            self._add(ex, msg, "consent.buyer_contact_share", True)

        if re.search(r"(?i)(thứ\s*(\d|hai|ba|tư|năm|sáu|bảy)|chủ nhật|ngày mai|\d{1,2}\s*(h|giờ)(?!\w))", t):
            when, place = N.parse_appointment(t, msg.timestamp)
            if when:
                ex.intents.append(Intent.SCHEDULE)
                self._add(ex, msg, "appointment.proposed_time", same_day(when, t, state), 0.85)
            if place:
                self._add(ex, msg, "appointment.proposed_place", place, 0.85)
        elif (state.consent.buyer_bridge or state.channel_id) and (am := N.find_address(t)):
            self._add(ex, msg, "appointment.proposed_place", am, 0.85)
        elif state.listing_context.selected and re.search(r"(?<!\w)(mai|mốt|cuối tuần)(?!\w)", t.lower()):
            ex.intents.append(Intent.REQUEST_VIEWING)     # "mai hoặc mốt thì đc": a day, no hour yet
        self._agree(msg, state, ex)

        if N.has_any(t, ["thôi không mua", "không cần nữa", "mua rồi", "hủy"]):
            ex.intents.append(Intent.DROP)

    def _agree(self, msg: Message, state: State, ex: Extraction) -> None:
        """"ok" / "được" while the other side's time or place waits for this side's answer."""
        ap, who = state.appointment, msg.sender.value
        pending = any(getattr(ap, f"proposed_{s}") and getattr(ap, f"{s}_by") not in (None, who)
                      and not getattr(ap, f"{s}_agreed") for s in ("time", "place"))
        if pending and accepts(msg.text):
            self._add(ex, msg, "appointment.agree", True, 0.85)

    def _seller(self, msg: Message, state: State, ex: Extraction) -> None:
        t = msg.text
        models = N.find_models(t)
        if models:
            brand, model, _ = models[0]
            self._add(ex, msg, "listing.model", model)
            self._add(ex, msg, "listing.brand", brand)
            years = N.parse_years(t)
            if years:
                self._add(ex, msg, "listing.year", years[0])
        if (odo := N.parse_odo(t)) is not None:
            self._add(ex, msg, "listing.odo_km", odo)

        for cl in N.clauses(t):
            if N.has_any(cl, DEPOSIT):
                continue      # "chuyển cọc trước 2 triệu": a deposit amount, not the price
            cut = N.parse_discount(cl)
            own = room_listing(state)
            if cut and own and own.ask_price and cut < own.ask_price * 0.5:
                # "bớt 500k" on a 24.5tr bike: the seller's new lowest price is 24tr
                self._add(ex, msg, "listing.seller_floor", own.ask_price - cut, 0.85, cl)
                continue
            is_floor = N.has_any(cl, FLOOR)
            ms = N.parse_money(cl, allow_bare=is_floor)
            if not ms:
                continue
            if is_floor:
                self._add(ex, msg, "listing.seller_floor", min(m.low for m in ms), 0.9 if N.MONEY_RE.search(cl) else 0.75, cl)
            else:
                ex.intents.append(Intent.OFFER_PRICE)
                self._add(ex, msg, "listing.ask_price", max(m.high for m in ms), 0.95, cl)

        self._agree(msg, state, ex)
        arranging = state.appointment.proposed_time or state.consent.buyer_bridge or state.channel_id
        if arranging and (N.TIME_RE.search(t) or N.PLACE_RE.search(t) or N.find_address(t)
                          or N.DISTRICT_RE.search(t)):
            # while a viewing is being arranged: "ok em, 4h qua 12 Phan Xích Long nhé" / "hẹn ở quận 1 được ko"
            when, place = N.parse_appointment(t, msg.timestamp)
            if when and N.TIME_RE.search(t):
                self._add(ex, msg, "appointment.proposed_time", same_day(when, t, state), 0.85)
            if place:
                self._add(ex, msg, "appointment.proposed_place", place, 0.85)

        if N.has_any(t, SELLER_FIRM):
            ex.intents.append(Intent.FIRM_PRICE)
            self._add(ex, msg, "negotiation.seller_stance", "firm", 0.85)
        elif N.has_any(t, SELLER_FLEX):
            self._add(ex, msg, "negotiation.seller_stance", "flexible", 0.85)
        for claim in CLAIMS:
            if N.has_any(t, [claim]):
                self._add(ex, msg, "listing.seller_claims", claim, 0.7)

        if N.has_any(t, PAPER_WORDS) and N.has_any(t, PAPER_BAD):
            status = "pending_original_docs" if N.has_any(t, ["rút hồ sơ", "hồ sơ gốc"]) else "issue_reported"
            self._add(ex, msg, "listing.paperwork_status", status, 0.9)
            ex.risks.append(RiskSignal(type=RiskType.PAPERWORK_PENDING, severity="high",
                                       evidence=msg.msg_id, note="seller reports paperwork not ready"))
        if N.has_any(t, INTERMEDIARY):
            ex.intents.append(Intent.REJECT_INTERMEDIARY)
            ex.risks.append(RiskSignal(type=RiskType.DISINTERMEDIATION, severity="medium",
                                       evidence=msg.msg_id, note="seller refuses platform involvement"))
        if (N.has_any(t, DEPOSIT) and N.has_any(t, PREPAY)) or N.has_any(t, ["chuyển khoản trước", "ck trước"]):
            ex.risks.append(RiskSignal(type=RiskType.DEPOSIT_REQUEST, severity="high",
                                       evidence=msg.msg_id, note="seller asks for deposit before viewing"))
        elif pii.has_account(t) or (N.has_any(t, ["ck", "chuyển khoản"]) and N.has_any(t, ["stk", "tài khoản"])):
            ex.risks.append(RiskSignal(type=RiskType.DEPOSIT_REQUEST, severity="high",
                                       evidence=msg.msg_id, note="seller asks for a bank transfer outside the platform"))

    def _common(self, msg: Message, ex: Extraction) -> None:
        t = msg.text
        if msg.sender == Sender.SELLER:
            # a bank account is a payment risk (DEPOSIT_REQUEST above), not a contact
            contact_kinds = set(pii.redact(t)[1]) - {"account", "number"}
            shares_own = bool(contact_kinds) or N.has_any(t, OWN_CONTACT)
            asks_buyer = N.has_any(t, CONTACT) and N.has_any(t, ASK_BUYER_CONTACT)
            if shares_own:
                ex.intents.append(Intent.SHARE_CONTACT)
                # low: the platform is still usable (unlike "không qua trung gian", which is medium)
                ex.risks.append(RiskSignal(type=RiskType.DISINTERMEDIATION, severity="low", evidence=msg.msg_id,
                                           note="seller posts own contact to move the deal off-platform"))
            if asks_buyer or (N.has_any(t, CONTACT) and not shares_own):
                ex.intents.append(Intent.REQUEST_CONTACT)
                ex.risks.append(RiskSignal(type=RiskType.PII_REQUEST, severity="medium",
                                           evidence=msg.msg_id, note="seller asks for buyer's phone number"))
        if N.has_any(t, ["không giấy", "mất giấy", "xe không giấy"]):
            ex.risks.append(RiskSignal(type=RiskType.STOLEN_RISK, severity="high",
                                       evidence=msg.msg_id, note="bike may have no registration"))


# --------------------------------------------------------------------------- #
# LLM extractor
# --------------------------------------------------------------------------- #
class LLMFact(BaseModel):
    field: Literal[tuple(FACT_FIELDS)]  # type: ignore[valid-type]
    value: str = Field(description="Normalised value as text. Money and km as plain integers (25000000, 19000).")
    confidence: float = Field(ge=0, le=1)
    raw: str = Field(description="The exact words in the message this came from")


class LLMRisk(BaseModel):
    type: RiskType
    severity: Literal["low", "medium", "high"]
    note: str


class LLMExtraction(BaseModel):
    intents: list[Intent]
    facts: list[LLMFact]
    risks: list[LLMRisk]


EXTRACT_SYSTEM = """You read one new message in a 3-way chat (buyer, seller, marketplace agent) about a used motorbike in Vietnam and extract structured information.

Rules:
- Extract only what THIS message states. Never infer values that are not in the text.
- Money: "25tr" / "25 triệu" / "25 củ" = 25000000. A range "25-26tr" for a budget means budget_max = 26000000.
- A price the buyer complains about ("32tr cao quá") is NOT their budget.
- Odo: "19k" = 19000 km.
- constraints.* are the BUYER's wishes. listing.* describe the bike being discussed (usually stated by the seller).
- listing.mentioned_model: buyer refers to a bike by model. listing.select_model: buyer picks one of the candidate bikes listed in context.
- listing.select_listing: the listing_id (e.g. "L002") when the buyer picks or refers to a specific bike from context.state.candidates
  or the selected bike, using any detail (year, district, price) or "cái đó / cũng okie" right after it was offered.
  Use this, not select_model, when two bikes share a model.
- A seller's cut BY an amount ("bớt 500k") sets listing.seller_floor = ask price minus that amount.
- Intent ASK_DISCOUNT when the buyer asks whether the price can come down without naming a number ("có giảm được không").
- negotiation.seller_stance: "firm" if the seller refuses to lower the price, "flexible" if open to it.
- constraints.urgency: "transfer_now" if the buyer needs ownership transfer immediately, "can_wait" otherwise.
- consent.buyer_contact_share: "true" only if the buyer explicitly agrees to share their phone number.
- Risks: PAPERWORK_PENDING (papers not ready / cannot transfer), DISINTERMEDIATION (seller refuses the platform),
  PII_REQUEST (someone asks for the BUYER's phone number / Zalo),
  DISINTERMEDIATION with severity "low" when the seller posts their OWN phone / Zalo,
  DEPOSIT_REQUEST (deposit or bank transfer before viewing: "high" from the seller, "medium" if the buyer offers),
  STOLEN_RISK (no registration).
- A deposit amount ("cọc trước 5 triệu") is never a price offer or a budget.
- Intent ASK_CONDITION when the buyer asks about flood damage, accidents, engine rebuilds or traffic fines.
- appointment.proposed_time: ISO 8601 with +07:00 ("2026-11-23T15:00:00+07:00"), only when the message gives an hour.
  A day alone ("chiều mai") is not a time; leave it out. appointment.agree: "true" when this message accepts the
  time or place the OTHER side proposed (see context.state.appointment).
  appointment.proposed_place: the street address ("12 Lê Lợi, Q1") when given, else the area ("Q1").
- Intent ASK_SELLER when the buyer asks you to check something with the seller ("hỏi chủ xe thử địa chỉ nào").
- Intent DROP only when the buyer says they no longer want this bike or the deal ("thôi không mua", "em mua xe khác rồi").
  Agreeing to a price or asking to see the bike ("ok em lấy giá đó") is the opposite of DROP.
- Return empty lists when nothing applies."""


def _coerce(field: str, value: str) -> Any:
    v = value.strip()
    if field in INT_FIELDS:
        digits = re.sub(r"[^\d.]", "", v)
        return int(float(digits)) if digits else None
    if field in BOOL_FIELDS:
        return v.lower() in {"true", "yes", "1", "có"}
    if field == "constraints.brands":
        return [b.strip() for b in re.split(r"[,/]| hoặc | or ", v) if b.strip()]
    return v


# intents only one side can have; an LLM sometimes tags the seller's "nếu bạn cần gấp giấy tờ..."
# as the buyer's PAPERWORK_CONCERN, which then triggers buyer-facing replies. Code decides.
BUYER_ONLY = {Intent.ASK_SELLER, Intent.SEARCH, Intent.ASK_INFO, Intent.ASK_PAPERWORK, Intent.PAPERWORK_CONCERN, Intent.ASK_CONDITION,
              Intent.ASK_DISCOUNT, Intent.OBJECT_PRICE, Intent.SELECT_LISTING, Intent.REFERENCE_LISTING,
              Intent.REQUEST_VIEWING, Intent.REQUEST_CONNECT, Intent.CONSENT_SHARE_CONTACT, Intent.DROP}
SELLER_ONLY = {Intent.FIRM_PRICE, Intent.REJECT_INTERMEDIARY, Intent.REQUEST_CONTACT, Intent.SHARE_CONTACT}

# DROP ends the lead, so an LLM DROP needs a withdrawal word in the text. Long-chat LLM eval, l2#40:
# "Ok em lấy giá đó, em muốn xem xe trực tiếp" came back as DROP and the lead was closed. Wider than the
# rule list (which acts on its own): here the LLM has already judged the meaning, the cue only confirms it.
DROP_CUES = ["không mua", "ko mua", "k mua", "không lấy", "ko lấy", "khỏi", "không cần", "ko cần", "mua rồi",
             "mua xe khác", "lấy xe khác", "hủy", "huỷ", "không quan tâm", "để sau", "bỏ qua", "thôi em", "thôi anh",
             "thôi chị", "thôi mình", "thôi bỏ", "không theo", "dừng"]


class LLMExtractor:
    name = "llm"

    def __init__(self, llm: StructuredLLM):
        self.llm = llm
        self.drops_rejected = 0

    def extract(self, msg: Message, state: State, context: Optional[dict] = None) -> Extraction:
        ex = Extraction(msg_id=msg.msg_id, extractor=self.name)
        if msg.sender == Sender.AGENT:
            return ex
        user = (
            "CONTEXT (do not extract from this, it is background):\n"
            + json.dumps(context or {}, ensure_ascii=False, default=str)
            + (f"\n\nTHE AGENT'S LAST MESSAGE TO THE {msg.sender.value.upper()} (a short reply like \"oki\" / \"được\" "
               f"answers this):\n{(context or {}).get('agent_last_to_sender')}" if (context or {}).get("agent_last_to_sender") else "")
            + f"\n\nNEW MESSAGE from {msg.sender.value} ({msg.msg_id}, {msg.timestamp.isoformat()}):\n{msg.text}"
        )
        out: LLMExtraction = self.llm.parse(EXTRACT_SYSTEM, user, LLMExtraction)
        wrong_side = SELLER_ONLY if msg.sender == Sender.BUYER else BUYER_ONLY
        ex.intents = [i for i in dict.fromkeys(out.intents) if i not in wrong_side]
        if Intent.DROP in ex.intents and not N.has_any(msg.text, DROP_CUES):
            ex.intents.remove(Intent.DROP)
            self.drops_rejected += 1
        for f in out.facts:
            val = _coerce(f.field, f.value)
            if val in (None, "", []):
                continue
            ex.facts.append(Fact(field=f.field, value=val, source_msg=msg.msg_id,
                                 speaker=msg.sender.value, confidence=f.confidence, raw=f.raw))
        ex.risks = [RiskSignal(type=r.type, severity=r.severity, evidence=msg.msg_id, note=r.note)
                    for r in out.risks]
        return ex


# Readings that need context: rules got these wrong in live chats, so when the LLM ran, only its reading counts
# (a rule-only reading is dropped). Every other field a rule finds is kept unless the LLM reads it differently.
# Meaning fields: when both read one, the LLM's reading wins; when only the rules read one, it is dropped, except a
# bike pick (BIKE_PICK), which the rules may still make when the LLM names no bike (Phase 5, live run 1: the LLM
# returned nothing for "chiếc 2022 thì có giảm thêm ko" and for "oki" to the offered bike, and the right pick was lost).
LLM_ONLY = {"listing.select_listing", "listing.select_model", "consent.buyer_bridge", "consent.buyer_contact_share",
            "appointment.agree", "appointment.proposed_time", "appointment.proposed_place",
            "negotiation.seller_stance", "negotiation.buyer_stance", "constraints.urgency"}
BIKE_PICK = {"listing.select_listing", "listing.select_model"}
# Intents that only say what kind of message it is (a question); harmless if a keyword over-fires.
RULE_INTENTS = {Intent.GREETING, Intent.SEARCH, Intent.ASK_DISCOUNT, Intent.ASK_SELLER, Intent.ASK_PAPERWORK,
                Intent.ASK_CONDITION, Intent.ASK_INFO}


class HybridExtractor:
    """The LLM decides what a message means; code checks it; rules are the fallback.

    - Meaning (which bike, consent, agreement, viewing time/place, stance, urgency, intents) comes from the LLM. Rule readings of these were the source of most live-chat errors ("ok 3h" read as 3am, "21 triệu" the
      bike's price read as a budget, "oki" not a yes) and used to override the LLM.
    - Other fields both read (numbers, fixed values like condition "good_looking"): the rules' value wins (format);
      a number disagreement is counted for review.
    - Rule facts the LLM is silent about are kept, except LLM_ONLY ones; a rule bike pick is kept when the LLM
      names no bike.
      Question-type intents in RULE_INTENTS are kept from the rules too.
    - Risks: union of both (recall matters for safety).
    - Code checks on the LLM's output: wrong-side intents, DROP needs a cue (LLMExtractor); sharing the buyer's
      contact needs the rules to agree too; normalising, impossible prices and times without an hour (reducer).
    - Any LLM failure: the rules alone.
    """
    name = "hybrid"

    def __init__(self, llm: StructuredLLM):
        self.rules = RuleExtractor()
        self.llm = LLMExtractor(llm)
        self.fallbacks = 0
        self.number_disagreements = 0
        self.overrides = 0            # LLM readings code refused (counted with fallbacks in the eval)

    def extract(self, msg: Message, state: State, context: Optional[dict] = None) -> Extraction:
        base = self.rules.extract(msg, state, context)
        if msg.sender == Sender.AGENT:
            return base
        try:
            llm_ex = self.llm.extract(msg, state, context)
        except LLMUnavailable as e:
            self.fallbacks += 1
            print(f"[extractor] LLM unavailable ({e}); using rules for {msg.msg_id}")
            return base
        self.overrides += self.llm.drops_rejected - getattr(self, "_drops_seen", 0)
        self._drops_seen = self.llm.drops_rejected
        rule_facts = {f.field: f for f in base.facts}
        lc = state.listing_context
        known = {x.ask_price for x in [lc.selected, *lc.candidates, *lc.previous] if x and x.ask_price}
        facts = []
        for f in llm_ex.facts:
            r = rule_facts.get(f.field)
            if r is not None and f.field not in LLM_ONLY:
                if f.field in INT_FIELDS and r.value != f.value:
                    self.number_disagreements += 1
                f = r                                       # same field: the rules' format wins
            elif f.field in INT_FIELDS:
                f.confidence = min(f.confidence, 0.8)       # a number the regex did not see
            if f.field == "constraints.budget_max" and r is None and f.value in known \
                    and not N.has_any(msg.text, BUDGET_FIRM + BUDGET_APPROX):
                self.overrides += 1                         # "Con Lead 2021 giá 26tr": a bike's price, not a budget
                continue
            if f.field == "consent.buyer_contact_share" and f.value is True and not (r and r.value is True):
                self.overrides += 1                         # sharing a phone number: both must read it
                continue
            facts.append(f)
        have = {f.field for f in facts}
        keep = lambda f: f.field not in LLM_ONLY or (f.field in BIKE_PICK and not have & BIKE_PICK)
        facts += [f for f in base.facts if f.field not in have and keep(f)]
        base.facts = facts
        base.intents = list(dict.fromkeys(llm_ex.intents + [i for i in base.intents if i in RULE_INTENTS]))
        known = {r.type for r in base.risks}
        base.risks += [r for r in llm_ex.risks if r.type not in known]
        base.extractor = self.name
        return base
