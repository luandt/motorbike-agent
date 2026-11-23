"""Turn executed actions into Vietnamese replies to buyer and/or seller.

TemplateResponder - deterministic templates; every sentence is backed by state
                    or a tool result.
LLMResponder      - Claude writes more natural replies from the same facts,
                    then guards run: no phone numbers, no promises of work
                    that no executed tool performs. A failed guard falls back
                    to the template reply and is counted (hallucination metric).
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel

from . import normalize as N
from . import pii
from . import signals as S
from .llm import LLMUnavailable, StructuredLLM
from .models import Action, RiskType, State

PHONE_RE = re.compile(r"(?:\+?84|0)(?:[\s.-]?\d){8,10}")
# commitment phrase -> actions that make it true. Offers ("mình có thể...") are fine;
# claims of work done or promised ("đã...", "sẽ...") need a tool call behind them.
PROMISES = {
    "sẽ kiểm tra": {"escalate_to_human", "get_listing_detail"},
    "đã chuyển cho": {"escalate_to_human", "handoff_to_human"},
    "đã đặt lịch": {"book_appointment"},
    "đã chốt lịch": {"book_appointment"},
    "chốt lịch xem": {"book_appointment"},
    "đã tạo": {"create_chat_bridge"},
    "sẽ liên hệ lại": {"escalate_to_human", "handoff_to_human"},
    "hỏi thêm": {"ask_clarification"},
    "tư vấn kỹ": {"escalate_to_human"},
}

QUESTIONS = {
    "brands_year": ("buyer", "Dạ bạn ưu tiên hãng nào và đời xe từ năm bao nhiêu ạ?"),
    "budget": ("buyer", "Bạn dự kiến ngân sách khoảng bao nhiêu để mình tìm thêm xe phù hợp ạ?"),
    "location": ("buyer", "Bạn ở khu vực nào để mình tìm xe gần bạn ạ?"),
    "odo_max": ("buyer", "Bạn muốn odo tối đa khoảng bao nhiêu km ạ?"),
    "urgency": ("buyer", "Bạn có cần sang tên ngay khi mua không, hay có thể chờ vài tuần để người bán hoàn tất hồ sơ ạ?"),
    "seller_floor_price": ("seller", "Anh/chị cho mình xin mức giá thấp nhất có thể bán, để mình trao đổi với người mua nhé?"),
    "listing_id": ("buyer", "Bạn đang quan tâm xe nào ạ?{options}"),
    "contact_consent": ("buyer", "Người bán muốn liên hệ trực tiếp với bạn. Bạn có đồng ý để mình chia sẻ số điện thoại không? "
                                 "Nếu chưa muốn, mình có thể mở kênh chat và đặt lịch xem xe ngay trên nền tảng."),
    "choose_alternative": ("buyer", "Bạn có muốn xem thử một trong các xe ở trên không ạ?"),
    "appointment_time": ("buyer", "Bạn muốn xem xe vào lúc nào và ở đâu ạ?"),
    "appointment_confirm": ("seller", ""),     # written from what the seller still has to agree to / give
    "appointment_ok": ("buyer", ""),
    "relay_question": ("seller", ""),
    "condition_history": ("seller", "Người mua hỏi xe đã từng bị ngập nước (thủy kích), tai nạn, làm lại máy "
                                    "hay còn phạt nguội chưa xử lý không. Anh/chị xác nhận giúp mình nhé?"),
    "relay_offer": ("seller", "Người mua trả {offer}, anh/chị có bớt được về mức này không ạ?"),
    "accept_floor": ("buyer", "Người bán để thấp nhất {floor}. Bạn có muốn hẹn xem xe với mức giá này không ạ?"),
}

# what the buyer is told when a case is escalated, per risk type
ESCALATE_TEXT = {
    "PAPERWORK_PENDING": "Trường hợp giấy tờ này mình đã chuyển cho bộ phận hỗ trợ giao dịch kiểm tra.",
    "STOLEN_RISK": "Trường hợp giấy tờ này mình đã chuyển cho bộ phận hỗ trợ giao dịch kiểm tra.",
    "DEPOSIT_REQUEST": "Yêu cầu chuyển cọc / chuyển khoản này mình đã chuyển cho bộ phận hỗ trợ giao dịch kiểm tra.",
}
DEPOSIT_WARNING = ("Bạn chưa nên chuyển cọc hay chuyển khoản trước khi xem xe và đối chiếu giấy tờ nhé. "
                   "Mình có thể mở kênh chat và đặt lịch xem xe với người bán ngay trên nền tảng.")

PAPER_TEXT = {"clean": "đầy đủ, chính chủ", "pending_original_docs": "đang chờ rút hồ sơ gốc, chưa sang tên ngay được",
              "issue_reported": "người bán báo có vấn đề"}


def vnd(v: Optional[int]) -> str:
    return f"{v/1e6:g}tr" if v else "?"


def _listing_line(x, budget: Optional[int] = None) -> str:
    notes = []
    if budget and x.ask_price and x.ask_price > budget:
        notes.append(f"cao hơn ngân sách {vnd(x.ask_price - budget)}")
    if x.paperwork_status and x.paperwork_status != "clean":
        notes.append("giấy tờ " + PAPER_TEXT.get(x.paperwork_status, x.paperwork_status))
    extra = ("; " + "; ".join(notes)) if notes else ""
    return f"- {x.model} {x.year}, odo {x.odo_km/1000:g}k km, {vnd(x.ask_price)} ({x.district or x.location}{extra})"


def _bike(x) -> str:
    where = f" ở {x.district}" if x.district else ""
    return f"{x.model} {x.year or ''}{where}".replace("  ", " ").strip()


class Reply(BaseModel):
    to: Literal["buyer", "seller"]
    text: str


class TemplateResponder:
    name = "templates"

    def write(self, state: State, executed: list[tuple[Action, object]], new_risks: set[str],
              intents: Optional[list[str]] = None, news: Optional[list[dict]] = None,
              sent: Optional[dict] = None, pinned: Optional[list[dict]] = None,
              message: Optional[dict] = None) -> list[Reply]:
        intents = intents or []                  # pinned, message: only the LLM responder reads them
        sent = sent or {}
        buyer: list[str] = []
        seller: list[str] = []
        sel = state.listing_context.selected
        names = {a.name for a, _ in executed}
        fields = {a.args.get("field") for a, _ in executed if a.name == "ask_clarification"}
        budget = state.cval("budget_max")
        room = state.participants.get("seller_id")
        seller_here = bool(sel and sel.seller_id and room and sel.seller_id == room)

        for n in news or []:
            # the seller lowered the price: tell the buyer now, not when they ask again
            if n["type"] == "seller_floor" and not ("accept_floor" in fields and sel and sel.listing_id == n["listing_id"]):
                line = f"Người bán xe {n['bike']} đồng ý giảm còn {vnd(n['floor'])}"
                if budget and n["floor"] > budget:
                    line += f", vẫn cao hơn ngân sách {vnd(budget)} của bạn khoảng {vnd(n['floor'] - budget)}"
                buyer.append(line + ".")
        for n in news or []:
            # "ok 3h": the other side agreed to a time / place; tell the side that proposed it
            if n["type"] == "appointment_agreed":
                what = f"giờ xem xe {_fmt_time(n['value'])}" if n["slot"] == "time" else f"địa điểm {n['value']}"
                line = f"{'Người bán' if n['by'] == 'seller' else 'Người mua'} đồng ý {what} ạ."
                if n["by"] == "buyer" and "appointment_confirm" not in fields:
                    from .reducer import waiting_on
                    if "address" in waiting_on(state, "seller"):     # the one thing left to book: say it with the news
                        line = line[:-3] + ", anh/chị gửi giúp mình địa chỉ cụ thể (số nhà, tên đường) để chốt lịch nhé."
                (buyer if n["by"] == "seller" else seller).append(line)
        for n in news or []:
            # answer "cần trong tuần này" about the bike being discussed, not with the next question on the list
            if n["type"] == "urgency" and sel and sel.listing_id and n["value"] == "transfer_now":
                if sel.paperwork_status == "clean":
                    line = f"Dạ, xe {_bike(sel)} giấy tờ đầy đủ, chính chủ nên có thể sang tên ngay."
                else:
                    line = (f"Dạ, xe {_bike(sel)} giấy tờ {PAPER_TEXT.get(sel.paperwork_status, 'chưa rõ')}, "
                            "nên khó kịp sang tên trong thời gian bạn cần.")
                if any(q.field == "accept_floor" for q in state.open_questions) and "accept_floor" not in fields:
                    line += f" Người bán để thấp nhất {vnd(sel.seller_floor)}, bạn có muốn hẹn xem xe trong tuần này không ạ?"
                buyer.append(line)
        if "ASK_DISCOUNT" in intents and sel and state.negotiation.seller_stance == "firm" \
                and sel.seller_floor is None and "seller_floor_price" not in fields and "other_seller" not in fields:
            buyer.append(f"Người bán xe {_bike(sel)} đã báo giá {vnd(sel.ask_price)} là khó giảm thêm.")

        dis = state.risk(RiskType.DISINTERMEDIATION)
        if RiskType.DISINTERMEDIATION.value in new_risks and dis and len(dis.evidence) > 1:
            seller.append("Mình vẫn đề nghị hai bên trao đổi qua kênh chat của nền tảng; "
                          "số của người mua chỉ được chia sẻ khi người mua đồng ý ạ.")
        elif RiskType.DISINTERMEDIATION.value in new_risks:
            seller.append("Dạ mình hiểu ạ. Kênh chat trên nền tảng giúp hai bên hẹn xem xe và lưu lại thỏa thuận, "
                          "tránh rủi ro cho cả người bán lẫn người mua. Số điện thoại của người mua chỉ được chia sẻ khi người mua đồng ý.")
        if "SHARE_CONTACT" in intents:
            buyer.append("Để an toàn, bạn nên trao đổi và hẹn xem xe qua kênh chat của nền tảng, "
                         "và không chuyển tiền trước khi xem xe nhé.")
        if RiskType.DEPOSIT_REQUEST.value in new_risks and "escalate_to_human" not in names:
            buyer.append(DEPOSIT_WARNING)
        paper = state.risk(RiskType.PAPERWORK_PENDING)
        if paper and "PAPERWORK_CONCERN" in intents and "escalate_to_human" not in names:
            buyer.append("Rủi ro chính là xe chưa sang tên được ngay: nếu hồ sơ gốc bị chậm hoặc có vấn đề, "
                         "bạn sẽ chưa đứng tên xe. Bạn chỉ nên đặt cọc khi bộ phận hỗ trợ xác nhận giấy tờ đã xong.")
        if "ASK_INFO" in intents and state.risk(RiskType.DISINTERMEDIATION):
            # every capability named here is a tool the agent really has
            buyer.append("Bên mình có thể mở kênh chat với người bán, kiểm tra thông tin và giấy tờ xe, "
                         "và đặt lịch xem xe giúp bạn. Bạn vẫn có thể tự liên hệ nếu hai bên đồng ý trao đổi số điện thoại.")
        for a, result in executed:
            if a.name == "escalate_to_human":
                lines = dict.fromkeys(ESCALATE_TEXT.get(t, "Trường hợp này mình đã chuyển cho bộ phận hỗ trợ giao dịch kiểm tra.")
                                      for t in a.args.get("risk_types", []))
                buyer.append(" ".join(lines) + " Trong lúc chờ, bạn chưa nên đặt cọc hay chuyển tiền nhé.")
            elif a.name == "get_listing_detail" and isinstance(result, dict) and "listing_id" in result:
                buyer.append(f"Thông tin xe {result['model']} {result['year']}: odo {result['odo_km']/1000:g}k km, "
                             f"giá {vnd(result['price'])}, giấy tờ: {PAPER_TEXT.get(result['paperwork_status'], result['paperwork_status'])}.")
            elif a.name == "search_listings":
                cands = state.listing_context.candidates
                if not cands:
                    buyer.append(_empty_search_text(state, a.args))
                elif state.listing_context.candidates_unchanged:
                    # don't re-send the same list after a narrower search
                    buyer.append(f"Với tiêu chí mới, {len(cands)} xe ở trên vẫn là những xe phù hợp nhất ạ.")
                elif state.listing_context.candidates_narrowed:
                    names = ", ".join(f"{x.model} {x.year} ({vnd(x.ask_price)}, {x.district or x.location})" for x in cands)
                    buyer.append(f"Lọc theo ý bạn thì trong các xe vừa gửi còn {len(cands)} xe: {names}."
                                 + (" Bạn thích xe nào ạ?" if len(cands) > 1 else ""))
                else:
                    gap = state.negotiation.gap_pct or 0
                    room = state.participants.get("seller_id")
                    others = room and any(x.seller_id and x.seller_id != room for x in cands)
                    head = (f"Xe {sel.model} đang chênh khoảng {vnd(state.negotiation.gap_vnd)} so với ngân sách của bạn. "
                            "Bạn tham khảo thêm vài xe gần tầm giá" + (" của người bán khác" if others else "") + ":"
                            if gap > 0 and sel else f"Mình tìm được {len(cands)} xe phù hợp:")
                    buyer.append(head + "\n" + "\n".join(_listing_line(x, budget) for x in cands))
            elif a.name == "create_chat_bridge" and isinstance(result, dict) and result.get("channel_id"):
                if seller_here:     # the seller is already in this chat: no new channel to announce
                    seller.append(f"Người mua muốn hẹn xem xe {_bike(sel)}.")
                else:
                    buyer.append(f"Mình đã tạo kênh chat giữa bạn và người bán xe {sel.model} {sel.year}.")
            elif a.name == "book_appointment" and isinstance(result, dict):
                if result.get("status") == "booked":
                    buyer.append(f"Mình đã đặt lịch xem xe {sel.model}: {_fmt_time(result['time'])} tại {result['place']}. "
                                 "Khi xem xe nhớ đối chiếu giấy tờ với số khung, số máy nhé.")
                    if seller_here:
                        seller.append(f"Mình đã chốt lịch: người mua xem xe {sel.model} {sel.year} lúc {_fmt_time(result['time'])} "
                                      f"tại {result['place']}. Cảm ơn anh/chị ạ.")
                else:
                    buyer.append("Mình chưa đặt được lịch, bạn cho mình xin lại thời gian và địa điểm nhé.")
            elif a.name == "handoff_to_human":
                buyer.append("Mình chuyển cho nhân viên hỗ trợ để kết nối bạn với người bán theo đúng thỏa thuận.")
            elif a.name == "ask_clarification":
                field = a.args.get("field")
                to, text = QUESTIONS.get(field, ("buyer", "Bạn cho mình hỏi thêm một chút ạ?"))
                if field == "relay_question" and state.relay:
                    text = f"Người mua hỏi: \u201c{state.relay['text']}\u201d. Anh/chị trả lời giúp mình nhé?"
                    buyer.append("Mình đã hỏi người bán giúp bạn, có trả lời mình báo bạn ngay ạ.")
                if field in ("appointment_confirm", "appointment_ok"):
                    text, note = _appointment_ask(state, "seller" if field == "appointment_confirm" else "buyer")
                    other = "buyer" if field == "appointment_confirm" else "seller"
                    if not any(S.repeats(note, t) for t in sent.get(other, [])[-2:]):
                        (buyer if other == "buyer" else seller).append(note)
                    if state.asked_counts.get(field, 0) > 1:
                        text = "Nhắc lại giúp mình: " + text[0].lower() + text[1:]
                if field == "appointment_time" and seller_here:
                    text = "Bạn muốn xem xe vào ngày giờ nào ạ?"     # the seller gives the place
                if field in ("relay_offer", "accept_floor"):
                    text = text.format(offer=vnd(state.negotiation.buyer_offer),
                                       floor=vnd(sel.seller_floor if sel else None))
                again = state.asked_counts.get(field, 0) > 1      # counted before this reply is written
                if field == "seller_floor_price" and again:
                    text = "Người mua hỏi thêm: anh/chị có bớt được nữa không, mức thấp nhất là bao nhiêu ạ?"
                if field not in ("seller_floor_price", "other_seller", "appointment_confirm", "appointment_ok") and again:
                    text = "Mình hỏi lại một chút để tìm đúng xe cho bạn: " + text[0].lower() + text[1:]
                if field == "other_seller" and sel and again:
                    text = (f"Như mình nói ở trên, xe {_bike(sel)} là của người bán khác nên chưa hỏi giá trong cuộc trò chuyện này được. "
                            "Bạn nhắn \"kết nối\" khi muốn mình mở kênh chat với người bán xe đó nhé.")
                elif field == "other_seller" and sel:
                    text = (f"Xe {_bike(sel)} ({vnd(sel.ask_price)}) là của người bán khác, không tham gia cuộc trò chuyện này, "
                            "nên người bán ở đây không trả lời giá xe đó được.")
                    if any(r.severity == "high" and r.status != "resolved" for r in state.active_risks()):
                        text += " Xe đang chờ kiểm tra giấy tờ; khi giấy tờ được xác nhận, bạn nhắn \"kết nối\" để mình mở kênh chat với người bán xe đó nhé."
                    else:
                        text += " Bạn có muốn mình kết nối với người bán xe đó để hỏi giá không ạ?"
                if field == "seller_floor_price" and "ASK_DISCOUNT" in intents:
                    buyer.append(f"Mình đã hỏi người bán xe {_bike(sel)} mức giá thấp nhất giúp bạn." if sel
                                 else "Mình đã hỏi người bán mức giá thấp nhất giúp bạn.")
                if field == "listing_id":
                    opts = state.listing_context.seller_listing_options
                    text = text.format(options=(" Người bán đang có: " + "; ".join(opts) + ".") if opts else "")
                (buyer if to == "buyer" else seller).append(text)
                if field == "condition_history":
                    buyer.append("Tin đăng chưa ghi về lịch sử ngập nước, tai nạn hay phạt nguội, mình đã hỏi người bán giúp bạn. "
                                 "Khi xem xe bạn nên kiểm tra kỹ máy, khung sườn và tra cứu phạt nguội theo biển số.")
        room = state.participants.get("seller_id")
        elsewhere = sel and sel.seller_id and room and sel.seller_id != room
        if "ASK_DISCOUNT" in intents and elsewhere and "other_seller" not in fields and not buyer:
            # no other bike named here: the WRONG_BIKE check stays strict
            buyer.append("Giá xe này chỉ người bán của nó trả lời được. Bạn có thể nhắn \"kết nối\" để mình mở kênh chat "
                         "với họ, hoặc tiếp tục trao đổi xe của người bán ở đây ạ.")
        if sel and sel.listing_id and {"REFERENCE_LISTING", "SELECT_LISTING"} & set(intents) \
                and "get_listing_detail" not in names:
            # say which bike we are on, so a mix-up shows at once instead of three messages later
            line = (f"Dạ, mình đang trao đổi về xe {_bike(sel)}, giá {vnd(sel.ask_price)}"
                    + (f", giấy tờ {PAPER_TEXT.get(sel.paperwork_status, sel.paperwork_status)}" if sel.paperwork_status else "")
                    + ".")
            if not any(line in t for t in sent.get("buyer", [])[-2:]):
                buyer.insert(0, line)
            elif not buyer:
                buyer.append(f"Dạ đúng rồi, vẫn là xe {_bike(sel)} ạ.")
        # someone wrote while the viewing waits on the other side, and nothing else is due to them: say what we wait for
        # (instead of asking the other side the same question again, which the cooldown now stops)
        frm = (message or {}).get("from")
        if frm == "buyer" and not buyer and N.ASKED_YET_RE.search((message or {}).get("text") or ""):
            from .reply_check import SELLER_FIELDS
            last = max((state.asked_fields[f] for f in SELLER_FIELDS if f in state.asked_fields), default=None)
            if last is not None and state.msg_count - last <= 6:
                buyer.append("Dạ, người bán chưa trả lời ạ. Mình đã hỏi lúc nãy, có tin là mình báo bạn ngay.")
        if frm == "buyer" and not buyer and state.appointment.proposed_time is None and state.consent.buyer_bridge \
                and re.search(r"(?i)mấy\s*(h|giờ)\b|giờ nào", (message or {}).get("text") or ""):
            buyer.append("Dạ giờ xem xe chưa có ạ. Bạn muốn xem lúc mấy giờ để mình hỏi người bán?")    # "mấy h?"
        if frm in ("buyer", "seller") and not (buyer if frm == "buyer" else seller):
            line = _waiting_line(state, "seller" if frm == "buyer" else "buyer")
            if line:
                (buyer if frm == "buyer" else seller).append(line)
        # what each person must be told this turn; the LLM responder writes from these and is checked against them
        self.last_items = [{"to": "buyer", "text": x} for x in buyer] + [{"to": "seller", "text": x} for x in seller]
        if not buyer and not seller:
            buyer.append("Dạ mình đã ghi nhận ạ.")
        out = []
        if buyer:
            out.append(Reply(to="buyer", text=" ".join(buyer) if "\n" not in "".join(buyer) else "\n".join(buyer)))
        if seller:
            out.append(Reply(to="seller", text=" ".join(seller)))
        return out


def _empty_search_text(state: State, args: dict) -> str:
    """Say which filter is blocking, using the relaxation counts, instead of a generic "loosen something"."""
    hints = state.listing_context.empty_search_hints
    if not hints:
        return ("Hiện chưa có xe nào khớp hết tiêu chí của bạn. "
                "Bạn có thể nới bớt một tiêu chí (ngân sách, đời xe, hãng hoặc khu vực) để mình tìm lại không ạ?")
    lines = []
    for h in hints[:2]:
        f, n = h["field"], h["count"]
        if f == "location":
            lines.append(f"ở {args.get('location')} hiện chưa có xe phù hợp, nhưng ở khu vực {', '.join(h['locations'])} "
                         f"đang có {n} xe khớp các tiêu chí còn lại")
        elif f == "price_range":
            lines.append(f"nếu nới ngân sách thì có {n} xe phù hợp (từ {vnd(h['min_price'])} đến {vnd(h['max_price'])})")
        elif f == "year_range":
            lines.append(f"nếu lấy đời cũ hơn thì có {n} xe phù hợp (từ đời {h['min_year']})")
        elif f in ("brand", "model"):
            lines.append(f"nếu cân nhắc hãng hoặc mẫu khác thì có {n} xe phù hợp")
        elif f == "vehicle_type":
            lines.append(f"nếu đổi loại xe thì có {n} xe phù hợp")
    body = "; ".join(lines)
    return f"Hiện chưa có xe nào khớp hết tiêu chí của bạn: {body}. Bạn muốn mình tìm theo hướng nào ạ?"


def _appointment_ask(state: State, side: str) -> tuple[str, str]:
    """(question to `side`, note to the other side) about the parts of the viewing `side` still has to answer."""
    from .reducer import waiting_on
    ap, sel = state.appointment, state.listing_context.selected
    bike = f"{sel.model} {sel.year}" if sel and sel.model else ""     # no district: it reads like the place
    w = waiting_on(state, side)
    when = f" lúc {_fmt_time(ap.proposed_time)}" if ap.proposed_time else ""
    where = f" tại {ap.proposed_place}" if ap.proposed_place else ""
    if side == "seller":
        agree = [p for p in w if p == "time" or (p == "place" and ap.proposed_place)]
        need = ("địa chỉ cụ thể (số nhà, tên đường)" + (f" ở {ap.proposed_place}" if ap.proposed_place else "")
                if "address" in w else "địa điểm xem xe" if ap.proposed_place is None else "")
        if agree:
            q = f"Người mua muốn xem xe {bike}{when if 'time' in agree else ''}{where if 'place' in agree else ''}. " \
                "Anh/chị đồng ý không" + (f", và cho mình xin {need}" if need else "") + " ạ?"
        else:
            q = f"Anh/chị cho mình xin {need or 'địa điểm xem xe'} để người mua tìm đến{when} nhé?"
        asked = ([f"xác nhận lịch{when}"] if "time" in w else []) + ([need] if need else [])
        note = "Mình đã hỏi người bán " + " và ".join(asked) + ", chờ người bán trả lời ạ."
        return q, note
    parts = (when if "time" in w else "") + (where if "place" in w else "")
    q = f"Người bán đề nghị xem xe {bike}{parts}. Bạn đồng ý không ạ?"
    return q, "Mình đã hỏi người mua, chờ người mua xác nhận ạ."


def _waiting_line(state: State, other: str) -> Optional[str]:
    """What the viewing still waits on from `other`, when they were asked and have not written since."""
    from .reducer import waiting_on
    q = "appointment_confirm" if other == "seller" else "appointment_ok"
    w = waiting_on(state, other)
    if q not in state.asked_fields or not w:
        return None
    if other == "buyer":
        return "Dạ, mình đang chờ người mua xác nhận lịch xem xe ạ."
    ap = state.appointment
    when = f" lúc {_fmt_time(ap.proposed_time)}" if ap.proposed_time else ""
    parts = ([f"xác nhận giờ xem xe{when}"] if "time" in w else []) \
        + (["gửi địa chỉ cụ thể (số nhà, tên đường)"] if "address" in w else [])
    if "place" in w and "address" not in w:
        parts.append("báo địa điểm xem xe" if ap.proposed_place is None else f"đồng ý địa điểm {ap.proposed_place}")
    return "Dạ, mình đang chờ người bán " + " và ".join(parts) + " ạ, có trả lời mình báo bạn ngay."


def _fmt_time(iso: str) -> str:
    try:
        from datetime import datetime
        d = datetime.fromisoformat(iso)
        return d.strftime("%Hh%M ngày %d/%m/%Y")
    except Exception:
        return iso


# --------------------------------------------------------------------------- #
def guard(replies: list[Reply], executed_names: set[str]) -> list[str]:
    """Return a list of violations (empty = safe to send)."""
    problems = []
    for r in replies:
        if PHONE_RE.search(r.text) or pii.has_contact(r.text):
            problems.append(f"contact details in reply to {r.to}")
        f = N.fold(r.text)
        for phrase, needs in PROMISES.items():
            if N.fold(phrase) in f and not (needs & executed_names):
                problems.append(f"promise '{phrase}' without action {sorted(needs)}")
    return problems


class Replies(BaseModel):
    replies: list[Reply]


RESPOND_SYSTEM = """You write the marketplace agent's chat replies in natural, polite Vietnamese (short, like a real chat).
Use ONLY facts in FACTS and TOOL_RESULTS. Do not promise any action that is not in EXECUTED_ACTIONS.
Never include phone numbers, Zalo IDs, bank accounts or emails. One message per recipient at most. Address questions to whoever they are for.
MESSAGE_NOW is the message you are replying to. If it asks something, answer that first, from FACTS, KEY_MESSAGES or
TOOL_RESULTS; if none of them has the answer, say plainly that you don't have that detail yet (never guess).
KEY_MESSAGES are the original words behind facts that are still live (a seller's paperwork note, a price, the bikes
shown): when you state such a fact, keep to what they say and never add details (dates, amounts) they do not contain.
MUST_SAY lists what code decided each person hears this turn ("to": buyer / seller): every question, news line, warning
and booking. Say every item to that person, in your own natural words, in one message per person; keep each item's
bike, numbers, times and addresses exactly, and who is asked what. Do not add questions, requests or amounts that are
not in MUST_SAY, FACTS or TOOL_RESULTS. If MUST_SAY is empty, just answer MESSAGE_NOW briefly.
If PROBLEMS_WITH_YOUR_LAST_DRAFT is given, rewrite YOUR_LAST_DRAFT so that none of those problems remain.
Never say a bike from another seller can be negotiated with the seller in this chat.
Say you asked the seller something only when EXECUTED_ACTIONS has an ask_clarification to the seller this turn or
ALREADY_SENT shows that question to the seller; otherwise say you have not asked yet.
Say a viewing is booked or confirmed only when EXECUTED_ACTIONS has book_appointment; never say a side agreed to a
time or place unless FACTS says it did.
Never tell the buyer about an internal step you could not take ("chưa có kênh chat", a tool); say what happens next.
ALREADY_SENT holds your last replies to each person: do not repeat facts or sentences from them; if the draft repeats
something already said, say it in a few words ("vẫn là xe đó ạ") and move on to what is new."""


class LLMResponder:
    """Writes the replies from MUST_SAY (what code decided each person hears this turn), then code checks them
    (agent/reply_check.py). One retry with the problems listed; after that the template replies are sent."""
    name = "llm"

    def __init__(self, llm: StructuredLLM):
        self.llm = llm
        self.templates = TemplateResponder()
        self.guard_failures = 0      # turns that fell back to the templates
        self.retries = 0             # turns where the first draft failed a check
        self.last_problems: list[str] = []

    def write(self, state: State, executed: list[tuple[Action, object]], new_risks: set[str],
              intents: Optional[list[str]] = None, news: Optional[list[dict]] = None,
              sent: Optional[dict] = None, pinned: Optional[list[dict]] = None,
              message: Optional[dict] = None) -> list[Reply]:
        from . import reply_check as RC
        draft = self.templates.write(state, executed, new_risks, intents, news, sent)
        items = self.templates.last_items
        names = {a.name for a, _ in executed}
        results = [r for _, r in executed]
        payload = {
            "MESSAGE_NOW": message or {},
            "FACTS": state.summary,
            "KEY_MESSAGES": pinned or [],
            "NEWS_THIS_TURN": news or [],
            "ALREADY_SENT": {k: v[-3:] for k, v in (sent or {}).items()},
            "EXECUTED_ACTIONS": [a.model_dump() for a, _ in executed],
            "TOOL_RESULTS": results,
            "MUST_SAY": items,
        }
        problems: list[str] = []
        for attempt in range(2):
            if problems:
                payload["PROBLEMS_WITH_YOUR_LAST_DRAFT"] = problems
            try:
                out: Replies = self.llm.parse(RESPOND_SYSTEM, json.dumps(payload, ensure_ascii=False, default=str), Replies)
            except LLMUnavailable:
                return draft
            problems = guard(out.replies, names) + \
                [f"names a bike that is not under discussion: {b}" for b in
                 S.wrong_bike([r.text for r in out.replies if r.to == "buyer"], state, S.allowed_bikes(state, names, news))] + \
                RC.check(out.replies, items, state, [a for a, _ in executed], results, (message or {}).get("text"), sent)
            if not problems:
                self.last_problems = []
                return out.replies
            if attempt == 0:
                self.retries += 1
                payload["YOUR_LAST_DRAFT"] = [r.model_dump() for r in out.replies]
        self.guard_failures += 1               # the template says the right thing by construction
        self.last_problems = problems
        return draft
