"""Pinned anchors: the messages behind facts that are still live stay in the prompt after they leave the
recent-message window, so an LLM step reads the seller's own words, not only a label in state."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent import llm as L
from agent import memory as M
from agent.extractor import LLMExtraction
from agent.memory import Store, build_context
from agent.models import Message, Sender
from agent.pipeline import Agent, load_messages
from agent.planner import PlannerOutput
from agent.responder import Replies, Reply

ROOT = Path(__file__).resolve().parent.parent
PAPERWORK = "giấy tờ đang chờ rút hồ sơ gốc"


def long_c2(agent, filler=30):
    """c2, then small talk until the seller's paperwork message is far outside the window."""
    msgs = load_messages([ROOT / "data/chat_history.jsonl"], "c2")
    t = None
    for m in msgs:
        t = agent.handle(m)
    last = msgs[-1]
    for i in range(1, filler + 1):
        sender = Sender.BUYER if i % 2 else Sender.SELLER
        t = agent.handle(Message(conversation_id="c2", msg_id=f"c2#{last.index + i}", index=last.index + i,
                                 sender=sender, text="dạ ok anh" if i % 2 else "ok em",
                                 timestamp=last.timestamp + timedelta(minutes=i)))
    return t


def pins(state, store):
    return build_context(state, store)["pinned_messages"]


def test_paperwork_note_stays_in_context_after_the_window_moves_on():
    agent = Agent(Store(":memory:"), mode="rules")
    t = long_c2(agent)
    ctx = build_context(t.state, agent.store)
    assert not any(PAPERWORK in m["text"] for m in ctx["recent_messages"])        # out of the window
    pinned = [m for m in ctx["pinned_messages"] if PAPERWORK in m["text"]]
    assert pinned and pinned[0]["id"] == "c2#2" and pinned[0]["from"] == "seller"
    assert pinned[0]["why"].startswith("risk PAPERWORK_PENDING")


def test_no_duplicates_while_the_window_still_shows_them():
    agent = Agent(Store(":memory:"), mode="rules")
    for m in load_messages([ROOT / "data/chat_history.jsonl"], "c2"):
        t = agent.handle(m)
    assert pins(t.state, agent.store) == []                                       # short chat: all in the window


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setattr(M, "RECENT_K", 2)


def test_seller_discount_and_selection_are_pinned(small_window, tmp_path):
    """x1: the buyer picked another seller's Vision 2020; the seller in the chat said "bớt 500k" on their 2022."""
    agent = Agent(Store(":memory:"), mode="rules", conversations_path=ROOT / "data/conversations.json")
    x1 = load_messages([ROOT / "data/scenarios.jsonl"], "x1")
    for m in x1[:6]:
        t = agent.handle(m)
    got = {m.get("id"): m for m in pins(t.state, agent.store)}
    assert got["x1#3"]["why"].startswith("bike under discussion: Vision 2020 (L002)")   # "cũng okie…"
    assert got["x1#4"]["text"] == "bớt 500k thôi" and got["x1#4"]["why"] == "listing.seller_floor on L008"
    ids = [m.get("id") or m.get("at") for m in pins(t.state, agent.store)]
    assert ids == sorted(ids, key=lambda i: int(i.split("#")[1]))                       # oldest first


def test_bikes_shown_are_pinned_as_the_agent_reply(small_window):
    agent = Agent(Store(":memory:"), mode="rules")
    msgs = [("buyer", "Mình cần xe tay ga tầm 25tr ở HCM"), ("buyer", "dạ"), ("buyer", "ok"), ("buyer", "vâng")]
    for i, (s, text) in enumerate(msgs, 1):
        t = agent.handle(Message(conversation_id="p", msg_id=f"p#{i}", index=i, sender=Sender(s), text=text,
                                 timestamp=datetime(2026, 11, 22, 9, i, tzinfo=timezone.utc)))
    shown = next(m for m in pins(t.state, agent.store) if m["from"] == "agent")
    assert shown["at"] == "p#1" and shown["to"] == "buyer" and shown["why"].startswith("bikes shown")
    assert any(c.model in shown["text"] for c in t.state.listing_context.candidates)


def test_pins_follow_the_live_facts(small_window, tmp_path):
    """x2: the buyer switches back to the seller's own bike; the other bike's risk and the old negotiation drop out."""
    agent = Agent(Store(":memory:"), mode="rules", conversations_path=ROOT / "data/conversations.json")
    for m in load_messages([ROOT / "data/scenarios.jsonl"], "x2"):
        t = agent.handle(m)
    st = t.state
    assert st.listing_context.selected.listing_id == "L008"
    whys = [m["why"] for m in pins(st, agent.store)]
    assert not any("L002" in w for w in whys)
    for key in st.anchors:
        assert "@" not in key or key.split("@")[1] in {"L008"}, key


def test_pins_are_capped_and_cut(small_window, monkeypatch):
    monkeypatch.setattr(M, "MAX_PINS", 1)
    monkeypatch.setattr(M, "PIN_CHARS", 10)
    agent = Agent(Store(":memory:"), mode="rules")
    t = long_c2(agent, filler=4)
    got = pins(t.state, agent.store)
    assert len(got) == 1 and got[0]["text"].endswith("…") and len(got[0]["text"]) == 11


class RecordingLLM(L.StructuredLLM):
    """Rules for extraction and planning; records what the responder is shown."""
    provider, model = "fake", "recording"

    def __init__(self):
        self.responder_input = None

    def parse(self, system, user, schema):
        if schema is Replies:
            self.responder_input = json.loads(user)
            return Replies(replies=[Reply(to="buyer", text="Dạ, người bán cho biết giấy tờ đang chờ rút hồ sơ gốc ạ.")])
        if schema in (LLMExtraction, PlannerOutput):
            raise L.LLMUnavailable("rules")
        raise AssertionError(schema)


def test_llm_steps_see_the_pinned_messages():
    rules = Agent(Store(":memory:"), mode="rules")
    t = long_c2(rules)
    llm = RecordingLLM()
    agent = Agent(rules.store, mode="llm", llm=llm)
    agent.handle(Message(conversation_id="c2", msg_id="c2#99", index=99, sender=Sender.BUYER,
                         text="Vậy khoảng bao lâu thì sang tên được anh?",
                         timestamp=datetime(2026, 2, 6, 12, 0, tzinfo=timezone.utc)))
    key = llm.responder_input["KEY_MESSAGES"]
    assert any(PAPERWORK in m["text"] for m in key)


def test_long_chats_pin_the_early_seller_details():
    """eval/gold_long: at the late question, the seller's early message is outside the window but pinned."""
    from eval.run_eval import reply_check
    for cid, early in (("l1", "l1#2"), ("l2", "l2#2")):
        agent = Agent(Store(":memory:"), mode="rules")
        msgs = load_messages([ROOT / "data/longchats.jsonl"], cid)
        assert len(msgs) >= 40
        for m in msgs[:38]:
            t = agent.handle(m)
        ctx = build_context(t.state, agent.store)
        assert early not in {m.get("id") for m in ctx["recent_messages"]}
        assert early in {m.get("id") for m in ctx["pinned_messages"]}
    gold = {"reply_must_include": [["3 tuần", "ba tuần"]], "reply_must_not_include": ["2 tuần"]}
    assert reply_check(gold, ["Người bán nói khoảng ba tuần nữa ạ"]) == []
    assert reply_check(gold, ["Khoảng 2 tuần ạ"]) == ["missing any of ['3 tuần', 'ba tuần']", "says '2 tuần'"]


def test_llm_responder_sees_the_message_it_answers():
    """Long-chat LLM eval: without the buyer's question, the responder answered "Dạ ạ." to "bao lâu thì sang tên?"."""
    rules = Agent(Store(":memory:"), mode="rules")
    long_c2(rules, filler=2)
    llm = RecordingLLM()
    Agent(rules.store, mode="llm", llm=llm).handle(
        Message(conversation_id="c2", msg_id="c2#90", index=90, sender=Sender.BUYER,
                text="Vậy khoảng bao lâu thì sang tên được anh? Số em 0909123456",
                timestamp=datetime(2026, 2, 6, 12, 0, tzinfo=timezone.utc)))
    now = llm.responder_input["MESSAGE_NOW"]
    assert now["from"] == "buyer" and "bao lâu" in now["text"] and "0909123456" not in now["text"]


# --------------------------------------------------------------------------- #
# Live web chat (seller s_008 in the room): the buyer picked the Vision 2022, said "mai hoặc mốt thì đc",
# then "chiều mai 4h". The bridge stayed blocked ("buyer has not asked to view") and the seller was never asked.
def _room_chat(lines, seller="s_008"):
    agent = Agent(Store(":memory:"), mode="rules")
    agent.meta["v"] = {"buyer_id": "b_v", "seller_id": seller}
    out = []
    for i, (sender, text) in enumerate(lines, 1):
        out.append(agent.handle(Message(conversation_id="v", msg_id=f"v#{i}", index=i, sender=Sender(sender), text=text,
                                        timestamp=datetime(2026, 11, 22, 9, i, tzinfo=timezone.utc))))
    return out


VIEWING = [("buyer", "Chào bạn, mình tìm xe tay ga tầm 25tr ở HCM, xe đẹp chút"),
           ("buyer", "chiếc vision 2022, giá fix luôn hay có giảm thêm ko"),
           ("seller", "nếu mua nhanh thì sẽ để lại giá đó"),
           ("buyer", "mai hoặc mốt thì đc"),
           ("buyer", "chiều mai 4h")]


def _to(t, who):
    return " ".join(r.text for r in t.replies if r.to == who)


def test_a_day_for_the_picked_bike_opens_the_viewing():
    t = _room_chat(VIEWING[:4])[-1]
    assert t.state.consent.buyer_bridge and t.state.channel_id
    assert "ngày giờ" in _to(t, "buyer") and "kênh chat" not in _to(t, "buyer")


def test_the_seller_confirms_the_buyers_time_and_the_buyer_the_sellers_place():
    ts = _room_chat(VIEWING + [("seller", "ok em, xem xe ở 25 Phan Xích Long nhé"), ("buyer", "ok")])
    ask = ts[4]
    assert "16h00 ngày 23/11/2026" in _to(ask, "seller") and "địa điểm" in _to(ask, "seller")
    assert "hỏi người bán xác nhận" in _to(ask, "buyer")
    place = ts[5]                                   # the seller agreed to 4pm and named a place: the buyer decides
    assert place.state.appointment.status != "booked"
    assert "Người bán đồng ý giờ" in _to(place, "buyer") and "Phan Xích Long. Bạn đồng ý" in _to(place, "buyer")
    done = ts[6]
    assert done.state.appointment.status == "booked" and done.state.appointment.proposed_place == "25 Phan Xích Long"
    assert "đặt lịch" in _to(done, "buyer") and "chốt lịch" in _to(done, "seller")


def test_seller_ok_without_a_place_is_asked_for_the_place():
    ts = _room_chat(VIEWING + [("seller", "ok em")])
    assert ts[-1].state.appointment.status != "booked" and ts[-1].state.appointment.time_agreed
    assert "địa điểm" in _to(ts[-1], "seller")


# Live web chat (LLM mode, seller s_009): the seller's "hẹn khách ở quận 1 được ko" was booked with no hour and
# without the buyer; the buyer's later "3h chiều mai" was "confirmed" without asking the seller.
AIRBLADE = [("buyer", "Chào bạn, mình tìm xe tay ga tầm 27tr ở HCM, xe đẹp chút"), ("buyer", "chiếc AB 2020 đc đó"),
            ("buyer", "hỏi thử giảm thêm 500k đc"), ("seller", "mua nhanh thì giảm đc nha"), ("seller", "27t"),
            ("buyer", "oki"), ("buyer", "chiều mai được ko"), ("seller", "được, hẹn khách ở quận 1 được ko"),
            ("buyer", "mấy h?"), ("buyer", "hỏi 3h chiều mai đc ko"), ("seller", "ok 3h"), ("buyer", "ok quận 1 được"),
            ("seller", "12 Lê Lợi nha")]


def test_each_side_agrees_to_the_others_part_before_booking():
    ts = _room_chat(AIRBLADE, seller="s_009")
    by_turn = [t.state.appointment.status for t in ts]
    assert "booked" not in by_turn[:-1] and by_turn[-1] == "booked"
    assert "Bạn đồng ý" in _to(ts[7], "buyer")                      # the seller's place goes to the buyer
    assert "15h00 ngày 23/11/2026" in _to(ts[9], "seller")          # the buyer's hour goes to the seller
    assert ts[10].state.appointment.proposed_time == "2026-11-23T15:00:00+07:00"   # "ok 3h" = agreed, not 3am
    assert ts[11].state.appointment.place_agreed and ts[11].state.appointment.status != "booked"   # area only
    ap = ts[-1].state.appointment
    assert (ap.proposed_place, ap.time_agreed, ap.place_agreed) == ("12 Lê Lợi, Quận 1", True, True)


def test_a_day_without_an_hour_is_not_booked_and_a_change_unbooks():
    from agent.models import Fact, State
    from agent.reducer import apply_fact
    st, diffs = State(conversation_id="v"), []
    apply_fact(st, Fact(field="appointment.proposed_time", value="tomorrow afternoon", source_msg="v#7",
                        speaker="buyer"), 7, diffs)
    assert st.appointment.proposed_time is None and diffs[0]["new"].startswith("rejected")
    st.appointment.proposed_time, st.appointment.status, st.appointment.time_agreed = "2026-11-23T15:00:00+07:00", "booked", True
    apply_fact(st, Fact(field="appointment.proposed_time", value="2026-11-23T17:00:00+07:00", source_msg="v#9",
                        speaker="buyer"), 9, diffs)
    assert st.appointment.status == "proposed" and not st.appointment.time_agreed


def test_asking_for_a_recommendation_after_the_brand_question_searches():
    """Live web chat (seller s_009): "27t", "sg", "có xe nào giới thiệu" got the brand/year question again,
    and "sg" was stored as "SG" while listings say "HCM"."""
    ts = _room_chat([("buyer", "Mình đang tìm xe khoảng 27t"), ("buyer", "sg"), ("buyer", "có xe nào giới thiệu")],
                    seller="s_009")
    assert ts[1].state.cval("location") == "HCM"
    shown = {c.listing_id for c in ts[2].state.listing_context.candidates}
    assert "L009" in shown and "Air Blade 2020" in _to(ts[2], "buyer")


def test_llm_location_is_normalised():
    from agent.models import Fact, State
    from agent.reducer import apply_fact
    st = State(conversation_id="v")
    apply_fact(st, Fact(field="constraints.location", value="SG", source_msg="v#2", speaker="buyer"), 2, [])
    assert st.cval("location") == "HCM"


# Live web chat (LLM mode, seller s_009): the buyer agreed to "Q1" and asked for the exact address in the same
# message; it was booked "ở Q1". Then "hỏi chủ xe thử địa chỉ nào" sent nothing to the seller, and the agent
# said "đã hỏi chủ xe rồi".
ADDRESS = [("buyer", "Mình đang tìm xe khoảng 27t"), ("buyer", "sg"), ("buyer", "có xe nào giới thiệu"), ("buyer", "honda"),
           ("buyer", "chiếc AB có giảm thêm ko"), ("seller", "nếu mua sớm thì giảm 500k"),
           ("buyer", "được, mai sắp xếp coi xe luôn được ko"), ("buyer", "4h chiều mai được ko"),
           ("seller", "okie, hỏi thử Q1 được ko"), ("buyer", "okie, gởi địa chỉ cụ thể nha"),
           ("buyer", "hỏi chủ xe thử địa chỉ nào, đường nào"), ("seller", "12 Lê Lợi nha em")]


def test_an_area_is_not_booked_until_the_seller_gives_a_street_address():
    ts = _room_chat(ADDRESS, seller="s_009")
    assert ts[8].state.appointment.proposed_place == "Quận 1"
    assert ts[9].state.appointment.place_agreed and ts[9].state.appointment.status != "booked"
    assert "địa chỉ cụ thể" in _to(ts[8], "seller")                 # asked once, when the seller named Q1
    assert _to(ts[9], "seller").startswith("Người mua đồng ý địa điểm Quận 1")   # news, not the same question again
    assert "chờ người bán gửi địa chỉ cụ thể" in _to(ts[9], "buyer")
    ap = ts[-1].state.appointment
    assert ap.status == "booked" and ap.proposed_place == "12 Lê Lợi, Quận 1" and ap.place_agreed
    assert "12 Lê Lợi, Quận 1" in _to(ts[-1], "buyer")


def test_ask_the_seller_relays_the_buyers_question():
    ts = _room_chat(ADDRESS[:11], seller="s_009")
    assert "Người mua hỏi: \u201cđịa chỉ nào, đường nào\u201d" in _to(ts[-1], "seller")
    assert "hỏi người bán giúp bạn" in _to(ts[-1], "buyer")


def test_exact_place():
    from agent.normalize import exact_place
    assert not any(exact_place(p) for p in ["Q1", "Quận 1", "Phú Nhuận", None])
    assert all(exact_place(p) for p in ["12 Lê Lợi", "quán cafe Highlands", "hẻm 45 Trần Phú"])
