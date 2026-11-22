"""Regression for a live chat where the agent negotiated the wrong bike.

Seller s_008 (Vision 2022, 24.5tr) is in the chat. The agent showed an alternative Vision 2020 in Gò Vấp
from ANOTHER seller; the buyer said "cũng okie, hỏi chủ có giảm" and the agent kept talking about the 2022,
never passed on the seller's "bớt 500k", and repeated the "other bikes?" question.
"""
import json
from datetime import datetime, timezone

import pytest

from agent import normalize as N
from agent.memory import Store
from agent.models import Message, Sender
from agent.pipeline import Agent

CHAT = [
    ("buyer", "đang có xe vision nào giá tốt mà xe mới ko"),
    ("buyer", "khoản 20t"),
    ("buyer", "cũng okie, hỏi chủ có giảm thêm được ko"),
    ("seller", "bớt 500k thôi"),
    ("buyer", "chủ có giảm ko"),
    ("buyer", "đang nói vision 2020 ờ gò vấp"),
    ("buyer", "đang nói chiếc vision 2020, 21t mà"),
    ("buyer", "cũng được nhưng có giảm giá thêm được ko"),
]


def run(tmp_path, seller, chat):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps({"m": {"buyer_id": "b", "seller_id": seller}}))
    agent = Agent(Store(":memory:"), mode="rules", conversations_path=conv)
    return [agent.handle(Message(conversation_id="m", msg_id=f"m#{i}", index=i, sender=Sender(s), text=text,
                                 timestamp=datetime(2026, 11, 21, 9, i, tzinfo=timezone.utc)))
            for i, (s, text) in enumerate(chat, 1)]


def to(turn, who):
    return " ".join(r.text for r in turn.replies if r.to == who)


def asked(turn):
    return [a.args.get("field") for a in turn.actions if a.name == "ask_clarification"]


@pytest.fixture
def turns(tmp_path):
    return run(tmp_path, "s_008", CHAT)


def test_chat_shorthand_money():
    assert [m.low for m in N.parse_money("khoản 20t")] == [20_000_000]
    assert N.parse_discount("bớt 500k thôi") == 500_000
    assert N.parse_discount("giảm 1tr nha") == 1_000_000
    assert N.parse_discount("bớt còn 24tr") is None          # a price TO, handled as a floor
    assert N.parse_money("20 tuổi") == []


def test_budget_and_alternative_marked_as_other_seller(turns):
    st = turns[1].state
    assert st.cval("budget_max") == 20_000_000
    assert [c.listing_id for c in st.listing_context.candidates] == ["L002"]
    buyer = to(turns[1], "buyer")
    assert "người bán khác" in buyer and "giấy tờ đang chờ" in buyer and "cao hơn ngân sách 1tr" in buyer


def test_okie_picks_the_bike_just_shown_not_the_one_in_the_chat(turns):
    t = turns[2]
    assert t.state.listing_context.selected.listing_id == "L002"
    assert "seller_floor_price" not in asked(t)               # don't ask s_008 about s_002's bike
    assert "other_seller" in asked(t) and not to(t, "seller")
    assert "escalate_to_human" in [a.name for a in t.actions]  # L002 papers are pending


def test_seller_discount_is_relayed_and_lands_on_their_own_bike(turns):
    t = turns[3]
    prev = {p.listing_id: p for p in t.state.listing_context.previous}
    assert prev["L008"].seller_floor == 24_000_000
    assert t.state.listing_context.selected.seller_floor is None   # not written onto L002
    assert "Vision 2022 đồng ý giảm còn 24tr" in to(t, "buyer")


def test_follow_ups_stay_on_the_right_bike(turns):
    for t in turns[4:]:
        assert t.state.listing_context.selected.listing_id == "L002"
        assert "ghi nhận" not in to(t, "buyer")
        assert "Vision 2022" not in to(t, "buyer")
    assert "Vision 2020 ở Gò Vấp" in to(turns[5], "buyer")
    assert turns[6].state.cval("budget_max") == 20_000_000       # "21t mà" is the bike's price, not a budget


def test_alternatives_question_asked_once(turns):
    assert sum("choose_alternative" in asked(t) for t in turns) == 1


def test_discount_on_bike_in_this_chat_goes_to_its_seller(tmp_path):
    t = run(tmp_path, "s_008", [("buyer", "Vision này còn không anh, ngân sách 25tr"),
                                ("buyer", "có giảm thêm được không anh")])[-1]
    assert "seller_floor_price" in asked(t)
    assert "giá thấp nhất" in to(t, "seller") and "Mình đã hỏi người bán" in to(t, "buyer")


def test_yes_connects_to_the_other_seller(tmp_path):
    ts = run(tmp_path, "s_001", [("buyer", "Air Blade này còn không anh, tầm 26tr"),
                                 ("buyer", "cũng okie"),
                                 ("buyer", "có")])
    assert ts[1].state.listing_context.selected.listing_id == "L009"
    assert "kết nối với người bán xe đó" in to(ts[1], "buyer")
    bridge = next(a for a in ts[2].actions if a.name == "create_chat_bridge")
    assert bridge.args["seller_id"] == "s_009" and bridge.args["listing_id"] == "L009"


SWITCH_BACK = [
    ("buyer", "đang có xe vision nào giá tốt mà xe còn mới ko"),
    ("buyer", "khoản 20t"),
    ("buyer", "chủ vision 2020 có giảm thêm được ko"),
    ("buyer", "chiếc 2022 thì chủ có giảm thêm ko"),
    ("seller", "giảm 1t nữa là tối đa, ko bớt đc nữa"),
    ("buyer", "cần trong tuần này"),
]


@pytest.fixture
def back(tmp_path):
    return run(tmp_path, "s_008", SWITCH_BACK)


def test_paperwork_risk_stays_with_its_bike(back):
    st = back[3].state
    assert st.listing_context.selected.listing_id == "L008"
    paper = next(r for r in st.risks if r.type.value == "PAPERWORK_PENDING")
    assert paper.listing_id == "L002" and not st.applies(paper)
    assert "urgency" not in asked(back[3])                           # L008's papers are clean
    assert not any(a.name == "search_listings" and a.args["paperwork_clean_only"] for a in back[3].actions)
    assert "chờ" not in to(back[3], "buyer")


def test_seller_firm_floor_and_urgency_answer(back):
    assert back[4].state.listing_context.selected.seller_floor == 23_500_000
    assert back[4].state.negotiation.seller_stance == "firm"
    last = back[5]
    assert last.state.cval("urgency") == "transfer_now"
    assert "brands_year" not in asked(last) and "hãng nào" not in to(last, "buyer")   # bike already chosen
    assert "sang tên ngay" in to(last, "buyer") and "23.5tr" in to(last, "buyer")


def test_empty_search_hints_skip_bikes_already_discussed(tmp_path):
    t = run(tmp_path, "s_001", [("buyer", "Air Blade này còn không anh, tối đa 26tr")])[0]
    hints = t.state.listing_context.empty_search_hints
    price = next((h for h in hints if h["field"] == "price_range"), None)
    assert price is None or price["max_price"] < 32_000_000          # L001 itself is not an alternative


def test_a_wish_is_not_a_pick():
    """LLM-mode eval, c1#3: "đời 2020 trở lên" matched a candidate Grande 2020 and switched bikes."""
    from agent.extractor import match_listing
    from agent.models import ListingRef, State
    st = State(conversation_id="t", participants={"seller_id": "s_001"}, msg_count=2)
    st.listing_context.candidates = [
        ListingRef(listing_id="L007", model="Grande", year=2020, ask_price=27_000_000, district="Quận 7", seller_id="s_007"),
        ListingRef(listing_id="L002", model="Vision", year=2020, ask_price=21_000_000, district="Gò Vấp", seller_id="s_002")]
    st.listing_context.candidates_at = 1
    assert match_listing("Honda hoặc Yamaha, đời 2020 trở lên, odo thấp", st) is None
    assert match_listing("tầm 21tr thôi", st) is None
    assert match_listing("chủ vision 2020 có giảm thêm được ko", st) == "L002"
    assert match_listing("chiếc 2020 ở quận 7 thì sao", st) == "L007"


def test_narrowed_request_searches_again(tmp_path):
    """LLM-mode eval, v1#2: after an early search, "ưu tiên Honda, đời 2019" got "Dạ mình đã ghi nhận"."""
    ts = run(tmp_path, None, [("buyer", "Mình cần xe tay ga tầm 30 củ ở HCM, odo thấp, không gấp"),
                              ("buyer", "Ưu tiên Honda, đời 2019 đổ lên")])
    assert "search_listings" in [a.name for a in ts[0].actions]
    again = next(a for a in ts[1].actions if a.name == "search_listings")
    assert again.args["brand"] == ["Honda"] and again.args["year_range"] == [2019, None]
    assert "ghi nhận" not in to(ts[1], "buyer")


def test_search_needs_60_percent_coverage_for_llm_and_rules(tmp_path):
    """Decision: search once budget + location + type are known (60%). The gate enforces it, so an LLM
    planner cannot search earlier either; a too-expensive bike may still look for alternatives."""
    from agent.policy import gates
    low = run(tmp_path, None, [("buyer", "Mình cần xe tay ga tầm 25tr")])[0]
    allowed, blocked = gates(low.state)
    assert "search_listings" not in allowed and "40%" in blocked["search_listings"]
    ok = run(tmp_path, None, [("buyer", "Mình cần xe tay ga tầm 25tr ở HCM")])[0]
    assert "search_listings" in [a.name for a in ok.actions]
