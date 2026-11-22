"""P1 extraction gaps found by the probe chats (written before the fixes)."""
import json
from datetime import datetime, timezone

import pytest

from agent import normalize as N
from agent.extractor import RuleExtractor
from agent.memory import Store
from agent.models import Message, Sender
from agent.pipeline import Agent
from agent.reducer import apply_extraction, new_state


def msg(text, sender="buyer", i=1, conv="t"):
    return Message(conversation_id=conv, msg_id=f"{conv}#{i}", index=i, sender=Sender(sender), text=text,
                   timestamp=datetime(2026, 3, 1, 8, i, tzinfo=timezone.utc))


def run(texts):
    st = new_state("t", {"buyer_id": "b", "seller_id": None})
    ex = RuleExtractor()
    for i, (sender, text) in enumerate(texts, 1):
        m = msg(text, sender, i)
        st, _ = apply_extraction(st, ex.extract(m, st), m)
    return st


# --- money written as "30m" / "30 chai" -------------------------------------
@pytest.mark.parametrize("text,value", [("30m", 30_000_000), ("budget 30M thôi", 30_000_000),
                                        ("tầm 25 chai", 25_000_000), ("27.5m", 27_500_000)])
def test_money_units(text, value):
    assert N.parse_money(text)[0].high == value


def test_m_budget_in_context():
    st = run([("buyer", "budget mình 30m thôi, chuyển khoản cọc trước 5 triệu được không")])
    assert st.cval("budget_max") == 30_000_000


def test_m_is_not_metres_or_minutes():
    assert N.parse_money("cách 500m là tới") == []
    assert N.parse_money("đợi 5m nha") == []


# --- "SH" is its own model, "SH Mode" still wins when written in full --------
def test_sh_model():
    assert [m[1] for m in N.find_models("Mình cần SH tầm 80 củ")] == ["SH"]
    assert [m[1] for m in N.find_models("SH Mode 2021 giá 52tr")] == ["SH Mode"]
    assert [m[1] for m in N.find_models("shmode 2021")] == ["SH Mode"]


# --- negation on brands -------------------------------------------------------
def test_brand_negation_replaces():
    st = run([("buyer", "Ưu tiên Honda, đời 2019 đổ lên"),
              ("buyer", "không lấy Honda nữa, mình thích Yamaha hơn")])
    assert st.cval("brands") == ["Yamaha"]


def test_brand_negation_alone_removes():
    st = run([("buyer", "Honda hay Yamaha đều được"), ("buyer", "thôi bỏ Honda đi")])
    assert st.cval("brands") == ["Yamaha"]


def test_brand_without_negation_unchanged():
    st = run([("buyer", "không biết Honda hay Yamaha bền hơn nhỉ")])
    assert st.cval("brands") == ["Honda", "Yamaha"]


# --- zero results: name the constraint that blocks the search -----------------
@pytest.fixture
def agent(tmp_path):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps({"z": {"buyer_id": "b_9", "seller_id": None}}))
    return Agent(Store(":memory:"), mode="rules", conversations_path=conv)


def test_empty_search_names_location(agent):
    t = agent.handle(msg("Mình tìm Yamaha tay ga tầm 25 củ ở Hà Nội", conv="z"))
    assert "search_listings" in [a.name for a in t.actions]
    text = " ".join(r.text for r in t.replies)
    assert "khu vực" in text and "HCM" in text        # stock exists in HCM
    assert "nới thêm ngân sách hoặc đời xe" not in text   # budget and year were never the problem
