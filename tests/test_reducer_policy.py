from datetime import datetime, timezone

from agent.extractor import RuleExtractor
from agent.models import Message, Risk, RiskType, Sender
from agent.policy import gates
from agent.reducer import apply_extraction, new_state


def msg(text, sender="buyer", i=1):
    return Message(conversation_id="t", msg_id=f"t#{i}", index=i, sender=Sender(sender), text=text,
                   timestamp=datetime(2026, 2, 5, tzinfo=timezone.utc))


def run(texts):
    st = new_state("t", {"buyer_id": "b", "seller_id": "s"})
    ex = RuleExtractor()
    for i, (sender, text) in enumerate(texts, 1):
        m = msg(text, sender, i)
        st, _ = apply_extraction(st, ex.extract(m, st), m)
    return st


def test_budget_is_superseded_with_history():
    st = run([("buyer", "tìm xe tầm 25tr"), ("buyer", "32tr cao quá, mình chỉ mua tối đa 25-26tr thôi")])
    b = st.constraints.budget_max
    assert b.value == 26_000_000 and b.source_msg == "t#2"
    assert b.history[0]["value"] == 25_000_000


def test_objected_price_is_not_a_budget():
    st = run([("buyer", "32tr cao quá")])
    assert st.constraints.budget_max is None


def test_price_gap_moves_stage_to_negotiation():
    st = run([("buyer", "tìm xe tay ga tầm 25tr ở HCM"), ("seller", "Mình có Air Blade 2021, odo 19k, giá 32tr")])
    assert st.negotiation.gap_pct == 0.28
    assert st.lead_stage.value == "NEGOTIATION"
    assert st.risk(RiskType.PRICE_GAP).status == "open"


def test_paperwork_risk_blocks_bridge_and_booking_even_when_escalated():
    st = run([("seller", "giấy tờ đang chờ rút hồ sơ gốc, chưa sang tên được")])
    allowed, _ = gates(st)
    assert "escalate_to_human" in allowed
    st.risk(RiskType.PAPERWORK_PENDING).status = "escalated"
    st.listing_context.selected.listing_id = "L002"
    st.listing_context.detail_fetched = ["L002"]
    st.consent.buyer_bridge = True
    allowed, blocked = gates(st)
    assert "create_chat_bridge" not in allowed and "PAPERWORK_PENDING" in blocked["create_chat_bridge"]
    assert "escalate_to_human" not in allowed


def test_share_contact_is_never_allowed():
    st = run([("seller", "chỉ cần số điện thoại người mua là được")])
    allowed, blocked = gates(st)
    assert "share_contact" in blocked and "handoff_to_human" not in allowed
    assert st.open_questions[0].field == "contact_consent"
