"""P0 failures found by unseen probe chats (written before the fixes).

p2: seller posts their own phone / Zalo  -> disintermediation, not a request for the buyer's number
p2: spelled-out numbers, Zalo IDs, bank accounts -> redacted in the raw log
p3: buyer asks about flood damage / accidents / fines -> asked to the seller, never answered by omission
p4: buyer offers a deposit by bank transfer -> not a price offer, warned
p4: seller asks for a deposit to a bank account -> high risk, escalated
"""
import json
from datetime import datetime, timezone

import pytest

from agent import pii
from agent.extractor import RuleExtractor
from agent.memory import Store
from agent.models import Message, RiskType, Sender
from agent.pipeline import Agent
from agent.reducer import apply_extraction, new_state


def msg(text, sender="buyer", i=1, conv="t"):
    return Message(conversation_id=conv, msg_id=f"{conv}#{i}", index=i, sender=Sender(sender), text=text,
                   timestamp=datetime(2026, 3, 1, 8, i, tzinfo=timezone.utc))


def run(texts):
    st = new_state("t", {"buyer_id": "b", "seller_id": "s"})
    ex = RuleExtractor()
    for i, (sender, text) in enumerate(texts, 1):
        m = msg(text, sender, i)
        st, _ = apply_extraction(st, ex.extract(m, st), m)
    return st


@pytest.fixture
def agent(tmp_path):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps({"t": {"buyer_id": "b_9", "seller_id": "s_001"},
                                "v": {"buyer_id": "b_8", "seller_id": "s_002"},
                                "x": {"buyer_id": "b_7", "seller_id": "s_003"}}))
    store = Store(":memory:")
    return Agent(store, mode="rules", conversations_path=conv), store


def play(agent, conv, texts):
    a, store = agent
    turns = [a.handle(msg(t, s, i, conv)) for i, (s, t) in enumerate(texts, 1)]
    return turns, store


def buyer_text(turn):
    return " ".join(r.text for r in turn.replies if r.to == "buyer")


def seller_text(turn):
    return " ".join(r.text for r in turn.replies if r.to == "seller")


# --------------------------------------------------------------------------- #
# p4: deposits
# --------------------------------------------------------------------------- #
def test_buyer_deposit_offer_is_not_a_price_offer():
    st = run([("seller", "Air Blade 2021 giá 32 triệu"),
              ("buyer", "chuyển khoản cọc trước 5 triệu được không")])
    assert st.negotiation.buyer_offer is None
    assert st.risk(RiskType.DEPOSIT_REQUEST) is not None


def test_implausible_buyer_offer_rejected():
    st = run([("seller", "Air Blade 2021 giá 32 triệu"), ("buyer", "5 triệu được không anh")])
    assert st.negotiation.buyer_offer is None


def test_plausible_buyer_offer_kept():
    st = run([("seller", "Air Blade 2021 giá 32 triệu"), ("buyer", "30 triệu được không anh")])
    assert st.negotiation.buyer_offer == 30_000_000


def test_seller_bank_transfer_deposit_is_high_risk_and_escalated(agent):
    turns, store = play(agent, "t", [("seller", "Air Blade 2021 giá 32 triệu"),
                                     ("seller", "ok bớt còn 31 triệu, em ck cọc qua số tài khoản này nhé")])
    r = store.load_state("t").risk(RiskType.DEPOSIT_REQUEST)
    assert r is not None and r.severity == "high" and r.status == "escalated"
    last = turns[-1]
    assert "escalate_to_human" in [a.name for a in last.actions]
    text = buyer_text(last)
    assert "cọc" in text and "giấy tờ" not in text     # the warning names the real risk


def test_buyer_deposit_offer_gets_warning_not_relay(agent):
    turns, _ = play(agent, "t", [("seller", "Air Blade 2021 giá 32 triệu"),
                                 ("buyer", "chuyển khoản cọc trước 5 triệu được không")])
    last = turns[-1]
    assert "relay_offer" not in [a.args.get("field") for a in last.actions]
    assert "5tr" not in seller_text(last)
    assert "cọc" in buyer_text(last)


# --------------------------------------------------------------------------- #
# p2: seller shares their own contact
# --------------------------------------------------------------------------- #
def test_seller_sharing_own_number_is_disintermediation():
    st = run([("seller", "số mình 0909 123 456 gọi trực tiếp cho nhanh")])
    assert st.risk(RiskType.DISINTERMEDIATION) is not None
    assert st.risk(RiskType.PII_REQUEST) is None


def test_seller_sharing_zalo_is_disintermediation():
    st = run([("seller", "add zalo mình: không chín không chín một hai ba bốn năm sáu nhé, hoặc zalo id vinh.honda88")])
    assert st.risk(RiskType.DISINTERMEDIATION) is not None
    assert st.risk(RiskType.PII_REQUEST) is None


def test_seller_asking_for_buyer_number_is_still_pii_request():
    st = run([("seller", "Mình không cần bên thứ ba đứng giữa, chỉ cần số điện thoại người mua là được")])
    assert st.risk(RiskType.PII_REQUEST) is not None


def test_seller_contact_share_does_not_trigger_consent_question(agent):
    turns, _ = play(agent, "v", [("buyer", "Xe Vision này còn không anh"),
                                 ("seller", "số mình 0909 123 456 gọi trực tiếp cho nhanh")])
    last = turns[-1]
    assert "contact_consent" not in [a.args.get("field") for a in last.actions]
    assert seller_text(last)            # seller is told why the platform channel is used
    assert "0909" not in buyer_text(last) + seller_text(last)


# --------------------------------------------------------------------------- #
# p2: PII redaction in the raw log
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,leak", [
    ("số mình 0909 123 456", "0909"),
    ("gọi +84 909 123 456 nhé", "909 123"),
    ("không chín không chín một hai ba bốn năm sáu bảy", "không chín không chín"),
    ("zalo id vinh.honda88", "vinh.honda88"),
    ("zalo: vinhhonda", "vinhhonda"),
    ("stk 0123456789012 vietcombank", "0123456789012"),
    ("số tài khoản 1903 4567 8901 23", "1903 4567"),
    ("mail mình vinh88@gmail.com", "vinh88@gmail.com"),
])
def test_redact(text, leak):
    out, kinds = pii.redact(text)
    assert leak not in out and kinds


@pytest.mark.parametrize("text", ["AB 2020 của mình 27tr5, odo 27 ngàn", "Thứ 7 này 10h sáng ở Quận 3",
                                  "đời 2019 đổ lên, tầm 30 củ", "một hai ngày nữa mình xem"])
def test_redact_leaves_normal_text(text):
    assert pii.redact(text) == (text, [])


def test_event_log_has_no_contact_details(agent):
    _, store = play(agent, "v", [("seller", "add zalo mình: không chín không chín một hai ba bốn năm sáu nhé, "
                                            "hoặc zalo id vinh.honda88"),
                                 ("seller", "ck vào stk 0123456789012 nhé")])
    log = json.dumps(store.events("v"), ensure_ascii=False)
    for leak in ("vinh.honda88", "không chín không chín", "0123456789012"):
        assert leak not in log


# --------------------------------------------------------------------------- #
# p3: vehicle condition questions
# --------------------------------------------------------------------------- #
def test_condition_question_goes_to_seller(agent):
    turns, _ = play(agent, "x", [("buyer", "Exciter này có bị thủy kích hay tai nạn gì không? "
                                           "Biển số có bị phạt nguội không?")])
    t = turns[-1]
    assert "condition_history" in [a.args.get("field") for a in t.actions]
    assert seller_text(t)
    b = buyer_text(t)
    assert "người bán" in b and "kiểm tra" in b        # says it's asking, and what to check at viewing
