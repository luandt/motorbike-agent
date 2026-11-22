"""One test per row of the README error table (written before the fixes)."""
from datetime import datetime, timezone
from pathlib import Path

from agent import normalize as N
from agent.extractor import RuleExtractor
from agent.memory import Store
from agent.models import Message, RiskType, Sender
from agent.pipeline import Agent, load_messages
from agent.reducer import apply_extraction, new_state

ROOT = Path(__file__).resolve().parent.parent


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


def replay(conv, path="data/heldout.jsonl"):
    store = Store(":memory:")
    agent = Agent(store, mode="rules")
    turns = [agent.handle(m) for m in load_messages([ROOT / path], conv)]
    return store.load_state(conv), turns


# row 1: model slang
def test_ab_alias_is_air_blade():
    assert [m[1] for m in N.find_models("AB 2020 của mình")] == ["Air Blade"]


# row 1b: seller with a single listing is resolved even without a model name
def test_v1_listing_resolved():
    st, _ = replay("v1")
    assert st.listing_context.selected.listing_id == "L009"


# row 2: "27tr5"
def test_tr_with_decimal_suffix():
    assert N.parse_money("27tr5")[0].high == 27_500_000
    assert N.parse_money("27tr500")[0].high == 27_500_000


# row 3: bare number in a seller price clause
def test_bare_number_floor():
    st = run([("seller", "Mình có Air Blade 2020 giá 27tr5"), ("seller", "26 là thấp nhất rồi em")])
    assert st.listing_context.selected.seller_floor == 26_000_000


# row 4: counter-offer is not a budget
def test_counter_offer_not_budget():
    st = run([("buyer", "tài chính tầm 30 củ"), ("buyer", "Bớt chút được không anh, 25tr được không?")])
    assert st.constraints.budget_max.value == 30_000_000
    assert st.negotiation.buyer_offer == 25_000_000


# row 5: deposit amount is not an ask price
def test_deposit_not_ask_price():
    st = run([("seller", "Vision 2020 giá 21tr"), ("seller", "Bạn chuyển cọc trước 2 triệu mình giữ xe cho nhé")])
    assert st.listing_context.selected.ask_price == 21_000_000
    assert st.risk(RiskType.DEPOSIT_REQUEST) is not None


def test_implausible_ask_price_rejected():
    st = run([("seller", "Vision 2020 giá 21tr"), ("seller", "giá 2tr")])
    assert st.listing_context.selected.ask_price == 21_000_000


# row 6: urgency without paperwork words
def test_urgency_can_wait():
    st = run([("buyer", "Em không gấp, đợi được. Giá có bớt không anh?")])
    assert st.constraints.urgency.value == "can_wait"


# row 7: consent closes the PII request
def test_v3_pii_resolved_after_consent():
    st, _ = replay("v3")
    assert st.risk(RiskType.PII_REQUEST).status == "resolved"


# row 10: a blocking question is not asked more than twice in a row
def test_c3_listing_question_capped():
    _, turns = replay("c3", "data/chat_history.jsonl")
    asks = sum(1 for t in turns for a in t.actions if a.args.get("field") == "listing_id")
    assert asks <= 2
