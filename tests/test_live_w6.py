"""Live web chat w6 (data/livechats.jsonl): a chat opened from the Vision 2020 listing (L002, papers pending).

What went wrong in the browser (LLM mode): "xe này bao lâu có giấy tờ" got "chưa có thông tin"; the budget "20t" was
dropped twice and asked again; paperwork questions were never passed to the seller; "hỏi chủ xe đi" went out as
"Người mua hỏi: “đi”"; the seller's answers ("chắc 2 tuần là có", "được") never reached the buyer, who was told
"chưa có cập nhật"; and "Mình đang trao đổi về xe Vision 2020..." came back on a message that changed nothing.
"""
from pathlib import Path

import pytest

from agent.memory import Store
from agent.pipeline import Agent, load_messages
from agent.responder import Replies, Reply
from tests.test_llm_first import FakeLLM, replay

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def turns():
    agent = Agent(Store(":memory:"), mode="rules", conversations_path=ROOT / "data/conversations.json")
    return [agent.handle(m) for m in load_messages([ROOT / "data/livechats.jsonl"], "w6")]


def to(t, who):
    return " ".join(r.text for r in t.replies if r.to == who)


def asked(t):
    return [a.args.get("field") for a in t.actions if a.name == "ask_clarification"]


def test_this_bike_is_the_listing_the_chat_was_opened_from(turns):
    t = turns[0]
    assert t.state.listing_context.selected.listing_id == "L002"
    assert "hồ sơ gốc" in to(t, "buyer") and "escalate_to_human" in [a.name for a in t.actions]


def test_the_budget_the_buyer_gave_is_kept_and_not_asked_again(turns):
    assert all(t.state.cval("budget_max") == 20_000_000 for t in turns[1:])
    assert not any("budget" in asked(t) for t in turns)
    assert "20tr" in to(turns[6], "buyer")                       # "đã nói ở trên rồi, 20t" is acknowledged


def test_a_paperwork_question_goes_to_the_seller_once(turns):
    assert "relay_question" in asked(turns[5]) and "giấy tờ" in to(turns[5], "seller")
    again = turns[7]                                             # the same question, still unanswered
    assert not to(again, "seller") and "chưa trả lời" in to(again, "buyer")
    assert not any("seller_floor_price" in asked(t) for t in turns[3:])     # the lowest price was asked once


def test_go_ask_the_seller_names_the_previous_question_not_di(turns):
    assert "“đi”" not in " ".join(to(t, "seller") for t in turns)
    assert "mai coi xe" in to(turns[12], "seller")               # "ý là đã hỏi mai coi xe trước được ko?"


def test_the_sellers_answers_reach_the_buyer(turns):
    assert "chắc 2 tuần là có" in to(turns[9], "buyer")
    assert "“được”" in to(turns[13], "buyer")
    for t in turns[14:]:                                         # "hỏi chưa?", "kết quả sao"
        assert "được" in to(t, "buyer") and "chưa trả lời" not in to(t, "buyer")
    assert to(turns[14], "buyer") != to(turns[15], "buyer")      # asked twice: not the same sentence twice


def test_the_bike_is_restated_only_when_it_changes_or_is_unclear(turns):
    assert not any("đang trao đổi về xe" in to(t, "buyer") for t in turns)


def test_a_held_viewing_is_explained(turns):
    assert "chưa đặt lịch xem xe" in to(turns[10], "buyer")


def test_in_llm_mode_the_sellers_answer_is_in_what_the_reply_must_say():
    """The LLM responder used to build MUST_SAY without the incoming message, so no line that depends on who wrote
    (the seller's answer, "đã hỏi chưa") ever reached the LLM."""
    llm = FakeLLM(reply=lambda u: Replies(replies=[Reply(to="buyer", text="Người bán trả lời: chắc 2 tuần là có.")]))
    agent, _ = replay("w6", 10, llm)
    assert any(it["to"] == "buyer" and "chắc 2 tuần là có" in it["text"] for it in agent.responder.templates.last_items)


def test_a_bare_amount_that_answers_the_budget_question_is_not_an_offer():
    """Paid run: the LLM read w6#2 "20t" as negotiation.buyer_offer, so the seller got "Người mua trả 20tr..." and a
    second price question. When the rules read the same amount as the budget, the LLM's offer is dropped."""
    from agent.extractor import LLMExtraction
    from tests.test_llm_first import fact
    llm = FakeLLM(extract=lambda u: LLMExtraction(intents=["OFFER_PRICE"], risks=[],
                                                  facts=[fact("negotiation.buyer_offer", 20_000_000, "20t")]))
    agent, t = replay("w6", 2, llm)
    assert t.state.cval("budget_max") == 20_000_000
    assert t.state.negotiation.buyer_offer is None and "OFFER_PRICE" not in [str(i) for i in t.extraction.intents]
    assert agent.extractor.overrides >= 1


def test_a_reply_that_passes_on_the_sellers_answer_must_use_their_words():
    """Paid run w6#12: "đã hỏi chủ xe chưa?" got "Dạ, vẫn là thông tin mình đã báo ở trên ạ" without "2 tuần"."""
    from agent import reply_check as R
    state = Agent(Store(":memory:"), mode="rules").handle(load_messages([ROOT / "data/livechats.jsonl"], "w6")[0]).state
    items = [{"to": "buyer", "text": "Dạ, như mình báo ở trên, người bán đã trả lời: “chắc 2 tuần là có”. Bạn cần mình hỏi người bán thêm gì không ạ?"}]
    vague = [Reply(to="buyer", text="Dạ, vẫn là thông tin mình đã báo ở trên ạ.")]
    good = [Reply(to="buyer", text="Dạ, người bán có trả lời là chắc 2 tuần là có giấy tờ ạ.")]
    assert any("chắc 2 tuần là có" in p for p in R.check(vague, items, state, [], [], "đã hỏi chủ xe chưa?"))
    assert not R.check(good, items, state, [], [], "đã hỏi chủ xe chưa?")
