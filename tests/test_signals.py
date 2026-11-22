"""In-chat quality signals: would they have flagged the original wrong-bike chat, and do flagged
turns reach the error-analysis queue instead of the few-shot examples?"""
import json
from datetime import datetime, timezone
from pathlib import Path

from agent import llm as L
from agent import signals as S
from agent.extractor import LLMExtraction
from agent.feedback import export_flags, mine_fewshot, record_outcome
from agent.memory import Store
from agent.models import Message, Sender
from agent.pipeline import Agent, load_messages
from agent.planner import PlannerOutput
from agent.responder import Replies, Reply

ROOT = Path(__file__).resolve().parent.parent
X1 = load_messages([ROOT / "data/scenarios.jsonl"], "x1")


def state_after(n, agent=None):
    agent = agent or Agent(Store(":memory:"), mode="rules")
    res = [agent.handle(m) for m in X1[:n]]
    return agent, res[-1].state


def test_original_llm_replies_would_be_flagged():
    """Replies the agent really sent in the live chat, checked against the (correct) state of that turn."""
    _, st = state_after(5)                                     # buyer: "chủ có giảm ko", on the Vision 2020
    said = "Dạ xe Vision 2022 bên bán có thể giảm 500.000đ, còn 24 triệu ạ. Mức này vẫn cao hơn ngân sách 20 triệu."
    flags = S.detect(st, "buyer", X1[4].text, ["ASK_DISCOUNT"], [Reply(to="buyer", text=said)], [],
                     S.allowed_bikes(st, set(), []), "x1#4")
    assert "WRONG_BIKE" in [f["type"] for f in flags]

    ack = S.detect(st, "buyer", X1[4].text, ["ASK_DISCOUNT"], [Reply(to="buyer", text=S.EMPTY_ACK_TEXT)], [],
                   set(), "x1#4")
    assert [f["type"] for f in ack] == ["EMPTY_ACK"]


def test_buyer_correction_and_repeat_blame_the_previous_turn():
    hist = [{"from": "buyer", "text": "cũng okie, hỏi chủ có giảm thêm được ko"},
            {"from": "agent", "to": "buyer", "text": "Bạn có muốn cân nhắc xe khác không ạ?"}]
    _, st = state_after(3)
    rep = S.detect(st, "buyer", "chủ có giảm ko", [], [], hist, set(), "x1#3")
    cor = S.detect(st, "buyer", "đang nói vision 2020 ờ gò vấp", [], [], hist, set(), "x1#5")
    assert rep[0]["type"] == "BUYER_REPEAT" and rep[0]["blames"] == "x1#3"
    assert cor[0]["type"] == "BUYER_CORRECTION" and cor[0]["blames"] == "x1#5"
    again = S.detect(st, "buyer", "ok", [], [Reply(to="buyer", text=hist[1]["text"])], hist, set(), None)
    assert [f["type"] for f in again] == ["AGENT_REPEAT"]


def test_fixed_agent_has_no_agent_side_flags_on_the_live_chats():
    for cid in ("x1", "x2"):
        agent = Agent(Store(":memory:"), mode="rules")
        flags = [f for m in load_messages([ROOT / "data/scenarios.jsonl"], cid) for f in agent.handle(m).signals]
        assert not [f for f in flags if f["side"] == "agent"], flags


def test_flagged_turns_go_to_the_queue_not_to_fewshot(tmp_path):
    store = Store(str(tmp_path / "f.db"))
    agent = Agent(store, mode="rules")
    for m in X1:
        agent.handle(m)
    record_outcome(store, "x1", "booked")                       # even a booked lead can contain bad turns
    queue = export_flags(store)
    bad = {r["bad_turn"] for r in queue}
    assert {"x1#4", "x1#5", "x1#6"} <= bad                      # turns the buyer reacted to
    row = next(r for r in queue if r["bad_turn"] == "x1#5")
    assert row["replies"] and row["signals"][0]["type"] == "BUYER_CORRECTION"
    examples = mine_fewshot(store, tmp_path / "fewshot.json")
    assert not any(e["at_msg"] in bad for e in examples)
    negatives = json.loads((tmp_path / "negative_examples.json").read_text())
    assert any(n.get("source") == "quality_signal" for n in negatives)


class WrongBikeLLM(L.StructuredLLM):
    """Extracts nothing, plans nothing, and writes a reply about the wrong bike."""
    provider, model = "fake", "wrong-bike"

    def parse(self, system, user, schema):
        if schema is LLMExtraction:
            return LLMExtraction(intents=[], facts=[], risks=[])
        if schema is PlannerOutput:
            raise L.LLMUnavailable("use the rule planner")
        return Replies(replies=[Reply(to="buyer", text="Dạ xe Vision 2022 bên bán giảm còn 24 triệu ạ.")])


def test_llm_reply_about_the_wrong_bike_falls_back_to_the_template():
    rules, _ = state_after(4)
    agent = Agent(rules.store, mode="llm", llm=WrongBikeLLM())
    t = agent.handle(X1[4])                                     # "chủ có giảm ko" while on the Vision 2020
    assert agent.responder.guard_failures == 1
    assert "Vision 2022" not in " ".join(r.text for r in t.replies if r.to == "buyer")


class SellerTaggedAsBuyerLLM(L.StructuredLLM):
    """LLM-mode eval, c2#5: the seller's "nếu bạn cần gấp giấy tờ..." came back as the buyer's PAPERWORK_CONCERN."""
    provider, model = "fake", "wrong-side"

    def parse(self, system, user, schema):
        if schema is LLMExtraction:
            return LLMExtraction(intents=["PAPERWORK_CONCERN", "FIRM_PRICE"], facts=[], risks=[])
        raise L.LLMUnavailable("rules for the rest")


def test_llm_intents_from_the_wrong_side_are_dropped():
    from agent.extractor import LLMExtractor
    from agent.models import State
    ex = LLMExtractor(SellerTaggedAsBuyerLLM()).extract(
        Message(conversation_id="w", msg_id="w#1", index=1, sender=Sender.SELLER,
                text="Nếu bạn cần gấp giấy tờ sang tên liền thì xe này chưa phù hợp đâu",
                timestamp=datetime(2026, 2, 6, tzinfo=timezone.utc)), State(conversation_id="w"))
    assert [i.value for i in ex.intents] == ["FIRM_PRICE"]


def test_high_odo_risk_belongs_to_its_bike():
    """LLM-mode eval, c1_followup: HIGH_ODO from an earlier bike stayed open after switching to the Lead."""
    from agent.models import ListingRef, State
    from agent.reducer import recompute
    st = State(conversation_id="o")
    st.listing_context.selected = ListingRef(listing_id="L999", model="Vision", year=2024, odo_km=60_000)
    msg = Message(conversation_id="o", msg_id="o#1", index=1, sender=Sender.BUYER, text="",
                  timestamp=datetime(2026, 2, 6, tzinfo=timezone.utc))
    recompute(st, msg)
    odo = next(r for r in st.risks if r.type.value == "HIGH_ODO")
    assert odo.listing_id == "L999"
    st.listing_context.selected = ListingRef(listing_id="L005", model="Lead", year=2021, odo_km=12_000)
    assert st.risk(odo.type) is None


class DropLLM(L.StructuredLLM):
    """Long-chat LLM eval, l2#40: "Ok em lấy giá đó, em muốn xem xe trực tiếp" came back as DROP."""
    provider, model = "fake", "drop"

    def parse(self, system, user, schema):
        if schema is LLMExtraction:
            return LLMExtraction(intents=["DROP", "REQUEST_VIEWING"], facts=[], risks=[])
        raise L.LLMUnavailable("rules for the rest")


def _buyer(text, i=1):
    return Message(conversation_id="d", msg_id=f"d#{i}", index=i, sender=Sender.BUYER, text=text,
                   timestamp=datetime(2026, 3, 3, 9, i, tzinfo=timezone.utc))


def test_llm_drop_needs_a_withdrawal_word():
    from agent.extractor import LLMExtractor
    from agent.models import State
    ex = LLMExtractor(DropLLM())
    took = ex.extract(_buyer("Ok em lấy giá đó, em muốn xem xe trực tiếp"), State(conversation_id="d"))
    assert [i.value for i in took.intents] == ["REQUEST_VIEWING"] and ex.drops_rejected == 1
    quit_ = ex.extract(_buyer("Thôi em mua xe khác rồi anh", 2), State(conversation_id="d"))
    assert "DROP" in [i.value for i in quit_.intents]


def test_taking_the_price_does_not_close_the_lead():
    rules = Agent(Store(":memory:"), mode="rules", conversations_path=ROOT / "data/conversations.json")
    msgs = load_messages([ROOT / "data/longchats.jsonl"], "l2")
    for m in msgs[:-1]:
        rules.handle(m)
    t = Agent(rules.store, mode="llm", llm=DropLLM(),
              conversations_path=ROOT / "data/conversations.json").handle(msgs[-1])
    assert not t.state.handed_off and t.state.lead_stage.value != "DROPPED"
