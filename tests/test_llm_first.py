"""LLM-first mode: the LLM reads and writes, code checks and decides. Fake LLMs, so these run free and offline."""
import json
from pathlib import Path

import pytest

from agent import llm as L
from agent.extractor import HybridExtractor, LLMExtraction, LLMFact
from agent.memory import Store
from agent.models import Intent
from agent.pipeline import Agent, load_messages
from agent.planner import PlannedAction, PlannerOutput
from agent.responder import Replies, Reply

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "data/livechats.jsonl"


class FakeLLM(L.StructuredLLM):
    """Answers each schema with a function; anything without one is 'unavailable' (code falls back to rules)."""
    provider, model = "fake", "llm-first"

    def __init__(self, extract=None, plan=None, reply=None):
        super().__init__()
        self.handlers = {LLMExtraction: extract, PlannerOutput: plan, Replies: reply}
        self.seen: dict = {}

    def parse(self, system, user, schema):
        self.calls += 1
        self.seen.setdefault(schema.__name__, []).append(user)
        fn = self.handlers.get(schema)
        if fn is None:
            raise L.LLMUnavailable("use the rules for this step")
        return fn(user)


def replay(cid, n, llm=None):
    """Rules up to message n-1, then message n with `llm` (or rules)."""
    msgs = load_messages([LIVE], cid)
    rules = Agent(Store(":memory:"), mode="rules", conversations_path=ROOT / "data/conversations.json")
    for m in msgs[:n - 1]:
        rules.handle(m)
    agent = Agent(rules.store, mode="llm", llm=llm, conversations_path=ROOT / "data/conversations.json") if llm else rules
    return agent, agent.handle(msgs[n - 1])


def nothing(_):
    return LLMExtraction(intents=[], facts=[], risks=[])


def fact(field, value, raw=""):
    return LLMFact(field=field, value=str(value), confidence=0.9, raw=raw)


# ---------------------------------------------------------------- Phase 1: reading
def test_the_llm_decides_which_bike_and_the_rules_fill_in_when_it_names_none():
    """w5 "oki" to the offered Gò Vấp bike: the rules pick L002."""
    _, rules_turn = replay("w5", 3)
    assert rules_turn.state.listing_context.selected.listing_id == "L002"
    other = FakeLLM(extract=lambda u: LLMExtraction(intents=[], risks=[], facts=[fact("listing.select_listing", "L008")]))
    _, t = replay("w5", 3, other)
    assert t.state.listing_context.selected.listing_id == "L008"               # both read a bike: the LLM wins
    _, t = replay("w5", 3, FakeLLM(extract=nothing))
    assert t.state.listing_context.selected.listing_id == "L002"               # the LLM named none: the rules' pick


def test_a_meaning_only_the_rules_read_is_dropped():
    """w2#5 "chiều mai 4h": a viewing time the LLM did not read is not taken from the rules."""
    _, rules_turn = replay("w2", 5)
    assert rules_turn.state.appointment.proposed_time
    _, t = replay("w2", 5, FakeLLM(extract=nothing))
    assert not t.state.appointment.proposed_time


def test_the_rules_fixed_value_wins_when_both_read_a_field():
    llm = FakeLLM(extract=lambda u: LLMExtraction(intents=["SEARCH"], risks=[],
                                                  facts=[fact("constraints.condition_pref", "xe đẹp")]))
    _, t = replay("w2", 1, llm)
    assert t.state.cval("condition_pref") == "good_looking"


def test_the_regex_reads_the_number_when_both_find_the_same_field():
    llm = FakeLLM(extract=lambda u: LLMExtraction(intents=["SEARCH"], risks=[],
                                                  facts=[fact("constraints.budget_max", 26_000_000, "tầm 25tr")]))
    agent, t = replay("w2", 1, llm)
    assert t.state.cval("budget_max") == 25_000_000
    assert agent.extractor.number_disagreements == 1


def test_sharing_the_buyers_phone_needs_the_rules_to_agree():
    llm = FakeLLM(extract=lambda u: LLMExtraction(intents=[], risks=[],
                                                  facts=[fact("consent.buyer_contact_share", "true", "ok")]))
    agent, t = replay("w5", 2, llm)
    assert t.state.consent.buyer_contact_share is None and agent.extractor.overrides == 1


def test_the_extractor_sees_what_the_agent_just_asked_the_sender():
    llm = FakeLLM(extract=nothing)
    replay("w5", 3, llm)
    prompt = llm.seen["LLMExtraction"][0]
    assert "THE AGENT'S LAST MESSAGE TO THE BUYER" in prompt and "Gò Vấp" in prompt.split("LAST MESSAGE")[1]


# ---------------------------------------------------------------- Phase 3: planner
def test_code_adds_the_question_the_seller_is_waiting_on():
    """w2#5 "chiều mai 4h": an LLM planner that only asks something else still sends the seller the time."""
    def plan(user):
        qs = [q["field"] for q in json.loads(user)["open_questions"] if q["field"] != "appointment_confirm"]
        return PlannerOutput(actions=[PlannedAction(name="ask_clarification", question_field=qs[0], reason="x")]
                             if qs else [PlannedAction(name="respond_only", reason="x")])
    agent, t = replay("w2", 5, FakeLLM(plan=plan))
    assert "appointment_confirm" in [a.args.get("field") for a in t.actions]


# ---------------------------------------------------------------- Phase 2: replies
def reply_with(*drafts):
    """A responder that answers with each draft in turn (to check the retry)."""
    it = iter(drafts)
    return lambda user: Replies(replies=[Reply(to=to, text=text) for to, text in next(it)])


def test_a_dropped_seller_message_is_retried_then_kept():
    first = [("buyer", "Mình đã hỏi người bán xác nhận lịch lúc 16h ngày 23/11 ạ.")]                  # seller missing
    second = first + [("seller", "Người mua muốn xem Vision 2022 lúc 16h ngày 23/11, anh/chị đồng ý và cho mình xin địa chỉ nhé?")]
    agent, t = replay("w2", 5, FakeLLM(reply=reply_with(first, second)))
    assert agent.responder.retries == 1 and agent.responder.guard_failures == 0
    assert any(r.to == "seller" and "16h" in r.text for r in t.replies)


def test_a_reply_that_keeps_failing_falls_back_to_the_templates():
    bad = [("buyer", "Dạ mình đã chốt lịch rồi ạ.")]
    agent, t = replay("w2", 5, FakeLLM(reply=reply_with(bad, bad)))
    assert agent.responder.guard_failures == 1
    assert any(r.to == "seller" and "16h00" in r.text for r in t.replies)        # the template's seller question


def test_reply_checks():
    from agent import reply_check as RC
    agent, t = replay("w1", 1)                                                   # rules state after "tầm 20 củ"
    st = t.state
    items = [{"to": "buyer", "text": "Xe Vision 2020 giá 21tr ở Gò Vấp."}]
    ok = [Reply(to="buyer", text="Dạ có Vision 2020 ở Gò Vấp, giá 21 triệu ạ.")]
    assert RC.check(ok, items, st, [], [], None) == []
    invented = [Reply(to="buyer", text="Dạ Vision 2020 ở Gò Vấp 21 triệu, chủ xe có thể bớt còn 19 triệu ạ.")]
    assert any("19,000,000" in p for p in RC.check(invented, items, st, [], [], None))
    missing = [Reply(to="buyer", text="Dạ có một xe Vision ở Gò Vấp ạ.")]
    assert any("leaves out" in p for p in RC.check(missing, items, st, [], [], None))
    claim = [Reply(to="buyer", text="Dạ Vision 2020 21 triệu ở Gò Vấp, mình đã hỏi chủ xe rồi ạ.")]
    assert RC.check(claim, items, st, [], [], None) == []                       # the seller was asked at w1#1
    st.msg_count += 4                                                            # ...but that was long ago
    assert any("seller was asked" in p for p in RC.check(claim, items, st, [], [], None))


# ---------------------------------------------------------------- Phase 4: record / replay
def test_record_then_replay_gives_the_same_turns(tmp_path):
    path = tmp_path / "rec.json"
    rec = L.CachedLLM(FakeLLM(extract=nothing), path, "record")
    _, t1 = replay("w2", 2, rec)
    rec.save()
    rep = L.CachedLLM(None, path, "replay")
    _, t2 = replay("w2", 2, rep)
    assert rep.hits > 0 and rep.misses == 0
    assert [a.name for a in t1.actions] == [a.name for a in t2.actions]
    with pytest.raises(L.LLMUnavailable):
        rep.parse("other system", "other prompt", LLMExtraction)


def test_a_listed_bike_price_the_llm_reads_as_a_budget_is_dropped():
    """w5#4 "đang hỏi xe ở gò vấp 21 triệu": 21tr is the Gò Vấp bike's price; the budget stays 20tr."""
    llm = FakeLLM(extract=lambda u: LLMExtraction(intents=[], risks=[],
                                                  facts=[fact("constraints.budget_max", 21_000_000, "21 triệu")]))
    agent, t = replay("w5", 4, llm)
    assert t.state.cval("budget_max") == 20_000_000 and agent.extractor.overrides == 1


# ---------------------------------------------------------------- repeats (Phase 5 live runs)
def test_the_llm_planner_does_not_ask_the_seller_again_before_the_seller_answers():
    """w1#6 "đã hỏi thử chủ xe đó chưa?": the floor price went to the seller at w1#4 with no answer since."""
    plan = lambda u: PlannerOutput(actions=[PlannedAction(name="ask_clarification", question_field="seller_floor_price",
                                                          reason="x")])
    _, t = replay("w1", 6, FakeLLM(plan=plan))
    assert "seller_floor_price" not in [a.args.get("field") for a in t.actions]


def test_asking_whether_the_seller_was_asked_is_not_a_question_to_pass_on():
    _, t = replay("w1", 6)
    assert Intent.ASK_SELLER not in t.extraction.intents and t.state.relay["msg"] == "w1#5"


def test_one_side_agreeing_does_not_ask_the_other_side_again():
    """w3#11 seller "ok 3h": the buyer was asked about Q1 at w3#10 and has not answered; not asked again."""
    _, t = replay("w3", 11)
    assert "appointment_ok" not in [a.args.get("field") for a in t.actions]
    assert not any(r.to == "buyer" and "bạn đồng ý không" in r.text for r in t.replies)


def test_a_reply_that_repeats_an_earlier_message_is_sent_back():
    from agent import reply_check as RC
    _, t = replay("w1", 1)
    said = "Dạ, mình đã hỏi người bán mức giá thấp nhất của xe Vision 2022 giúp bạn ạ."
    again = [Reply(to="buyer", text=said)]
    assert any("repeats" in p for p in RC.check(again, [], t.state, [], [], None, {"buyer": [said]}))
    assert not any("repeats" in p for p in RC.check(again, [], t.state, [], [], None, {"buyer": ["Chào bạn"]}))
