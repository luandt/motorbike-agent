from pathlib import Path

from agent.memory import Store
from agent.pipeline import Agent, load_messages
from agent.responder import PHONE_RE

ROOT = Path(__file__).resolve().parent.parent


def replay(conv, *files):
    store = Store(":memory:")
    agent = Agent(store, mode="rules")
    turns = [agent.handle(m) for m in load_messages([ROOT / f for f in files], conv)]
    return store, turns


def actions(turns):
    return [a.name for t in turns for a in t.actions]


def test_c1_offers_alternatives_instead_of_pushing():
    store, turns = replay("c1", "data/chat_history.jsonl")
    assert "search_listings" in actions(turns)
    assert "book_appointment" not in actions(turns)
    assert store.load_state("c1").lead_stage.value == "NEGOTIATION"


def test_c1_followup_reaches_booked_viewing():
    store, turns = replay("c1", "data/chat_history.jsonl", "data/c1_followup.jsonl")
    st = store.load_state("c1")
    assert st.appointment.status == "booked" and st.lead_stage.value == "APPOINTMENT"
    assert st.listing_context.selected.listing_id == "L005"


def test_c2_escalates_and_stores_listing_memory():
    store, turns = replay("c2", "data/chat_history.jsonl")
    assert "escalate_to_human" in actions(turns)
    assert store.listing_facts("L002")["paperwork_status"]["value"] == "pending_original_docs"
    assert any(e["type"] == "ESCALATION" for e in store.events("c2"))


def test_c3_never_shares_contact_and_asks_consent():
    store, turns = replay("c3", "data/chat_history.jsonl")
    assert "handoff_to_human" not in actions(turns)
    assert any(a.args.get("field") == "contact_consent" for t in turns for a in t.actions)


def test_phone_numbers_are_redacted_from_log_and_replies():
    store, turns = replay("v3", "data/heldout.jsonl")
    assert not any(PHONE_RE.search(e["payload"].get("text", "")) for e in store.events("v3"))
    assert not any(PHONE_RE.search(r.text) for t in turns for r in t.replies)
    assert "handoff_to_human" in actions(turns)


def test_every_event_type_is_logged():
    store, _ = replay("c2", "data/chat_history.jsonl")
    types = {e["type"] for e in store.events("c2")}
    assert {"USER_MESSAGE", "AGENT_ACTION", "TOOL_CALL", "TOOL_RESULT", "STATE_UPDATE", "ESCALATION"} <= types
