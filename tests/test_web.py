"""Browser demo server: two participants over WebSockets, agent replies routed to the right person."""
import pytest
from fastapi.testclient import TestClient

from web import server


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    server.Config.mode, server.Config.llm = "rules", None
    server.Config.db = tmp_path_factory.mktemp("web") / "web.db"
    with TestClient(server.app) as c:
        yield c


def until(ws, kind):
    """Read events until one of `kind` arrives; return it and everything seen before it."""
    seen = []
    while True:
        m = ws.receive_json()
        seen.append(m)
        if m["type"] == kind:
            return m, seen


def turn_done(ws):
    """Read one full agent turn (ends with busy off). Returns all events in it."""
    seen = []
    while True:
        m = ws.receive_json()
        seen.append(m)
        if m["type"] == "busy" and not m["on"]:
            return seen


def test_pages_and_lobby_api(client):
    assert "Motorbike marketplace agent" in client.get("/").text
    assert client.get("/chat").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/api/info").json()["mode"] == "rules"
    assert any(s["id"] == "s_001" for s in client.get("/api/sellers").json())


def test_two_party_chat(client):
    conv = client.post("/api/conversations", json={"seller": "s_001"}).json()["conv"]
    with client.websocket_connect(f"/ws/{conv}/buyer") as buyer, \
            client.websocket_connect(f"/ws/{conv}/seller") as seller:
        hello, _ = until(buyer, "hello")
        assert hello["role"] == "buyer" and hello["listings"][0]["model"] == "Air Blade"
        until(seller, "hello")

        buyer.send_json({"text": "Air Blade này còn không anh, budget em 30m thôi"})
        b, s = turn_done(buyer), turn_done(seller)
        assert any(m["type"] == "message" and m["from"] == "buyer" for m in s)        # seller sees the buyer
        to_buyer = [m["text"] for m in b if m["type"] == "agent"]
        to_seller = [m["text"] for m in s if m["type"] == "agent"]
        assert any("32tr" in t for t in to_buyer)                                     # listing details to buyer
        assert any("giá thấp nhất" in t for t in to_seller)                           # floor question to seller
        assert not any("giá thấp nhất" in t for t in to_buyer)                        # ...and only to the seller
        state = next(m for m in b if m["type"] == "state")["state"]
        assert state["stage"] == "NEGOTIATION" and state["listing"]["id"] == "L001"

        seller.send_json({"text": "ok em ck cọc qua stk 0123456789012 nhé"})
        b, s = turn_done(buyer), turn_done(seller)
        msg_for_buyer = next(m for m in b if m["type"] == "message")
        msg_for_seller = next(m for m in s if m["type"] == "message")
        assert "0123456789012" not in msg_for_buyer["text"] and "[account]" in msg_for_buyer["text"]
        assert "0123456789012" in msg_for_seller["text"]                              # sender sees own text
        turn = next(m for m in b if m["type"] == "turn")
        assert "escalate_to_human" in [a["name"] for a in turn["actions"]]


def test_late_joiner_gets_history_for_their_role(client):
    conv = client.post("/api/conversations", json={"seller": "s_002"}).json()["conv"]
    with client.websocket_connect(f"/ws/{conv}/buyer") as buyer:
        until(buyer, "hello")
        buyer.send_json({"text": "Xe Vision này còn không anh"})
        turn_done(buyer)
    with client.websocket_connect(f"/ws/{conv}/seller") as seller:
        _, seen = until(seller, "presence")
        kinds = [m["type"] for m in seen]
        assert "message" in kinds and "state" in kinds
        assert all(m.get("to") in (None, "seller") for m in seen if m["type"] == "agent")


def test_unknown_conversation(client):
    with client.websocket_connect("/ws/nope/buyer") as ws:
        assert ws.receive_json()["type"] == "error"
