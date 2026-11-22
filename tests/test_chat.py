"""The live demo: `python run.py chat`, and the gap it exposed (seller's floor never reached the buyer)."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.memory import Store
from agent.models import Message, Sender
from agent.pipeline import Agent

ROOT = Path(__file__).resolve().parent.parent


def test_chat_runs_demo_script(tmp_path):
    out = subprocess.run([sys.executable, "run.py", "chat", "--mode", "rules", "--store", str(tmp_path / "c.db")],
                         cwd=ROOT, input=(ROOT / "demo/live_demo.txt").read_text(), capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    t = out.stdout
    assert "khu vực HCM" in t                       # empty search names the blocking filter
    assert "DEPOSIT_REQUEST[high]:escalated" in t   # /state after the bank-transfer request
    assert "[contact_id]" in t and "vinh.honda88\"" not in t.split("/events")[-1]
    assert "b> live" not in t                        # prompt printed once per line when piped
    assert "None" not in t


def test_seller_floor_reaches_buyer_when_budget_known(tmp_path):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps({"f": {"buyer_id": "b", "seller_id": "s_001"}}))
    agent = Agent(Store(":memory:"), mode="rules", conversations_path=conv)
    turns = []
    for i, (s, text) in enumerate([("buyer", "Air Blade này còn không anh, budget em 30m thôi"),
                                   ("seller", "còn em, bớt còn 31 triệu")], 1):
        turns.append(agent.handle(Message(conversation_id="f", msg_id=f"f#{i}", index=i, sender=Sender(s), text=text,
                                          timestamp=datetime(2026, 3, 1, 8, i, tzinfo=timezone.utc))))
    last = turns[-1]
    assert "accept_floor" in [a.args.get("field") for a in last.actions]
    assert "31tr" in " ".join(r.text for r in last.replies if r.to == "buyer")
