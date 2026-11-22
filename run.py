#!/usr/bin/env python3
"""CLI for the motorbike marketplace agent.

  python run.py replay --conv c1                 # replay one conversation, turn by turn
  python run.py replay --conv c1 --followup      # ...plus a follow-up that reaches a booked viewing
  python run.py replay --all                     # c1, c2, c3
  python run.py replay --conv v1                 # a held-out conversation (v1-v3)
  python run.py state --conv c1                  # print the stored state JSON
  python run.py events --conv c2                 # print the raw event log
  python run.py feedback --conv c1 --outcome booked
  python run.py mine-fewshot                     # build data/fewshot.json from good outcomes
  python run.py flags --db runs/web.db           # error-analysis queue: turns flagged by in-chat signals
  python run.py chat --seller s_002              # live demo: type as buyer (b:) or seller (s:)
  python run.py web                              # browser demo: buyer and seller in two tabs
  python run.py chat --llm openai:<model>        # any provider: anthropic | openai | gemini | ollama | ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.feedback import export_flags, mine_fewshot, record_outcome  # noqa: E402
from agent.memory import Store  # noqa: E402
from agent.models import Message, Sender, State, Tracked  # noqa: E402
from agent.pipeline import Agent, TurnResult, load_messages  # noqa: E402
from agent.reducer import slot_coverage  # noqa: E402

DB = ROOT / "runs/agent.db"
CHAT_DB = ROOT / "runs/chat.db"
CHAT = ROOT / "data/chat_history.jsonl"
FOLLOWUP = ROOT / "data/c1_followup.jsonl"
HELDOUT = ROOT / "data/heldout.jsonl"   # v1-v3


def show(t: TurnResult, verbose: bool) -> None:
    m = t.message
    print(f"\n{'─' * 78}\n[{m.msg_id}] {m.sender.value.upper()}: {m.text}")
    if t.legacy:
        print("   (old agent's reply in the log - kept as history, not re-processed)")
        return
    s = t.state
    ex = t.extraction
    print(f"   intents : {', '.join(i.value for i in ex.intents) or '-'}")
    for f in ex.facts:
        print(f"   fact    : {f.field} = {f.value!r}  (conf {f.confidence})")
    for r in ex.risks:
        print(f"   risk    : {r.type.value} [{r.severity}] {r.note}")
    shown = [d for d in t.diffs if d["field"] not in ("summary",)]
    if verbose:
        for d in shown:
            print(f"   Δ state : {d['field']}: {d['old']!r} -> {d['new']!r}   ({d['source']})")
    print(f"   stage   : {s.lead_stage.value}   coverage {slot_coverage(s):.0%}   "
          f"gap {'-' if s.negotiation.gap_pct is None else f'{s.negotiation.gap_pct:.0%}'}   "
          f"risks {[f'{r.type.value}:{r.status}' for r in s.risks] or '-'}")
    if verbose:
        print(f"   allowed : {', '.join(t.allowed)}")
        for k, v in t.blocked.items():
            if k != "share_contact":
                print(f"   blocked : {k} - {v}")
    for a, res in zip(t.actions, t.tool_results):
        detail = a.args.get("field") if a.name == "ask_clarification" else (json.dumps(a.args, ensure_ascii=False) if a.args else "")
        print(f"   ACTION  : {a.name} {detail}")
        print(f"             why: {a.reason}")
    for f in getattr(t, "signals", []):
        print(f"   ⚠ {f['type']:<16}: {f['note']}" + (f"  (blames {f['blames']})" if f.get("blames") else ""))
    for r in t.replies:
        text = r.text.replace("\n", "\n" + " " * 21)
        print(f"   → {r.to:<6}: {text}")


def cmd_replay(args) -> None:
    db = Path(args.db)
    if db.exists() and not args.keep:
        db.unlink()
    store = Store(db)
    agent = Agent(store, mode=args.mode, llm=args.llm)
    print(_header(agent))
    paths = [CHAT, HELDOUT] + ([FOLLOWUP] if args.followup else [])
    convs = ["c1", "c2", "c3"] if args.all else [args.conv]
    for c in convs:
        print(f"\n{'=' * 78}\nCONVERSATION {c}\n{'=' * 78}")
        for m in load_messages(paths, c):
            show(agent.handle(m), args.verbose)
        s = store.load_state(c)
        print(f"\nSUMMARY {c}: {s.summary}")
    print(f"\nstate + event log saved to {db.relative_to(ROOT)}")


def cmd_state(args) -> None:
    s = Store(args.db).load_state(args.conv)
    print(s.model_dump_json(indent=2) if s else f"no state for {args.conv} (run replay first)")


def cmd_events(args) -> None:
    for e in Store(args.db).events(args.conv):
        print(f"{e['at_msg'] or '-':<6} {e['type']:<13} {json.dumps(e['payload'], ensure_ascii=False)[:160]}")


def cmd_feedback(args) -> None:
    record_outcome(Store(args.db), args.conv, args.outcome, args.note or "")
    print(f"recorded outcome {args.outcome!r} for {args.conv}")


def cmd_flags(args) -> None:
    rows = export_flags(Store(args.db))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        print("no flagged turns")
    for r in rows:
        print(f"\n{r['conversation_id']} · bad turn {r['bad_turn']}: {r['message']!r}")
        print("   actions : " + (", ".join(a["name"] + (f"({a['field']})" if a["field"] else "") for a in r["actions"]) or "-"))
        for rep in r["replies"]:
            print(f"   → {rep['to']:<6}: {rep['text'][:160]}")
        for f in r["signals"]:
            print(f"   ⚠ {f['type']} [{f['severity']}] seen at {f['seen_at']}: {f['note'][:140]}")
    print(f"\n{len(rows)} flagged turn(s). Next: confirm each, add it to eval/gold_scenarios with a failing test, fix, re-run evals.")


def cmd_mine(args) -> None:
    examples = mine_fewshot(Store(args.db), ROOT / "data/fewshot.json")
    print(f"wrote {len(examples)} few-shot example(s) to data/fewshot.json")


def _header(agent: Agent) -> str:
    return (f"mode: {agent.mode}  llm: {agent.llm_name}  "
            f"(extractor={type(agent.extractor).__name__}, planner={type(agent.planner).__name__})")


# --------------------------------------------------------------------------- #
# Live chat demo
# --------------------------------------------------------------------------- #
CHAT_HELP = """Type a message as the buyer or the seller:
  b: Mình cần xe ga tầm 25 củ ở Sài Gòn       (buyer)
  s: Xe mình giá 21tr, giấy tờ đang rút hồ sơ   (seller)
  a line without a prefix is sent by whoever spoke last
Commands:
  /state            compact state: stage, constraints, listing, risks, open questions, summary
  /json             full state JSON
  /events [n]       last n raw log events (default 8), contact details already redacted
  /listings         this seller's listings (or the whole inventory when there is no seller)
  /seller ID|none   start a new conversation with another seller (e.g. /seller s_003)
  /new              start a new conversation with the same seller
  /verbose          toggle state diffs and allowed/blocked actions per turn
  /help, /quit"""


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, int) and v >= 1_000_000:
        return f"{v / 1e6:g}tr"
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)


def print_state(s: State) -> None:
    print(f"   stage       : {s.lead_stage.value}   coverage {slot_coverage(s):.0%}")
    cons = [f"{k}={_fmt(t.value)} ({t.source_msg})" for k, t in s.constraints if isinstance(t, Tracked)]
    print("   constraints : " + ("; ".join(cons) or "-"))
    if s.constraints.deal_breakers:
        print(f"   deal-breaks : {', '.join(s.constraints.deal_breakers)}")
    sel = s.listing_context.selected
    if sel:
        print(f"   listing     : {sel.listing_id or '?'} {sel.model or ''} {sel.year or ''}  ask {_fmt(sel.ask_price)}  "
              f"floor {_fmt(sel.seller_floor)}  paperwork {sel.paperwork_status or '?'}")
    if s.listing_context.candidates:
        print("   candidates  : " + ", ".join(f"{c.listing_id} {c.model} {_fmt(c.ask_price)}" for c in s.listing_context.candidates))
    n = s.negotiation
    if n.gap_pct is not None or n.buyer_offer:
        print(f"   negotiation : gap {'-' if n.gap_pct is None else f'{n.gap_pct:.0%}'}  buyer offer {_fmt(n.buyer_offer)}  "
              f"seller {n.seller_stance or '-'}")
    print("   risks       : " + (", ".join(f"{r.type.value}[{r.severity}]:{r.status}"
                                          + (f"@{r.listing_id}" + ("" if s.applies(r) else " (other bike)") if r.listing_id else "")
                                          for r in s.risks) or "-"))
    print("   open qs     : " + (", ".join(f"{q.field}->{q.ask}" for q in s.open_questions) or "-"))
    if s.channel_id or s.appointment.status != "none":
        print(f"   channel     : {s.channel_id or '-'}   appointment {s.appointment.status} "
              f"{s.appointment.proposed_time or ''} {s.appointment.proposed_place or ''}")
    print(f"   summary     : {s.summary or '-'}")


def cmd_chat(args) -> None:
    try:
        import readline  # noqa: F401  (arrow keys and history when typing)
    except ImportError:
        pass
    db = Path(args.store)
    if db.exists() and not args.keep:
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)
    store = Store(db)
    agent = Agent(store, mode=args.mode, llm=args.llm)
    echo = not sys.stdin.isatty()           # piped script: print each input line so the transcript reads well
    seller = None if args.seller in (None, "none") else args.seller
    verbose = args.verbose
    session = {"n": 1}

    def conv_id() -> str:
        return args.conv if session["n"] == 1 else f"{args.conv}-{session['n']}"

    def start(new_seller) -> None:
        nonlocal seller
        seller = new_seller
        agent.meta[conv_id()] = {"buyer_id": "b_live", "seller_id": seller}
        listings = agent.tools.find_seller_listings(seller) if seller else []
        who = f"seller {seller} (" + ", ".join(f"{l['model']} {l['year']} {_fmt(l['price'])}" for l in listings) + ")" \
            if seller else "no seller yet (buyer is searching)"
        print(f"\n── conversation {conv_id()}: buyer b_live + {who}")

    print(_header(agent))
    print(CHAT_HELP)
    start(seller)
    sender = "buyer"
    while True:
        prompt = f"\n{conv_id()} {sender[0]}> "
        try:
            line = input("" if echo else prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if echo and line:                  # piped script: show comments as narration, messages after the prompt
            print(("\n" + line) if line.startswith("#") else (prompt + line))
        if not line or line.startswith("#"):
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            arg = arg.strip()
            st = store.load_state(conv_id())
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(CHAT_HELP)
            elif cmd == "state":
                print_state(st) if st else print("   (no messages yet)")
            elif cmd == "json":
                print(st.model_dump_json(indent=2) if st else "   (no messages yet)")
            elif cmd == "events":
                for e in store.events(conv_id())[-(int(arg) if arg.isdigit() else 8):]:
                    print(f"   {e['at_msg'] or '-':<9} {e['type']:<13} {json.dumps(e['payload'], ensure_ascii=False)[:150]}")
            elif cmd == "listings":
                rows = agent.tools.find_seller_listings(seller) if seller else agent.tools.listings
                for l in rows:
                    print(f"   {l['listing_id']} {l['model']} {l['year']} {_fmt(l['price'])}"
                          + (f" {l['location']} {l['paperwork_status']}" if "location" in l else ""))
            elif cmd == "seller":
                session["n"] += 1
                start(None if arg in ("", "none") else arg)
                sender = "buyer"
            elif cmd == "new":
                session["n"] += 1
                start(seller)
                sender = "buyer"
            elif cmd == "verbose":
                verbose = not verbose
                print(f"   verbose {'on' if verbose else 'off'}")
            else:
                print(f"   unknown command /{cmd}; /help lists them")
            continue
        m = re.match(r"^(b|buyer|s|seller)\s*[:>]\s*(.*)$", line, re.I)
        if m:
            sender = "buyer" if m.group(1)[0].lower() == "b" else "seller"
            line = m.group(2).strip()
        if not line:
            continue
        if sender == "seller" and not seller:
            print("   (no seller in this chat yet; use /seller s_002 to start one with a seller)")
        st = store.load_state(conv_id())
        i = (st.msg_count if st else 0) + 1
        msg = Message(conversation_id=conv_id(), msg_id=f"{conv_id()}#{i}", index=i,
                      timestamp=datetime.now(timezone.utc), sender=Sender(sender), text=line)
        show(agent.handle(msg), verbose)
    print(f"state + event log saved to {db.relative_to(ROOT) if db.is_relative_to(ROOT) else db}")


def cmd_web(args) -> None:
    from web.server import serve
    print(f"live demo on http://{args.host}:{args.port}  (open the lobby, then the buyer and seller links)")
    serve(host=args.host, port=args.port, mode=args.mode, llm=args.llm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(DB))
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--conv", default="c1")
    r.add_argument("--all", action="store_true")
    r.add_argument("--followup", action="store_true", help="append data/c1_followup.jsonl")
    r.add_argument("--mode", default="auto", choices=["auto", "rules", "llm"])
    r.add_argument("--llm", default=None, help='"provider:model", e.g. anthropic:claude-opus-5-5, openai:<model>, gemini:<model>, ollama:<model> (default: $AGENT_LLM or anthropic:claude-opus-5-5)')
    r.add_argument("--verbose", "-v", action="store_true")
    r.add_argument("--keep", action="store_true", help="keep the existing db instead of starting fresh")
    r.set_defaults(fn=cmd_replay)
    s = sub.add_parser("state"); s.add_argument("--conv", required=True); s.set_defaults(fn=cmd_state)
    e = sub.add_parser("events"); e.add_argument("--conv", required=True); e.set_defaults(fn=cmd_events)
    f = sub.add_parser("feedback")
    f.add_argument("--conv", required=True)
    f.add_argument("--outcome", required=True, choices=["booked", "showed_up", "closed", "dropped", "bad_reply"])
    f.add_argument("--note")
    f.set_defaults(fn=cmd_feedback)
    m = sub.add_parser("mine-fewshot"); m.set_defaults(fn=cmd_mine)
    fl = sub.add_parser("flags", help="turns flagged by in-chat quality signals (error-analysis queue)")
    fl.add_argument("--out", help="also write the queue as JSON")
    fl.add_argument("--db", default=argparse.SUPPRESS, help="sqlite store to read, e.g. runs/web.db or runs/chat.db")
    fl.set_defaults(fn=cmd_flags)
    c = sub.add_parser("chat", help="live demo: type messages as buyer or seller")
    c.add_argument("--seller", default=None, help="seller id in this 3-way chat (s_001..s_012), or none")
    c.add_argument("--conv", default="live")
    c.add_argument("--mode", default="auto", choices=["auto", "rules", "llm"])
    c.add_argument("--llm", default=None, help='"provider:model", e.g. anthropic:claude-opus-5-5, openai:<model>, gemini:<model>, ollama:<model> (default: $AGENT_LLM or anthropic:claude-opus-5-5)')
    c.add_argument("--verbose", "-v", action="store_true")
    c.add_argument("--store", default=str(CHAT_DB), help="sqlite file for this demo (fresh unless --keep)")
    c.add_argument("--keep", action="store_true")
    c.set_defaults(fn=cmd_chat)
    w = sub.add_parser("web", help="live demo in the browser: buyer and seller in two tabs")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--mode", default="auto", choices=["auto", "rules", "llm"])
    w.add_argument("--llm", default=None, help="provider:model (see agent/llm.py)")
    w.set_defaults(fn=cmd_web)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
