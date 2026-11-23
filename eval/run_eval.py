#!/usr/bin/env python3
"""Offline replay evaluation against hand-labelled gold cases.

For each case in eval/gold/*.json: replay the conversation message by message
on a fresh in-memory store, then compare per-turn actions and the final state
with the gold labels. The same gold is scored for the *legacy* agent (the
replies already in the chat log: text only, no tools, no escalation) as a baseline.

  python -m eval.run_eval              # rules mode (offline, deterministic)
  python -m eval.run_eval --mode llm   # same gold, LLM extractor/planner/responder
  python -m eval.run_eval --mode llm --llm openai:<model>   # any provider (see agent/llm.py)
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import normalize as N  # noqa: E402
from agent import memory as M  # noqa: E402
from agent.memory import Store  # noqa: E402
from agent.models import Sender, State, Tracked  # noqa: E402
from agent.pipeline import Agent, load_messages  # noqa: E402
from agent.reducer import slot_coverage  # noqa: E402
from agent import llm as L  # noqa: E402
from agent import pii  # noqa: E402
from agent.responder import PHONE_RE, Reply, guard  # noqa: E402

GOLD_SETS = {"dev": ROOT / "eval/gold", "heldout": ROOT / "eval/gold_heldout",
             "scenarios": ROOT / "eval/gold_scenarios",   # failures found in live chats, kept as regression cases
             "long": ROOT / "eval/gold_long",             # 40+ message chats: early details asked again late
             "live": ROOT / "eval/gold_live"}             # live web-demo chats (two tabs) where the agent went wrong
ALL_SETS = ("dev", "heldout", "scenarios")                 # "all" keeps its meaning (the 9 original cases)


def reply_check(gold: dict, texts: list[str]) -> list[str]:
    """Facts the buyer's reply must state (each inner list: any one phrase) and details it must not invent."""
    said = N.fold(" ".join(texts))
    miss = [f"missing any of {g}" for g in gold.get("reply_must_include", [])
            if not any(N.fold(x) in said for x in g)]
    bad = [f"says {x!r}" for x in gold.get("reply_must_not_include", []) if N.fold(x) in said]
    return miss + bad


def flatten(state: State) -> dict:
    out = {}
    for name in type(state.constraints).model_fields:
        v = getattr(state.constraints, name)
        if isinstance(v, Tracked):
            if not str(v.source_msg).startswith("implied:"):   # inferences aren't scored as extraction
                out[f"constraints.{name}"] = v.value
        elif isinstance(v, list) and v:
            out[f"constraints.{name}"] = v
    sel = state.listing_context.selected
    if sel:
        for k in ("listing_id", "model", "brand", "year", "odo_km", "ask_price", "seller_floor", "paperwork_status"):
            if getattr(sel, k) is not None:
                out[f"listing.{k}"] = getattr(sel, k)
    for k in ("seller_stance", "buyer_stance", "buyer_offer"):
        if getattr(state.negotiation, k):
            out[f"negotiation.{k}"] = getattr(state.negotiation, k)
    for k in ("buyer_bridge", "buyer_contact_share"):
        if getattr(state.consent, k) is not None:
            out[f"consent.{k}"] = getattr(state.consent, k)
    ap = state.appointment
    if ap.proposed_place:
        out["appointment.proposed_place"] = ap.proposed_place
    if ap.status != "none":
        out["appointment.status"] = ap.status
    return out


def same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    return a == b


def score_turn(gold: dict, names: list[str], asked: list[str], listing: str | None = None) -> tuple[bool, list[str]]:
    why = []
    if "expected_listing" in gold and listing != gold["expected_listing"]:
        why.append(f"on bike {listing}, expected {gold['expected_listing']}")
    wrong_q = [q for q in asked if q in gold.get("forbidden_questions", [])]
    if wrong_q:
        why.append(f"asked forbidden {wrong_q}")
    if not any(n in gold["acceptable_actions"] for n in names):
        why.append(f"none of {names} in acceptable {gold['acceptable_actions']}")
    bad = [n for n in names if n in gold.get("forbidden_actions", [])]
    if bad:
        why.append(f"forbidden {bad}")
    missing = [r for r in gold.get("required_actions", []) if r not in names]
    if missing:
        why.append(f"missing required {missing}")
    if gold.get("required_question") and gold["required_question"] not in asked:
        why.append(f"did not ask {gold['required_question']}")
    return not why, why


def run_case(gold: dict, mode: str, llm=None) -> dict:
    store = Store(":memory:")
    agent = Agent(store, mode=mode, llm=llm)
    calls0, chars0 = getattr(agent.llm, "calls", 0), getattr(agent.llm, "prompt_chars", 0)   # llm may be shared
    msgs = load_messages([ROOT / p for p in gold["sources"]], gold["conversation_id"])
    by_msg = {t["after_msg"]: t for t in gold["turns"]}
    turns, legacy_turns, replies_all, flags = [], [], [], []
    first_search_cov = None
    pins = turns_pinned = 0
    pending_legacy: dict | None = None
    legacy_promises = []

    for m in msgs:
        res = agent.handle(m)
        if m.sender == Sender.AGENT:
            # the old agent's reply answers the previous gold turn
            # the legacy agent never called tools, so every commitment it made is unbacked
            hits = guard([Reply(to="buyer", text=m.text)], set())
            if hits:
                legacy_promises.append({"msg": m.msg_id, "promise": hits})
            continue
        names = [a.name for a in res.actions]
        asked = [a.args.get("field") for a in res.actions if a.name == "ask_clarification"]
        flags += [{**f, "msg": m.msg_id} for f in res.signals]
        n = len(M.pinned_messages(res.state, store, store.recent_messages(m.conversation_id)))
        pins, turns_pinned = pins + n, turns_pinned + bool(n)
        replies_all.append((res.replies, set(names)))
        if first_search_cov is None and "search_listings" in names:
            first_search_cov = slot_coverage(res.state)
        g = by_msg.get(m.msg_id)
        if not g:
            continue
        sel = res.state.listing_context.selected
        ok, why = score_turn(g, names, asked, sel.listing_id if sel else None)
        escalated = any(r.status == "escalated" for r in res.state.active_risks())
        checks = {"reply_must_include", "reply_must_not_include", "seller_reply_must_include", "seller_reply_must_not_include"}
        reply_miss = None
        if checks & set(g):
            reply_miss = reply_check(g, [r.text for r in res.replies if r.to in ("buyer", "both")])
            reply_miss += [f"seller: {x}" for x in reply_check(
                {"reply_must_include": g.get("seller_reply_must_include", []),
                 "reply_must_not_include": g.get("seller_reply_must_not_include", [])},
                [r.text for r in res.replies if r.to in ("seller", "both")])]
        phone_leak = any(PHONE_RE.search(r.text) or pii.has_contact(r.text) for r in res.replies)
        ex = res.extraction
        turns.append({"msg": m.msg_id, "actions": names, "asked": asked, "correct": ok, "why": why,
                      "intents": [i.value for i in ex.intents] if ex else [],
                      "facts": [{"field": f.field, "value": f.value, "conf": f.confidence} for f in ex.facts] if ex else [],
                      "selected": sel.listing_id if sel else None,
                      "replies": [r.model_dump() for r in res.replies],
                      "should_escalate": g["should_escalate"], "escalated": escalated,
                      "stage": res.state.lead_stage.value, "expected_stage": g.get("expected_stage"),
                      "forbidden_hit": [n for n in names if n in g.get("forbidden_actions", [])],
                      "phone_leak": phone_leak, "reply_check": reply_miss})
        lok, lwhy = score_turn(g, ["respond_only"], [])
        legacy_turns.append({"msg": m.msg_id, "correct": lok, "should_escalate": g["should_escalate"], "escalated": False})

    final = store.load_state(gold["conversation_id"])
    pred = flatten(final)
    exp = gold["expected_final_state"]
    fields = {}
    for k in set(pred) | set(exp):
        p, e = pred.get(k), exp.get(k)
        fields[k] = {"pred": p, "gold": e,
                     "tp": int(p is not None and e is not None and same(p, e)),
                     "fp": int(p is not None and (e is None or not same(p, e))),
                     "fn": int(e is not None and (p is None or not same(p, e)))}
    # risks that apply now: pending papers on a bike the buyer moved away from no longer count
    open_risks = sorted(r.type.value for r in final.active_risks() if r.status != "resolved")
    unsourced = [k for k in type(final.constraints).model_fields
                 if isinstance(getattr(final.constraints, k), Tracked) and not getattr(final.constraints, k).source_msg]
    unbacked = sum(1 for reps, names in replies_all if guard(reps, names))
    return {
        "case_id": gold["case_id"], "objection": gold["objection"], "turns": turns, "legacy_turns": legacy_turns,
        "fields": fields, "final_stage": final.lead_stage.value, "expected_final_stage": gold["expected_final_stage"],
        "open_risks": open_risks, "expected_open_risks": sorted(gold["expected_open_risks"]),
        "slot_coverage_final": slot_coverage(final), "slot_coverage_first_search": first_search_cov,
        "unsourced_facts": unsourced, "replies": sum(len(r) for r, _ in replies_all), "unbacked_promises": unbacked,
        "legacy_unbacked_promises": legacy_promises, "llm_fallbacks": _fallbacks(agent),
        "signals": flags, "pins": pins, "turns_with_pins": turns_pinned,
        "llm_calls": getattr(agent.llm, "calls", 0) - calls0, "prompt_chars": getattr(agent.llm, "prompt_chars", 0) - chars0,
        "reply_retries": getattr(agent.responder, "retries", 0),
        "reply_fallbacks": getattr(agent.responder, "guard_failures", 0),
        "number_disagreements": getattr(agent.extractor, "number_disagreements", 0),
    }


def _fallbacks(agent: Agent) -> int:
    return sum(getattr(x, "fallbacks", 0) + getattr(x, "overrides", 0) for x in (agent.extractor, agent.planner)) + \
        getattr(agent.responder, "guard_failures", 0)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def escalation_pr(turns: list[dict]) -> tuple[float, float]:
    tp = sum(t["escalated"] and t["should_escalate"] for t in turns)
    fp = sum(t["escalated"] and not t["should_escalate"] for t in turns)
    fn = sum(not t["escalated"] and t["should_escalate"] for t in turns)
    p, r, _ = prf(tp, fp, fn)
    return p, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rules", choices=["rules", "auto", "llm"])
    ap.add_argument("--llm", default=None, help="provider:model, e.g. openai:<model>, gemini:<model> (see agent/llm.py)")
    ap.add_argument("--set", default="dev", choices=["dev", "heldout", "scenarios", "long", "live", "all"],
                    help="dev = cases written while building; heldout = paraphrased cases written afterwards; "
                         "scenarios = failures found in live chats; long = 40+ message chats (not in all)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--recent-k", type=int, default=None, help=f"recent-message window (default {M.RECENT_K})")
    ap.add_argument("--no-pins", action="store_true", help="turn pinned anchors off (memory.MAX_PINS = 0)")
    ap.add_argument("--record", metavar="FILE", help="LLM mode: save every LLM answer to FILE (costs as a normal run)")
    ap.add_argument("--replay", metavar="FILE", help="LLM mode from a recording: free, same result every time")
    args = ap.parse_args()
    llm_obj = args.llm
    if args.replay:
        args.mode, llm_obj = "llm", L.CachedLLM(None, args.replay, "replay")
    elif args.record:
        args.mode, llm_obj = "llm", L.CachedLLM(L.build_llm(args.llm), args.record, "record")
    if args.recent_k:
        M.RECENT_K = args.recent_k
    if args.no_pins:
        M.MAX_PINS = 0
    tag = (f"_k{args.recent_k}" if args.recent_k else "") + ("_nopins" if args.no_pins else "")

    dirs = [GOLD_SETS[s] for s in ALL_SETS] if args.set == "all" else [GOLD_SETS[args.set]]
    files = [p for d in dirs for p in sorted(d.glob("*.json"))]
    results = [run_case(json.loads(p.read_text(encoding="utf-8")), args.mode, llm_obj) for p in files]
    if isinstance(llm_obj, L.CachedLLM):
        llm_obj.save()
        llm_name = f"{llm_obj.name} ({llm_obj.mode} {llm_obj.path.name}; {llm_obj.hits} hits, {llm_obj.misses} misses)"
    else:
        llm_name = "-" if args.mode == "rules" else (L.make_llm(args.mode, args.llm) or SimpleNamespace(name="- (no credentials, rules)")).name
    print(f"gold set: {args.set} ({len(files)} cases), mode: {args.mode}, llm: {llm_name}, "
          f"window: {M.RECENT_K} messages, pins: {'off' if args.no_pins else f'≤{M.MAX_PINS}'}")
    all_turns = [t for r in results for t in r["turns"]]
    legacy = [t for r in results for t in r["legacy_turns"]]

    per_field = defaultdict(lambda: [0, 0, 0])
    for r in results:
        for k, v in r["fields"].items():
            group = k.split(".")[0]
            for i, key in enumerate(("tp", "fp", "fn")):
                per_field[group][i] += v[key]
                per_field["ALL"][i] += v[key]

    na = sum(t["correct"] for t in all_turns) / len(all_turns)
    na_legacy = sum(t["correct"] for t in legacy) / len(legacy)
    ep, er = escalation_pr(all_turns)
    lp, lr = escalation_pr(legacy)
    staged = [t for t in all_turns if t["expected_stage"]]          # the live set labels outcomes, not stages
    stage_acc = sum(t["stage"] == t["expected_stage"] for t in staged) / len(staged) if staged else None
    risk_match = sum(r["open_risks"] == r["expected_open_risks"] for r in results) / len(results)
    violations = sum(len(t["forbidden_hit"]) + t["phone_leak"] for t in all_turns)
    replies = sum(r["replies"] for r in results)
    unbacked = sum(r["unbacked_promises"] for r in results)
    legacy_unbacked = sum(len(r["legacy_unbacked_promises"]) for r in results)

    if not args.quiet:
        for r in results:
            print(f"\n=== {r['case_id']}: {r['objection']}")
            for t in r["turns"]:
                mark = "✓" if t["correct"] else "✗"
                esc = "esc" if t["escalated"] else "   "
                print(f"  {mark} {t['msg']:<6} {esc} stage={t['stage']:<12} actions={t['actions']}"
                      + (f" asked={t['asked']}" if t["asked"] else "") + (f"  <- {'; '.join(t['why'])}" if t["why"] else ""))
                if t.get("reply_check"):
                    print(f"      reply ✗ {'; '.join(t['reply_check'])}: "
                          f"{' | '.join(x['text'] for x in t['replies'] if x['to'] != 'seller')[:160]!r}")
            wrong = {k: v for k, v in r["fields"].items() if v["fp"] or v["fn"]}
            for k, v in sorted(wrong.items()):
                print(f"    field {k}: pred={v['pred']!r} gold={v['gold']!r}")
            if r["open_risks"] != r["expected_open_risks"]:
                print(f"    open risks {r['open_risks']} vs gold {r['expected_open_risks']}")
            for f in r["signals"]:
                print(f"    ⚠ {f['msg']} {f['type']} ({f['side']}): {f['note'][:100]}")
            for lp_ in r["legacy_unbacked_promises"]:
                print(f"    legacy agent at {lp_['msg']} promised {lp_['promise']} with no tool behind it")

    print("\n" + "=" * 66)
    print("reply-only = the legacy agent's behaviour in the log: text replies, no tools, no escalation")
    print(f"{'metric':<40}{'this agent':>13}{'reply-only':>13}")
    print("-" * 66)
    print(f"{'next-action accuracy':<40}{na:>13.0%}{na_legacy:>13.0%}")
    print(f"{'escalation precision':<40}{ep:>13.0%}{'n/a':>13}")
    print(f"{'escalation recall':<40}{er:>13.0%}{lr:>13.0%}")
    print(f"{'stage accuracy (per turn)':<40}{(f'{stage_acc:.0%}' if stage_acc is not None else 'n/a'):>13}{'n/a':>13}")
    print(f"{'open-risk set exact match (per case)':<40}{risk_match:>13.0%}{'n/a':>13}")
    for g in ("constraints", "listing", "negotiation", "consent", "appointment", "ALL"):
        if g in per_field:
            p, rc, f = prf(*per_field[g])
            print(f"{'extraction F1 ' + g:<40}{f:>13.0%}{'n/a':>13}   (P {p:.0%} R {rc:.0%})")
    covs = [r["slot_coverage_first_search"] for r in results if r["slot_coverage_first_search"] is not None]
    print(f"{'slot coverage at first search (avg)':<40}{(sum(covs)/len(covs) if covs else 0):>13.0%}{'n/a':>13}")
    print(f"{'unsourced facts in state':<40}{sum(len(r['unsourced_facts']) for r in results):>13}{'n/a':>13}")
    print(f"{'unbacked promises in replies':<40}{f'{unbacked}/{replies}':>13}{legacy_unbacked:>13}")
    print(f"{'policy violations (forbidden / phone)':<40}{violations:>13}{'n/a':>13}")
    sig = [f for r in results for f in r["signals"]]
    agent_sig = [f for f in sig if f["side"] == "agent"]
    print(f"{'agent quality flags (repeat/ack/bike)':<40}{len(agent_sig):>13}{'n/a':>13}"
          + (f"   {sorted({f['type'] for f in agent_sig})}" if agent_sig else ""))
    print(f"{'buyer corrections / repeats in script':<40}{len(sig) - len(agent_sig):>13}{'n/a':>13}"
          "   (scripted buyer text, so not a score)")
    print(f"{'LLM fallbacks / code overrides':<40}{sum(r['llm_fallbacks'] for r in results):>13}{'':>13}")
    if args.mode != "rules":
        print(f"{'reply retries · template fallbacks':<40}{sum(r['reply_retries'] for r in results):>6} · "
              f"{sum(r['reply_fallbacks'] for r in results)}")
        print(f"{'number disagreements (regex vs LLM)':<40}{sum(r['number_disagreements'] for r in results):>13}")
    checked = [t for t in all_turns if t.get("reply_check") is not None]
    if checked:
        print(f"{'reply fact checks (stated, not invented)':<40}"
              f"{sum(not t['reply_check'] for t in checked):>9}/{len(checked):<3}{'':>13}")
    calls = sum(r["llm_calls"] for r in results)
    print(f"{'turns with pinned messages':<40}{sum(r['turns_with_pins'] for r in results):>13}{'':>13}"
          f"   ({sum(r['pins'] for r in results)} pins)")
    if calls:
        print(f"{'LLM calls · avg prompt chars per call':<40}{calls:>13}{sum(r['prompt_chars'] for r in results) // calls:>13,}")
    print("=" * 66)

    tag += "_replay" if args.replay else "_llm" if args.mode != "rules" else ""
    out = ROOT / f"runs/eval_report_{args.set}{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"mode": args.mode, "llm": llm_name, "set": args.set, "next_action_accuracy": na, "legacy_next_action_accuracy": na_legacy,
                               "escalation_precision": ep, "escalation_recall": er, "stage_accuracy": stage_acc,
                               "policy_violations": violations,
                               "agent_quality_flags": len([f for r in results for f in r["signals"] if f["side"] == "agent"]),
                               "cases": results},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"report: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
