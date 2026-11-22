"""Find and redact contact details and payment identifiers in chat text.

Used before anything is written to the raw event log, and by the reply guard.
Covers what people actually type in Vietnamese marketplace chats:
  - phone numbers in digits, with spaces/dots/dashes, +84 or 0 prefix
  - phone numbers spelled out in words ("không chín không chín một hai ...")
  - bank account numbers after "stk" / "số tài khoản" / "tk", and any long digit run
  - Zalo / Facebook / Telegram / Instagram IDs
  - email addresses
"""
from __future__ import annotations

import re

_DIGIT_WORD = r"(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|\d)"

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    # account number after a banking keyword: "stk 0123456789012", "số tài khoản: 1903 4567 8901 23"
    ("account", re.compile(r"(?i)(?<!\w)(?:stk|số\s*tài\s*khoản|tài\s*khoản|số\s*tk|tk)(?!\w)[\s:.]*"
                           r"(\d(?:[\s.-]?\d){5,18})")),
    ("phone", re.compile(r"(?:\+?84|(?<!\d)0)(?:[\s.-]?\d){8,10}(?!\d)")),
    # any other run of 10+ digits (account numbers without a keyword, IDs)
    ("number", re.compile(r"(?<![\w.])\d(?:[\s.-]?\d){9,18}(?!\w)")),
    # 7+ digits spelled out in words, possibly mixed with digits
    ("spelled_number", re.compile(rf"(?i)(?<!\w){_DIGIT_WORD}(?:[\s,.-]+{_DIGIT_WORD}){{6,}}(?!\w)")),
    # "zalo id vinh.honda88", "zalo: vinhhonda", "fb vinh.honda88" (a bare word after "zalo" must look like an ID)
    ("contact_id", re.compile(r"(?i)(?<!\w)(?:zalo|fb|facebook|telegram|tele|ig|instagram)"
                              r"(?:\s*id\s*[:\-]?\s*|\s*[:\-]\s*|\s+(?=[A-Za-z0-9._-]*[\d._]))"
                              r"([A-Za-z0-9][A-Za-z0-9._-]{2,})")),
]

LABEL = {"email": "[email]", "account": "[account]", "phone": "[phone]", "number": "[number]",
         "spelled_number": "[phone]", "contact_id": "[contact_id]"}


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted text, kinds found). Unchanged text and [] when nothing is found."""
    kinds: list[str] = []
    out = text
    for kind, pat in PATTERNS:
        def sub(m: re.Match, kind=kind) -> str:
            kinds.append(kind)
            if m.groups():                      # keep the keyword ("stk", "zalo id"), drop the value
                start, end = m.span(1)
                return m.group(0)[: start - m.start()] + LABEL[kind] + m.group(0)[end - m.start():]
            return LABEL[kind]
        out = pat.sub(sub, out)
    return out, list(dict.fromkeys(kinds))


def has_contact(text: str) -> bool:
    return bool(redact(text)[1])


def has_account(text: str) -> bool:
    return "account" in redact(text)[1]
