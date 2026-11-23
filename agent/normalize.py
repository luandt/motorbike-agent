"""Parse chat logs and normalise Vietnamese marketplace text.

Numbers (price, odo, year) are parsed with rules, not the LLM: they are
cheap, exact and unit-testable. "25tr" -> 25_000_000, "odo 19k" -> 19_000.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from .models import Message, Sender

VN_TZ = timezone(timedelta(hours=7))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_jsonl(paths: Iterable[str | Path]) -> list[Message]:
    """Load one or more jsonl files, skip blank/bad lines, number messages per conversation."""
    rows: list[dict] = []
    for p in paths:
        for line_no, line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[normalize] skipping bad json at {p}:{line_no}")
    rows.sort(key=lambda r: (r["conversation_id"], r["timestamp"]))
    counters: dict[str, int] = {}
    out: list[Message] = []
    for r in rows:
        sender = str(r.get("sender", "")).strip().lower()
        if sender not in {s.value for s in Sender}:
            print(f"[normalize] unknown sender {sender!r}, skipping")
            continue
        cid = r["conversation_id"]
        counters[cid] = counters.get(cid, 0) + 1
        out.append(Message(
            conversation_id=cid,
            msg_id=f"{cid}#{counters[cid]}",
            index=counters[cid],
            timestamp=datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
            sender=Sender(sender),
            text=" ".join(str(r.get("text", "")).split()),
        ))
    return out


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def fold(text: str) -> str:
    """Lower-case, accent-free form used for phrase matching."""
    return strip_accents(text).lower()


def has_any(text: str, phrases: Iterable[str]) -> bool:
    f = fold(text)
    return any(re.search(r"(?<!\w)" + re.escape(fold(p)) + r"(?!\w)", f) for p in phrases)


def clauses(text: str) -> list[str]:
    parts = re.split(r"[,.;!?\n]|\s(?:nhưng|nhung|mà)\s", text)
    return [p.strip() for p in parts if p and p.strip()]


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
_NUM = r"(\d+(?:[.,]\d+)?)"
_MONEY_UNIT = r"(tr|triệu|trieu|củ|cu|chai|t)"     # "20t" is chat shorthand for 20 triệu
MONEY_RANGE_RE = re.compile(_NUM + r"\s*(?:-|–|~|đến|den|tới|toi)\s*" + _NUM + r"\s*" + _MONEY_UNIT + r"(?!\w)", re.I)
MONEY_RE = re.compile(_NUM + r"\s*" + _MONEY_UNIT + r"(?!\w)", re.I)
MONEY_M_RE = re.compile(_NUM + r"\s*m(?!\w)", re.I)    # "30m"; only 10-300 counts (not "500m", "5m")
MONEY_DEC_RE = re.compile(r"(\d+)\s*" + _MONEY_UNIT + r"(\d{1,3})(?!\w)", re.I)      # "27tr5" = 27.5tr
BARE_NUM_RE = re.compile(r"(?<![\d.,\w])(\d{2,3}(?:[.,]\d)?)(?![\d\w])")              # "26" in a price clause
ODO_RE = re.compile(r"odo\s*:?\s*" + _NUM + r"\s*(k|nghìn|ngàn|nghin|ngan|vạn|van|km)?(?!\w)", re.I)
YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")
YEAR_MIN_RE = re.compile(r"(?:đời|doi|năm|nam|từ|tu)?\s*(20[0-3]\d)\s*(?:trở lên|tro len|đổ lên|do len|trở về sau|\+)", re.I)


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


@dataclass
class MoneyMention:
    low: int
    high: int
    span: tuple[int, int]


def parse_money(text: str, allow_bare: bool = False) -> list[MoneyMention]:
    """allow_bare: also read a bare 10-200 as millions (only call it for clauses known to be about price)."""
    out: list[MoneyMention] = []
    taken: list[tuple[int, int]] = []
    for m in MONEY_DEC_RE.finditer(text):
        v = int((int(m.group(1)) + int(m.group(3)) / 10 ** len(m.group(3))) * 1_000_000)
        out.append(MoneyMention(v, v, m.span()))
        taken.append(m.span())
    for m in MONEY_RANGE_RE.finditer(text):
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        out.append(MoneyMention(int(lo * 1_000_000), int(hi * 1_000_000), m.span()))
        taken.append(m.span())
    for m in MONEY_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        v = int(_to_float(m.group(1)) * 1_000_000)
        out.append(MoneyMention(v, v, m.span()))
        taken.append(m.span())
    for m in MONEY_M_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        x = _to_float(m.group(1))
        if 10 <= x <= 300:
            out.append(MoneyMention(int(x * 1_000_000), int(x * 1_000_000), m.span()))
            taken.append(m.span())
    if allow_bare and not out:
        for m in BARE_NUM_RE.finditer(text):
            x = _to_float(m.group(1))
            if 10 <= x <= 200:
                out.append(MoneyMention(int(x * 1_000_000), int(x * 1_000_000), m.span()))
    out.sort(key=lambda x: x.span[0])
    return out


DISCOUNT_RE = re.compile(r"(?<!\w)(?:bot|giam)\s+(?:them\s+|cho\s+|duoc\s+)?(\d+(?:[.,]\d+)?)\s*"
                         r"(k|nghin|ngan|tr|trieu|cu|t)(?!\w)")


def parse_discount(text: str) -> Optional[int]:
    """A cut BY an amount ("bớt 500k", "giảm 1tr"), not a price TO ("bớt còn 24tr"). Returns VND."""
    f = fold(text)
    if re.search(r"(?<!\w)(?:bot|giam)\s+(?:xuong\s+)?con(?!\w)", f):
        return None
    m = DISCOUNT_RE.search(f)
    if not m:
        return None
    x = _to_float(m.group(1))
    return int(x * 1_000) if m.group(2) in ("k", "nghin", "ngan") else int(x * 1_000_000)


def parse_odo(text: str) -> Optional[int]:
    m = ODO_RE.search(text)
    if not m:
        return None
    v = _to_float(m.group(1))
    unit = fold(m.group(2) or "")
    if unit in {"k", "nghin", "ngan"}:
        v *= 1_000
    elif unit == "van":
        v *= 10_000
    elif unit == "" and v < 1000:          # "odo 19" almost always means 19k km
        v *= 1_000
    return int(v)


def parse_year_min(text: str) -> Optional[int]:
    m = YEAR_MIN_RE.search(text)
    return int(m.group(1)) if m else None


def parse_years(text: str) -> list[int]:
    return [int(y) for y in YEAR_RE.findall(text)]


# --------------------------------------------------------------------------- #
# Vocab
# --------------------------------------------------------------------------- #
BRANDS = ["Honda", "Yamaha", "Suzuki", "SYM", "Piaggio", "Vespa", "VinFast"]

# model keyword -> (brand, canonical model, vehicle_type)
MODELS = {
    "air blade": ("Honda", "Air Blade", "scooter"),
    "ab": ("Honda", "Air Blade", "scooter"),          # common slang
    "airblade": ("Honda", "Air Blade", "scooter"),
    "vision": ("Honda", "Vision", "scooter"),
    "lead": ("Honda", "Lead", "scooter"),
    "sh mode": ("Honda", "SH Mode", "scooter"),
    "shmode": ("Honda", "SH Mode", "scooter"),
    "sh": ("Honda", "SH", "scooter"),                 # SH 125/150/160, a different (pricier) model from SH Mode
    "pcx": ("Honda", "PCX", "scooter"),
    "future": ("Honda", "Future", "underbone"),
    "raider": ("Suzuki", "Raider", "manual_clutch"),
    "satria": ("Suzuki", "Satria", "manual_clutch"),
    "vario": ("Honda", "Vario", "scooter"),
    "wave": ("Honda", "Wave Alpha", "underbone"),
    "winner": ("Honda", "Winner", "manual_clutch"),
    "exciter": ("Yamaha", "Exciter", "manual_clutch"),
    "janus": ("Yamaha", "Janus", "scooter"),
    "grande": ("Yamaha", "Grande", "scooter"),
    "nvx": ("Yamaha", "NVX", "scooter"),
    "sirius": ("Yamaha", "Sirius", "underbone"),
}

LOCATIONS = {
    "hcm": "HCM", "tphcm": "HCM", "tp hcm": "HCM", "tp.hcm": "HCM", "sài gòn": "HCM", "saigon": "HCM", "sg": "HCM",
    "hồ chí minh": "HCM",
    "hà nội": "Hà Nội", "hn": "Hà Nội", "đà nẵng": "Đà Nẵng",
}

VEHICLE_TYPES = {"tay ga": "scooter", "xe ga": "scooter", "xe số": "underbone", "côn tay": "manual_clutch"}


def find_models(text: str) -> list[tuple[str, str, str]]:
    f = fold(text)
    hits: list[tuple[int, int, tuple[str, str, str]]] = []
    for kw, info in MODELS.items():
        for m in re.finditer(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", f):
            hits.append((m.start(), -len(kw), info))
    hits.sort(key=lambda h: (h[0], h[1]))
    seen, out, taken = set(), [], []
    for start, neg_len, info in hits:
        end = start - neg_len
        if any(a <= start < b for a, b in taken):
            continue                      # "sh" inside "sh mode": the longer name wins
        taken.append((start, end))
        if info[1] not in seen:
            seen.add(info[1])
            out.append(info)
    return out


NEGATION_RE = re.compile(r"(?<!\w)(?:khong|chang|ko|k)\s+(?:lay|thich|muon|mua|can|chon|phai|xai|dung|ung)"
                         r"(?:\s+\w+){0,2}\s*$|(?<!\w)(?:bo|tru|ngoai|khoi|ne)(?:\s+\w+){0,1}\s*$")


def find_brands(text: str) -> list[str]:
    return [b for b in BRANDS if has_any(text, [b])]


def find_brands_polarity(text: str) -> tuple[list[str], list[str]]:
    """(wanted, excluded). "không lấy Honda nữa, thích Yamaha" -> (["Yamaha"], ["Honda"])."""
    wanted, excluded = [], []
    for cl in clauses(text):
        f = fold(cl)
        for b in BRANDS:
            m = re.search(r"(?<!\w)" + re.escape(fold(b)) + r"(?!\w)", f)
            if not m:
                continue
            (excluded if NEGATION_RE.search(f[:m.start()]) else wanted).append(b)
    return list(dict.fromkeys(wanted)), [b for b in dict.fromkeys(excluded) if b not in wanted]


def find_location(text: str) -> Optional[str]:
    for kw, loc in LOCATIONS.items():
        if has_any(text, [kw]):
            return loc
    return None


def find_vehicle_type(text: str) -> Optional[str]:
    for kw, vt in VEHICLE_TYPES.items():
        if has_any(text, [kw]):
            return vt
    return None


# --------------------------------------------------------------------------- #
# Time / place for appointments ("Thứ 7 này 10h sáng ở Quận 3")
# --------------------------------------------------------------------------- #
_WEEKDAYS = {"2": 0, "hai": 0, "3": 1, "ba": 1, "4": 2, "tư": 2, "tu": 2, "5": 3, "năm": 3, "nam": 3,
             "6": 4, "sáu": 4, "sau": 4, "7": 5, "bảy": 5, "bay": 5}
TIME_RE = re.compile(r"(\d{1,2})\s*(?:h|giờ|gio)\s*(\d{2})?", re.I)
PLACE_RE = re.compile(r"(?<!\w)(?:ở|tại|tai)\s+((?:quận|quan|q\.?)\s*\d+|[\wÀ-ỹ]+(?:\s+[\wÀ-ỹ]+){0,2})", re.I)


# "đã hỏi thử chủ xe đó chưa?", "hỏi chưa?": a progress question, not a question to pass on
ASKED_YET_RE = re.compile(r"(?i)(đã|có)\s+hỏi\b.*\bchưa\b|^\s*hỏi\s+(người bán\s+|chủ xe\s+)?chưa\b")
# "xe này", "chiếc đó": a bike the buyer points at without naming it
THIS_BIKE_RE = re.compile(r"(?i)\b(xe|chiếc|con)\s+(này|đó|kia|ấy)\b")
# "kết quả sao", "sao rồi": asking what the seller answered
RESULT_RE = re.compile(r"(?i)\bkết quả\b|\bsao rồi\b|\brồi sao\b|thế nào rồi|\bntn rồi\b|trả lời chưa|trả lời sao|nói sao")
ADDRESS_RE = re.compile(r"(?<![\w/])(\d{1,4}[A-Za-z]?(?:/\d+[A-Za-z]?)*)\s+((?:đường\s+)?\w+(?:\s+\w+){0,3})")


def find_address(text: str) -> Optional[str]:
    """"12 Lê Lợi", "45/3 Trần Phú": a house number, then a capitalised street name (not "Quận 3 được")."""
    for m in ADDRESS_RE.finditer(text):
        if re.search(r"(?i)(quận|q\.?)\s*$", text[:m.start()]):
            continue                                   # "Quận 3": a district number, not a house number
        words = m.group(2).split()
        if words[0].lower() == "đường":
            words = words[1:]
        street = []
        for w in words:
            if not w[:1].isupper():
                break
            street.append(w)
        if street:
            lead = "đường " if m.group(2).lower().startswith("đường") else ""
            return f"{m.group(1)} {lead}{' '.join(street)}"
    return None


DISTRICT_RE = re.compile(r"(?i)(?<!\w)(?:q\.?\s*|quận\s*)(\d{1,2})(?!\w)")
EXACT_WORDS = ["đường", "hẻm", "ngõ", "ngách", "số nhà", "chung cư", "tòa", "quán", "cafe", "cà phê", "chợ",
               "cây xăng", "trường", "bệnh viện", "công viên", "siêu thị", "ngã tư", "vòng xoay"]


def exact_place(place: Optional[str]) -> bool:
    """A place someone can find ("12 Lê Lợi", "cafe Highlands Phan Xích Long"), not just an area ("Q1", "Quận 1",
    "Phú Nhuận"). Live web chat: a viewing was booked "ở Q1" while the buyer asked for the exact address."""
    if not place:
        return False
    low = place.lower()      # accents matter here: folded, "quận" (district) would match "quán" (café)
    return bool(find_address(place) or any(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low) for w in EXACT_WORDS))


def parse_appointment(text: str, ref: datetime) -> tuple[Optional[str], Optional[str]]:
    """Return (ISO datetime in +07:00, place) if the text proposes a time/place."""
    f = fold(text)
    local = ref.astimezone(VN_TZ)
    day: Optional[datetime] = None
    m = re.search(r"thu\s*(\d|hai|ba|tu|nam|sau|bay)", f)
    if m and m.group(1) in _WEEKDAYS:
        target = _WEEKDAYS[m.group(1)]
        delta = (target - local.weekday()) % 7 or 7
        day = local + timedelta(days=delta)
    elif "chu nhat" in f:
        day = local + timedelta(days=(6 - local.weekday()) % 7 or 7)
    elif "ngay mai" in f or re.search(r"(?<!\w)mai(?!\w)", f):
        day = local + timedelta(days=1)

    when: Optional[str] = None
    tm = TIME_RE.search(text)
    if day is not None or tm:
        day = day or local
        hour = int(tm.group(1)) if tm else 9
        minute = int(tm.group(2)) if tm and tm.group(2) else 0
        if tm and ("chieu" in f or "toi" in f) and hour < 12:
            hour += 12
        when = day.replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat()

    place = None
    am = find_address(text)
    pm = None if am else PLACE_RE.search(text)
    dm = None if am or pm else DISTRICT_RE.search(text)
    if am:
        place = am
    elif dm:
        place = "Quận " + dm.group(1)            # "hỏi thử Q1 được ko"
    if pm:
        place = pm.group(1).strip()
        place = re.sub(r"(?i)(\s+(nhé|nhe|nha|nhá|ạ|đó|em|anh|chị|luôn|được|đc|không|ko))+$", "", place)  # "Phú Nhuận nhé"
        if re.match(r"(?i)(quan|quận|q\.?)\s*\d+", place):
            place = "Quận " + re.findall(r"\d+", place)[0]
    return when, place
