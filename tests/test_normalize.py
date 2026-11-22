from datetime import datetime, timezone

from agent import normalize as N


def test_money_units_and_ranges():
    assert [m.high for m in N.parse_money("tầm 25tr ở HCM")] == [25_000_000]
    r = N.parse_money("tối đa 25-26tr thôi")[0]
    assert (r.low, r.high) == (25_000_000, 26_000_000)
    assert N.parse_money("30 củ")[0].high == 30_000_000
    assert N.parse_money("giá 32 triệu")[0].high == 32_000_000


def test_odo_and_year():
    assert N.parse_odo("odo 19k") == 19_000
    assert N.parse_odo("odo 27 ngàn") == 27_000
    assert N.parse_year_min("đời 2020 trở lên") == 2020
    assert N.parse_year_min("đời 2019 đổ lên") == 2019


def test_accent_insensitive_matching():
    assert N.has_any("khong muon qua trung gian", ["không muốn qua trung gian"])
    assert N.find_location("ở Sài Gòn") == "HCM"
    assert [m[1] for m in N.find_models("Mình có Air Blade 2021")] == ["Air Blade"]


def test_appointment_parsing_relative_weekday():
    ref = datetime(2026, 2, 5, 2, 12, tzinfo=timezone.utc)  # Thursday 09:12 in Vietnam
    when, place = N.parse_appointment("Thứ 7 này 10h sáng ở Quận 3 được không?", ref)
    assert when.startswith("2026-02-07T10:00")
    assert place == "Quận 3"


def test_load_jsonl_numbers_messages_per_conversation(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"conversation_id":"a","timestamp":"2026-01-01T00:00:00Z","sender":"buyer","text":"hi"}\n\n'
                 'not json\n{"conversation_id":"a","timestamp":"2026-01-01T00:01:00Z","sender":"seller","text":"yo"}\n')
    msgs = N.load_jsonl([p])
    assert [m.msg_id for m in msgs] == ["a#1", "a#2"]
