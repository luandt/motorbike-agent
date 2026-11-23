"""Builds data/livechats.jsonl: the live web-demo chats (two browser tabs, LLM mode) where the agent went wrong,
replayed as fixed scripts. Gold labels: eval/gold_live/.

w1  seller s_008  paperwork question on another seller's bike, then "hỏi thử chủ xe có giảm thêm ko" x3:
    the seller follow-up was planned but the LLM reply dropped it, while telling the buyer "đã hỏi"
w2  seller s_008  "mai hoặc mốt thì đc", "chiều mai 4h": the viewing stayed blocked, the seller was never asked
w3  seller s_009  the seller's "hẹn khách ở quận 1 được ko" was booked with no hour and without the buyer; the
    buyer's later "3h chiều mai" was "confirmed" without asking the seller
w4  seller s_009  "có xe nào giới thiệu" got the brand question again; a viewing was booked "ở Q1" while the buyer
    asked for the exact address; "hỏi chủ xe thử địa chỉ nào" sent nothing and the agent said "đã hỏi rồi"
w5  seller s_008  "oki" to the Gò Vấp bike showed the seller's own Vision 2022 instead; "rồi sao" got nothing
w6  seller s_002  opened from the Vision 2020 listing: "xe này bao lâu có giấy tờ" got "chưa có thông tin"; the budget
                "20t" was dropped twice; paperwork questions were never passed on; "hỏi chủ xe đi" went out as "đi"; the
                seller's answers ("chắc 2 tuần là có", "được") never reached the buyer, who heard "chưa có cập nhật"

Run: python data/make_livechats.py
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

CHATS = {
    "w1": ("s_008", [
        ("buyer", "Mình muốn tìm xe vision tầm 20 củ"),
        ("buyer", "chiếc 2020 thì bao lâu sẽ làm xong giấy tờ gốc"),
        ("buyer", "chờ 2 tuần thì được, lâu hơn thì chắc tìm xe khác"),
        ("buyer", "chiếc 2022 thì có giảm thêm ko"),
        ("buyer", "hỏi thử chủ xe có giảm thêm ko"),
        ("buyer", "đã hỏi thử chủ xe đó chưa?"),
    ]),
    "w2": ("s_008", [
        ("buyer", "Chào bạn, mình tìm xe tay ga tầm 25tr ở HCM, xe đẹp chút"),
        ("buyer", "chiếc vision 2022, giá fix luôn hay có giảm thêm ko"),
        ("seller", "nếu mua nhanh thì sẽ để lại giá đó"),
        ("buyer", "mai hoặc mốt thì đc"),
        ("buyer", "chiều mai 4h"),
        ("seller", "ok em, xem xe ở 25 Phan Xích Long nhé"),
        ("buyer", "ok"),
    ]),
    "w3": ("s_009", [
        ("buyer", "Chào bạn, mình tìm xe tay ga tầm 27tr ở HCM, xe đẹp chút"),
        ("buyer", "chiếc AB 2020 đc đó"),
        ("buyer", "hỏi thử giảm thêm 500k đc"),
        ("seller", "mua nhanh thì giảm đc nha"),
        ("seller", "27t"),
        ("buyer", "oki"),
        ("buyer", "chiều mai được ko"),
        ("seller", "được, hẹn khách ở quận 1 được ko"),
        ("buyer", "mấy h?"),
        ("buyer", "hỏi 3h chiều mai đc ko"),
        ("seller", "ok 3h"),
        ("buyer", "ok quận 1 được"),
        ("seller", "12 Lê Lợi nha"),
    ]),
    "w4": ("s_009", [
        ("buyer", "Mình đang tìm xe khoảng 27t"),
        ("buyer", "sg"),
        ("buyer", "có xe nào giới thiệu"),
        ("buyer", "honda"),
        ("buyer", "chiếc AB có giảm thêm ko"),
        ("seller", "nếu mua sớm thì giảm 500k"),
        ("buyer", "được, mai sắp xếp coi xe luôn được ko"),
        ("buyer", "4h chiều mai được ko"),
        ("seller", "okie, hỏi thử Q1 được ko"),
        ("buyer", "okie, gởi địa chỉ cụ thể nha"),
        ("buyer", "hỏi chủ xe thử địa chỉ nào, đường nào"),
        ("seller", "12 Lê Lợi nha em"),
    ]),
    "w5": ("s_008", [
        ("buyer", "Mình muốn tìm xe vision tầm 20 củ"),
        ("buyer", "đang ở sg"),
        ("buyer", "oki"),
        ("buyer", "đang hỏi xe ở gò vấp 21 triệu, sao lại đưa xe kia"),
        ("buyer", "rồi sao"),
    ]),
    "w6": ("s_002", [
        ("buyer", "xe này bao lâu có thể có giấy tờ đầy đủ"),
        ("buyer", "20t"),
        ("buyer", "sg"),
        ("buyer", "vision, đời 2021 trở lên"),
        ("buyer", "ko cần ngay, có thể đợi 1-2 tuần"),
        ("buyer", "chiếc vision có thể xong giấy tờ 1-2 tuần ko?"),
        ("buyer", "đã nói ở trên rồi, 20t"),
        ("buyer", "chiếc vision có thể xong giấy tờ 1-2 tuần ko?"),
        ("buyer", "hỏi chủ xe đi"),
        ("seller", "chắc 2 tuần là có"),
        ("buyer", "vậy mai coi xe được ko"),
        ("buyer", "đã hỏi chủ xe chưa?"),
        ("buyer", "ý là đã hỏi mai coi xe trước được ko?"),
        ("seller", "được"),
        ("buyer", "hỏi chưa?"),
        ("buyer", "kết quả sao"),
    ]),
}
START = datetime(2026, 11, 22, 2, 0, tzinfo=timezone.utc)      # 09:00 in Vietnam, so "chiều mai" is 23/11


def build() -> list[dict]:
    rows = []
    for n, (cid, (_, msgs)) in enumerate(CHATS.items()):
        for i, (sender, text) in enumerate(msgs):
            ts = START + timedelta(hours=n, minutes=i)
            rows.append({"conversation_id": cid, "timestamp": ts.isoformat().replace("+00:00", "Z"),
                         "sender": sender, "text": text})
    return rows


if __name__ == "__main__":
    out = Path(__file__).with_name("livechats.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build()), encoding="utf-8")
    meta_path = Path(__file__).with_name("conversations.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for cid, (seller, _) in CHATS.items():
        meta[cid] = {"buyer_id": f"b_{cid}", "seller_id": seller}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(build())} messages) and {len(CHATS)} rooms in {meta_path.name}")
