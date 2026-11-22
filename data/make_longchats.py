"""Builds data/longchats.jsonl: three long three-way chats (40+ messages) where a detail the seller gives
early is asked about again ~35 messages later, after it has left the recent-message window.

l1  paperwork detail: originals held in Đồng Nai, about 3 weeks to transfer   (a live risk: pinned)
l2  price detail: lowest price 23tr5 if bought this week, with 2 helmets      (a live floor: pinned)
l3  condition detail: scratch on the left side, battery replaced in March    (not a state fact: NOT pinned,
    so this case shows the limit of pins)

Run: python data/make_longchats.py
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Small talk about the bike that a real chat has between the important messages. No prices, years or
# paperwork words, so it adds no facts; it only pushes the early messages out of the window.
FILLER = [
    ("buyer", "Xe màu gì vậy anh?"),
    ("seller", "Màu đen nhám em, nhìn khá sang"),
    ("buyer", "Lốp xe còn tốt không anh?"),
    ("seller", "Lốp sau mới thay, lốp trước còn khá"),
    ("buyer", "Xe có đi thử được không anh?"),
    ("seller", "Được em, em ghé thì anh cho chạy thử một vòng"),
    ("buyer", "Xe chạy có êm không anh?"),
    ("seller", "Êm em, máy không kêu lạ"),
    ("buyer", "Xe hay đi đường nào vậy anh?"),
    ("seller", "Chủ yếu đi làm trong thành phố thôi em"),
    ("buyer", "Có hay bảo dưỡng không anh?"),
    ("seller", "Bảo dưỡng định kỳ ở hãng, có sổ đàng hoàng"),
    ("buyer", "Đèn xe còn sáng không anh?"),
    ("seller", "Sáng em, đèn pha mới vệ sinh"),
    ("buyer", "Phanh còn ăn không anh?"),
    ("seller", "Ăn tốt em, mới chỉnh lại"),
    ("buyer", "Cốp xe rộng không anh, để vừa nón không?"),
    ("seller", "Để vừa một nón bảo hiểm cả đầu em"),
    ("buyer", "Xe có chìa khóa dự phòng không anh?"),
    ("seller", "Có đủ hai chìa em"),
    ("buyer", "Yên xe có bị rách không anh?"),
    ("seller", "Yên còn đẹp, không rách"),
    ("buyer", "Gương chiếu hậu còn zin không anh?"),
    ("seller", "Còn zin cả hai bên"),
    ("buyer", "Xe đi có hao xăng không anh?"),
    ("seller", "Không hao lắm em, đi làm cả tuần đổ một lần"),
    ("buyer", "Anh bán vì sao vậy?"),
    ("seller", "Anh đổi xe khác cho tiện chở con đi học"),
    ("buyer", "Dạ em hiểu rồi"),
    ("seller", "Em cứ hỏi thoải mái nha"),
    ("buyer", "Xe có gắn thêm đồ chơi gì không anh?"),
    ("seller", "Có gắn thêm baga sau thôi em"),
    ("buyer", "Dạ vậy cũng tiện"),
    ("seller", "Ừ em"),
]

CHATS = {
    "l1": {"start": datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc), "opening": [
        ("buyer", "Chào anh, con Vision bên anh còn không ạ? Em ngân sách tầm 22tr, ở HCM, tìm xe tay ga đi làm"),
        ("seller", "Còn em. Nói trước là giấy tờ đang chờ rút hồ sơ gốc ở Đồng Nai, khoảng 3 tuần nữa mới sang tên được"),
        ("buyer", "Dạ em không gấp, đợi được"),
    ], "late": [
        ("buyer", "À anh ơi, vậy chính xác khoảng bao lâu nữa thì sang tên được ạ?"),
        ("seller", "Như anh nói lúc đầu đó em"),
        ("buyer", "Hồ sơ gốc đang nằm ở đâu vậy anh?"),
    ]},
    "l2": {"start": datetime(2026, 3, 3, 9, 0, tzinfo=timezone.utc), "opening": [
        ("buyer", "Vision này còn không anh? Em ngân sách tầm 23tr, ở HCM"),
        ("seller", "Còn em, giá thấp nhất anh để 23tr5 nếu em lấy trong tuần, tặng kèm 2 nón bảo hiểm"),
        ("buyer", "Dạ để em suy nghĩ thêm"),
    ], "late": [
        ("buyer", "Anh nhắc lại giúp em giá thấp nhất anh để là bao nhiêu và có kèm gì không ạ?"),
        ("seller", "Như anh nói từ đầu nha em"),
        ("buyer", "Ok em lấy giá đó, em muốn xem xe trực tiếp"),
    ]},
    "l3": {"start": datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc), "opening": [
        ("buyer", "Chào anh, con Lead này còn không ạ? Em ngân sách tầm 26tr, ở HCM"),
        ("seller", "Còn em. Xe có một vết trầy bên hông trái, bình ắc quy mới thay tháng 3"),
        ("buyer", "Dạ em biết rồi"),
    ], "late": [
        ("buyer", "Anh ơi xe có bị trầy chỗ nào không anh, em quên mất?"),
        ("seller", "Anh nói ở trên rồi đó em"),
        ("buyer", "Ắc quy thay lúc nào vậy anh?"),
    ]},
}


def build() -> list[dict]:
    rows = []
    for cid, c in CHATS.items():
        msgs = c["opening"] + FILLER + c["late"]
        for i, (sender, text) in enumerate(msgs):
            ts = c["start"] + timedelta(minutes=i)
            rows.append({"conversation_id": cid, "timestamp": ts.isoformat().replace("+00:00", "Z"),
                         "sender": sender, "text": text})
    return rows


if __name__ == "__main__":
    out = Path(__file__).with_name("longchats.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build()), encoding="utf-8")
    print(f"wrote {out} ({len(build())} messages)")
