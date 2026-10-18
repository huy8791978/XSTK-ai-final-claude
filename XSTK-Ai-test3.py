"""
XSTK Tutor - trợ lí AI học Xác suất thống kê (Streamlit).
Chạy thử : streamlit run app.py
Triển khai: Streamlit Community Cloud, Main file path = app.py
"""
import copy
import hashlib
import io
import json
import os
import random
import re
import threading
import time
import datetime as dt
from collections import Counter, OrderedDict, deque
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types
from PIL import Image, ImageOps

# ───────────────────────── CẤU HÌNH ─────────────────────────
MODEL = "gemini-3.5-flash-lite"
# Cách 1 (khuyên dùng): đặt GEMINI_API_KEY trong Streamlit Cloud > Settings > Secrets.
# Cách 2: dán khóa (dạng AIza...) vào dòng dưới. Chỉ làm vậy khi repo GitHub ở chế độ PRIVATE:
# khóa nằm trong repo public sẽ bị lộ và bị thu hồi.
EMBEDDED_KEY = "DAN_API_KEY_VAO_DAY"

# ── Hạn mức gói free của AI Studio cho Gemini 3.5 Flash Lite: RPM 15 · TPM 250k · RPD 500.
# Ngưỡng dưới đây đặt thấp hơn hạn mức thật để chừa biên an toàn (đồng hồ Google lệch, app khác dùng chung key).
RPM_CAP = 12            # số lượt gọi tối đa trong 60 giây, toàn hệ thống
TPM_CAP = 200_000       # token (đầu vào + đầu ra ước lượng) tối đa trong 60 giây
RPD_CAP = 450           # số lượt gọi tối đa mỗi ngày (reset 0h giờ Thái Bình Dương, khoảng 14-15h giờ Việt Nam)
USER_DAILY = 30         # lượt AI tối đa mỗi người mỗi ngày
IP_DAILY = 90           # lượt AI tối đa mỗi địa chỉ mạng mỗi ngày (cả lớp có thể dùng chung wifi)
QUEUE_MAX = 40          # giây tối đa một yêu cầu chịu xếp hàng trước khi báo bận
CACHE_TTL = 24 * 3600   # bộ nhớ chung giữ kết quả 24 giờ
POOL_SIZE = 3           # mỗi (chủ đề, mức độ) chỉ sinh tối đa 3 bộ đề, người sau dùng lại
HEAD = {"minimal": 0, "medium": 6000, "high": 12000}  # chỗ chừa cho token "suy nghĩ" (tính vào max_output_tokens)
PROFILE_DIR = Path("profiles")  # shortcut: đĩa của Streamlit Cloud bị xoá khi app khởi động lại, nên có nút tải/khôi phục hồ sơ

CONCEPTS = [
    "Phép thử, không gian mẫu và biến cố",
    "Xác suất của biến cố (định nghĩa cổ điển)",
    "Biến cố độc lập, biến cố xung khắc",
    "Xác suất có điều kiện",
    "Biến ngẫu nhiên rời rạc",
    "Kì vọng và phương sai",
    "Phân phối nhị thức",
    "Phân phối Poisson",
    "Biến ngẫu nhiên liên tục, hàm mật độ",
    "Phân phối chuẩn",
    "Mẫu số liệu ghép nhóm: trung bình, trung vị, tứ phân vị",
    "Phương sai và độ lệch chuẩn của mẫu",
    "Khoảng tin cậy",
    "Giả thuyết thống kê và mức ý nghĩa",
    "Hệ số tương quan và hồi quy tuyến tính",
]
THEOREMS = [
    "Công thức cộng xác suất",
    "Công thức nhân xác suất",
    "Công thức xác suất toàn phần",
    "Công thức Bayes",
    "Tính chất của kì vọng và phương sai",
    "Công thức xác suất của phân phối nhị thức",
    "Xấp xỉ Poisson cho phân phối nhị thức",
    "Quy tắc 68-95-99,7 của phân phối chuẩn",
    "Bất đẳng thức Chebyshev",
    "Luật số lớn",
    "Định lí giới hạn trung tâm",
    "Công thức khoảng tin cậy cho trung bình",
    "Tiêu chuẩn kiểm định giả thuyết (z, t)",
    "Công thức đường hồi quy bình phương tối thiểu",
]
LEVELS = ["Cơ bản", "Trung bình", "Nâng cao"]
ERR = [
    "Hiểu sai khái niệm", "Áp dụng sai công thức", "Sai tính toán",
    "Thiếu hoặc sai lập luận", "Đọc sai đề hoặc dữ kiện",
    "Trình bày thiếu chặt chẽ", "Khác",
]

# ───────────────────────── PROMPT ─────────────────────────
BASE = (
    "Bạn là gia sư Xác suất thống kê (XSTK) cho học sinh, sinh viên Việt Nam. "
    "Luôn trả lời bằng tiếng Việt, ngắn gọn, chính xác, thân thiện, không dùng emoji. "
    "Công thức toán viết bằng LaTeX: $...$ trong dòng, $$...$$ cho dòng riêng. "
)
JSON_RULE = (
    r' Chỉ trả về MỘT đối tượng JSON hợp lệ, không có chữ nào bên ngoài, không dùng khối mã. '
    r'Trong chuỗi JSON, mọi dấu gạch chéo ngược của LaTeX phải viết đôi (ví dụ "\\frac{1}{2}"). '
    r'Xuống dòng trong chuỗi dùng \n.'
)
SOLVE_FLOW = (
    "Quy trình giải toán 4 bước: (1) Tìm hiểu đề: phân biệt cái đã cho, cái phải tìm, kí hiệu hoá; "
    "(2) Xây dựng chương trình giải: quy về bài toán đơn giản quen thuộc, chọn kiến thức gần nhất; "
    "(3) Trình bày lời giải; (4) Kiểm tra, nhìn lại, rút phương pháp, mở rộng. "
)
SYS_HINT = BASE + SOLVE_FLOW + (
    "Nhiệm vụ: đưa CHIẾN LƯỢC giải bài theo các bước 1, 2 và gợi ý bước 4 (cách tự kiểm tra). "
    "Nêu: cái đã cho, cái cần tìm, dạng bài, kiến thức/công thức cần dùng, cách chia nhỏ bài toán. "
    "TUYỆT ĐỐI không đưa đáp án, không tính ra kết quả số, không thay số hoàn chỉnh vào công thức, "
    "không viết lời giải từng bước. Kết thúc bằng 1-2 câu hỏi gợi mở để học sinh tự làm."
)
SYS_SOL = BASE + SOLVE_FLOW + (
    "Nhiệm vụ: viết BÀI GIẢI CHI TIẾT theo 4 bước, rõ ràng từng phép biến đổi, in đậm kết quả cuối, "
    "cuối cùng nêu 1-2 sai lầm thường gặp của dạng bài này."
)
SYS_GRADE = BASE + (
    "Bạn chấm bài làm XSTK của học sinh (có thể là ảnh chụp chữ viết tay hoặc PDF: hãy đọc/OCR cẩn thận). "
    "Tự giải đề trước để biết kết quả đúng, nhưng CHỈ chấm điểm và chỉ ra chỗ sai: KHÔNG viết lại lời giải đúng, "
    "KHÔNG nêu kết quả cuối cùng đúng. Với mỗi lỗi, nói sai ở bước nào, vì sao, và gợi hướng sửa bằng một câu hỏi "
    "hoặc nhắc kiến thức. Chấm thang 10, ghi nhận phần đúng. Nếu không đọc được bài làm thì diem = 0 và nói rõ. "
    'Trường "loai" của loi_sai phải là một trong: ' + "; ".join(ERR) + "." + JSON_RULE +
    ' Cấu trúc: {"van_ban_chep_lai":"nội dung bài làm đọc được", "diem":số, "nhan_xet":"2-4 câu", '
    '"buoc":[{"ten":"...","dung":true,"ghi_chu":"..."}], "loi_sai":[{"loai":"...","mo_ta":"..."}], '
    '"lo_hong":["kiến thức còn thiếu"], "chu_de":"tên chủ đề ngắn"}'
)
_PRACTICE = (
    "Soạn bài tập XSTK bám sát chương trình, số liệu cụ thể, đủ dữ kiện, giải được, có lời giải đúng. "
)
FOCUS = {
    ("concept", "cung_co"): "Củng cố khái niệm: nhận diện khái niệm trong trường hợp đơn giản có tính đặc trưng; thể hiện khái niệm trong các ngữ cảnh khác nhau và trong mối liên hệ logic với khái niệm khác.",
    ("concept", "van_dung"): "Vận dụng khái niệm trong các tình huống gián tiếp, phức hợp và giải quyết vấn đề thực tiễn.",
    ("theorem", "cung_co"): "Củng cố định lí: nhận biết giả thiết, kết luận, điều kiện áp dụng; vận dụng định lí hoặc công thức trong trường hợp đơn giản có tính đặc trưng.",
    ("theorem", "van_dung"): "Vận dụng định lí để giải quyết vấn đề toán học hoặc vấn đề gắn với thực tiễn.",
}


def sys_practice(kind, proc, stage):
    if kind == "mc":
        body = (
            "Soạn đúng 5 câu trắc nghiệm 4 phương án. Phương án nhiễu phản ánh sai lầm phổ biến. Vị trí đáp án đúng phân bố "
            "ngẫu nhiên giữa A-D. 'chien_luoc' theo bước 1-2 của quy trình giải, TUYỆT ĐỐI không tiết lộ đáp án hay kết quả số. "
            "'sai_lam' chỉ gồm các phương án sai. Kiểm tra kĩ đáp án." + JSON_RULE +
            ' Cấu trúc: {"cau":[{"de":"...","lua_chon":{"A":"..","B":"..","C":"..","D":".."},"dap_an":"B",'
            '"chien_luoc":"...","loi_giai":"lời giải chi tiết nêu rõ đáp án","sai_lam":{"A":{"loai":"(một trong: '
            + "; ".join(ERR) + ')","mo_ta":"vì sao học sinh hay chọn nhầm"}}}]}')
    else:
        body = (
            "Soạn đúng 5 câu tự luận tăng dần độ khó (2 cơ bản, 2 trung bình, 1 nâng cao hoặc gắn thực tiễn). 'chien_luoc' theo "
            "bước 1-2 của quy trình giải, TUYỆT ĐỐI không tiết lộ kết quả. 'tieu_chi' là thang điểm 10 tham khảo." + JSON_RULE +
            ' Cấu trúc: {"cau":[{"de":"...","chien_luoc":"...","loi_giai":"lời giải chi tiết 4 bước","tieu_chi":"..."}]}')
    return BASE + _PRACTICE + "Mục tiêu: " + FOCUS[(proc, stage)] + " " + body


SYS_ANALYZE = BASE + (
    "Nhiệm vụ: PHÂN TÍCH ĐỀ theo bước 1 của quy trình giải toán, KHÔNG giải. Trình bày: **Dữ kiện đã cho**; **Cái phải tìm**; "
    "**Điều kiện và ràng buộc**; **Diễn tả bằng kí hiệu** (đặt biến cố hoặc biến ngẫu nhiên, viết điều cần tính thành công thức "
    "nhưng chưa tính); **Dạng bài và chủ đề**. Tuyệt đối không nêu kết quả."
)
SYS_ALT = BASE + SOLVE_FLOW + (
    "Nhiệm vụ (bước 4: kiểm tra và nghiên cứu lời giải): (1) **Kiểm tra kết quả** bằng một cách độc lập (ước lượng, trường hợp đặc "
    "biệt, điều kiện hợp lí như xác suất thuộc [0; 1]); (2) **Cách giải khác** nếu có, trình bày gọn; (3) **Phương pháp rút ra** cho "
    "dạng bài; (4) **Khai thác**: nêu một bài toán đặc biệt và một khái quát hoá của đề. Nếu có bài làm của học sinh, nêu thêm điều "
    "học sinh nên chú ý khi tự kiểm tra."
)
SYS_SIMILAR = BASE + (
    "Soạn đúng 3 bài toán dựa trên đề đã cho, KHÔNG kèm lời giải hay đáp án: 1 bài tương tự (đổi số liệu, cùng cách giải), 1 bài đặc "
    "biệt (trường hợp riêng), 1 bài khái quát hoá hoặc nâng cao nhẹ. Đề đủ dữ kiện, giải được." + JSON_RULE +
    ' Cấu trúc: {"bai":[{"muc":"Tương tự","de":"..."}]}'
)


def sys_situation(proc):
    if proc == "concept":
        goal = ("TÌNH HUỐNG TRẢI NGHIỆM để học sinh tự cảm nhận khái niệm trước khi học định nghĩa: tình huống đời sống hoặc thí nghiệm "
                "ngẫu nhiên đơn giản có số liệu cụ thể, kèm 3 câu hỏi dẫn dắt từ quan sát đến nhận ra dấu hiệu bản chất. "
                "KHÔNG nêu định nghĩa, không dùng thuật ngữ chưa học.")
    else:
        goal = ("TÌNH HUỐNG TRẢI NGHIỆM khơi gợi định lí/công thức: học sinh quan sát giả thiết, tính hoặc kiểm chứng trên số liệu cụ thể, "
                "rồi dự đoán mối quan hệ (kết luận). Kèm 3 câu hỏi dẫn dắt. KHÔNG phát biểu định lí hay công thức.")
    return BASE + "Soạn một " + goal + JSON_RULE + ' Cấu trúc: {"tinh_huong":"...","cau_hoi":["...","...","..."],"goi_y":"một gợi ý quan sát ngắn"}'


SYS_OBS = BASE + (
    "Học sinh vừa quan sát tình huống trải nghiệm và ghi nhận xét hoặc dự đoán. Nhận xét ngắn (3-5 câu): ghi nhận điểm đúng, chỉ ra "
    "điều chưa đủ hoặc chưa chính xác, gợi mở hướng tới dấu hiệu bản chất (với khái niệm) hoặc giả thiết và kết luận (với định lí). "
    "KHÔNG nêu định nghĩa hay định lí đầy đủ."
)


def sys_frame(proc):
    if proc == "concept":
        return BASE + (
            "Nhiệm vụ: lập KHUNG ĐỊNH NGHĨA khái niệm (bước hình thành định nghĩa): dấu hiệu bản chất, thuật ngữ và kí hiệu then chốt, "
            "định nghĩa chuẩn, ví dụ và phản ví dụ." + JSON_RULE +
            ' Cấu trúc: {"dinh_nghia":"...","dau_hieu":["..."],"thuat_ngu":[{"ky_hieu":"...","y_nghia":"..."}],'
            '"vi_du":["..."],"phan_vi_du":["..."],"luu_y":"..."}')
    return BASE + (
        "Nhiệm vụ: lập KHUNG ĐỊNH LÍ (bước hình thành định lí): phát biểu, tách rõ giả thiết và kết luận, điều kiện áp dụng, luận cứ, "
        "ý tưởng và cách chứng minh (ngắn gọn, chặt chẽ), cấu trúc logic (mệnh đề đảo, trường hợp đặc biệt)." + JSON_RULE +
        ' Cấu trúc: {"phat_bieu":"...","gia_thiet":["..."],"ket_luan":["..."],"dieu_kien":"...","luan_cu":["..."],'
        '"y_tuong":"...","chung_minh":"...","cau_truc":"..."}')


ORAL_CONCEPT = (
    "Thứ tự 5 câu hỏi (học khái niệm): (1) phát biểu định nghĩa và dấu hiệu bản chất; (2) thuật ngữ, kí hiệu then chốt "
    "hoặc phân biệt với khái niệm dễ nhầm; (3) nhận diện qua ví dụ và phản ví dụ đơn giản; (4) mối liên hệ logic với khái "
    "niệm khác; (5) vận dụng trong một tình huống thực tiễn."
)
ORAL_THEOREM = (
    "Thứ tự 5 câu hỏi (học định lí, công thức): (1) phát biểu định lí/công thức; (2) chỉ rõ giả thiết, kết luận, điều kiện "
    "áp dụng; (3) luận cứ và ý tưởng chứng minh; (4) cấu trúc logic (trường hợp đặc biệt, mệnh đề đảo); (5) vận dụng "
    "vào bài toán hoặc thực tiễn."
)
SYS_ORAL_GRADE = BASE + (
    "Chấm phần vấn đáp lí thuyết XSTK dựa trên bản ghi. Câu chưa trả lời hoặc trả lời sai hoàn toàn tính 0. "
    "Chỉ ra lỗ hổng kiến thức và sai lầm cụ thể của học sinh. " 'Trường "loai" của loi_sai là một trong: '
    + "; ".join(ERR) + "." + JSON_RULE +
    ' Cấu trúc: {"diem":số 0-10,"nhan_xet":"2-4 câu","tung_cau":[{"cau":"tóm tắt câu hỏi","diem":số,"nhan_xet":"..."}],'
    '"loi_sai":[{"loai":"...","mo_ta":"..."}],"lo_hong":["..."],"goi_y_on_tap":["..."],"chu_de":"tên chủ đề"}'
)
SYS_MAP = BASE + (
    "Tạo bản đồ tư duy cho một chủ đề XSTK. Gốc là tên chủ đề. 5-6 nhánh cấp 1 (ví dụ: Khái niệm; Công thức và tính chất; "
    "Dạng bài và phương pháp; Sai lầm thường gặp; Ví dụ và ứng dụng; Liên hệ chủ đề khác), mỗi nhánh 2-4 nút con, "
    "có thể thêm cấp 3. Viết tiếng Việt CÓ ĐẦY ĐỦ DẤU. Tên nút tối đa 8 từ, văn bản thuần, không LaTeX, không dùng dấu nháy kép hay dấu gạch chéo ngược." + JSON_RULE +
    ' Cấu trúc: {"ten":"gốc","con":[{"ten":"...","con":[{"ten":"..."}]}]}'
)
SYS_PLAN = BASE + (
    "Bạn nhận thống kê hồ sơ học XSTK của một học sinh (JSON). Chỉ dựa vào số liệu đó, không bịa. Trả lời dưới 350 từ, gồm: "
    "**Nhận định** (2-4 gạch đầu dòng nêu quy luật lỗi sai xuyên suốt); **Ưu tiên ôn** (tối đa 3 chủ đề, vì sao); "
    "**Kế hoạch 7 ngày** (mỗi ngày một việc cụ thể, dùng các chức năng: Chấm bài, Luyện tập, Vấn đáp, Bản đồ tư duy); "
    "**Cách tự kiểm tra tiến bộ**. Nếu dữ liệu còn ít, nói rõ và đề xuất làm thêm."
)

S = st.session_state


# ───────────────────────── BỘ ĐIỀU TIẾT HẠN MỨC (DÙNG CHUNG MỌI NGƯỜI) ─────────────────────────
def _day():
    try:
        tz = ZoneInfo("America/Los_Angeles")
    except Exception:
        tz = dt.timezone(dt.timedelta(hours=-8))
    return dt.datetime.now(tz).date().isoformat()


class Shared:
    """Một đối tượng duy nhất cho cả tiến trình (st.cache_resource): đếm lượt gọi, hàng đợi, bộ nhớ đệm, kho đề.
    shortcut: nằm trong RAM, mất khi app khởi động lại (khi đó bộ đếm về 0, nên ngưỡng mới chừa biên an toàn)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls, self.toks = deque(), deque()
        self.day, self.rpd, self.block_until, self.used = _day(), 0, 0.0, {}
        self.cache, self.pool, self.flight = OrderedDict(), OrderedDict(), {}

    def _roll(self, now):
        d = _day()
        if d != self.day:
            self.day, self.rpd, self.used = d, 0, {}
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        while self.toks and now - self.toks[0][0] > 60:
            self.toks.popleft()

    def reserve(self, who, est):
        """-> ("ok", mục_token) | ("wait", giây) | ("day", 0) | ("user", loại). Chỉ tính lượt khi trả về ok."""
        now = time.time()
        with self.lock:
            self._roll(now)
            if self.rpd >= RPD_CAP:
                return "day", 0
            for k, cap in who:
                if self.used.get(k, 0) >= cap:
                    return "user", k[0]
            if now < self.block_until:
                return "wait", self.block_until - now
            if len(self.calls) >= RPM_CAP:
                return "wait", self.calls[0] + 60 - now
            if self.toks and sum(t for _, t in self.toks) + est > TPM_CAP:
                return "wait", self.toks[0][0] + 60 - now
            self.calls.append(now)
            entry = [now, est]
            self.toks.append(entry)
            self.rpd += 1
            for k, _ in who:
                self.used[k] = self.used.get(k, 0) + 1
            return "ok", entry

    def settle(self, entry, actual):
        with self.lock:
            entry[1] = actual  # thay ước lượng bằng số token thật Google báo về

    def penalize(self, seconds, day=False):
        with self.lock:
            self.block_until = time.time() + seconds
            if day:
                self.rpd = RPD_CAP

    def used_of(self, k):
        with self.lock:
            self._roll(time.time())
            return self.used.get(k, 0)

    def cget(self, k):
        with self.lock:
            v = self.cache.get(k)
            return copy.deepcopy(v[1]) if v and time.time() - v[0] < CACHE_TTL else None

    def cput(self, k, val):
        with self.lock:
            self.cache[k] = (time.time(), copy.deepcopy(val))
            self.cache.move_to_end(k)
            while len(self.cache) > 400:
                self.cache.popitem(last=False)

    def flight_lock(self, k):
        with self.lock:
            return self.flight.setdefault(k, threading.Lock())

    def pool_get(self, k):
        with self.lock:
            return list(self.pool.get(k, []))

    def pool_add(self, k, item):
        with self.lock:
            self.pool.setdefault(k, []).append(item)
            while len(self.pool) > 200:
                self.pool.popitem(last=False)


@st.cache_resource
def shared():
    return Shared()


def _ip():
    try:
        return st.context.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    except Exception:
        return ""


def est_tokens(system, content, msgs):
    n = len(system) / 2.5  # tiếng Việt tốn token hơn tiếng Anh nên chia 2.5 (ước lượng nhiều lên cho an toàn)
    items = [m["content"] for m in msgs] if msgs else (content if isinstance(content, list) else [content])
    for p in items:
        if isinstance(p, str):
            n += len(p) / 2.5
        elif getattr(p, "text", None):
            n += len(p.text) / 2.5
        elif getattr(p, "inline_data", None):
            n += 6000 if p.inline_data.mime_type == "application/pdf" else 1500
    return int(n)


def _digest(system, content, tag):
    h = hashlib.sha256()
    for x in (MODEL, system, tag):
        h.update(x.encode())
    for p in content if isinstance(content, list) else [content]:
        if isinstance(p, str):
            h.update(p.encode())
        elif getattr(p, "text", None) is not None:
            h.update(p.text.encode())
        else:
            h.update(p.inline_data.data)
    return h.hexdigest()


def memo(key, fn):
    """Có kết quả cũ thì dùng lại (không tốn lượt). Nhiều người hỏi cùng lúc một thứ thì chỉ gọi AI một lần."""
    sh = shared()
    v = sh.cget(key)
    if v is not None:
        return v
    with sh.flight_lock(key):
        v = sh.cget(key)
        if v is None:
            v = fn()
            if v is not None:
                sh.cput(key, v)
        sh.flight.pop(key, None)
        return v


# ───────────────────────── GỌI AI (GEMINI) ─────────────────────────
def _key():
    """Thứ tự ưu tiên: biến môi trường (GitHub secret chạy qua Codespaces/Actions) -> Secrets của Streamlit Cloud -> khóa dán trong code."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            v = str(st.secrets[name]).strip()
        except Exception:
            v = ""
        if v:
            return v
    return EMBEDDED_KEY


def _has_key():
    k = _key()
    return k != EMBEDDED_KEY and len(k) >= 20 and " " not in k


@st.cache_resource
def _client(key):
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))


def ask(system, content=None, msgs=None, max_tokens=3000, think="minimal", json_mode=False):
    """Một lượt gọi thật tới Gemini, đi qua bộ điều tiết. Trả về text hoặc None (đã báo lỗi, không lộ khóa)."""
    if not _has_key():
        st.error("Chưa cấu hình API key. Quản trị viên cần thêm khóa vào Secrets hoặc vào mã nguồn.")
        return None
    sh, ip = shared(), _ip()
    who = [(("u", (S.get("user") or "?").lower()), USER_DAILY)] + ([(("ip", ip), IP_DAILY)] if ip else [])
    est = est_tokens(system, content, msgs) + max_tokens + HEAD[think]
    state, h = sh.reserve(who, est)
    if state == "wait":
        with st.spinner("Hệ thống đang đông, bạn đang được xếp hàng..."):
            end = time.time() + QUEUE_MAX
            while state == "wait" and time.time() < end:
                time.sleep(min(max(h, 0.3), 1.5))
                state, h = sh.reserve(who, est)
    if state == "wait":
        st.error("Hệ thống đang rất đông. Hãy thử lại sau ít phút.")
        return None
    if state == "day":
        st.error("Hệ thống đã dùng hết hạn mức AI của hôm nay. Hạn mức làm mới khoảng 14-15h (giờ Việt Nam).")
        return None
    if state == "user":
        st.error(f"Bạn đã dùng hết {USER_DAILY} lượt AI hôm nay. Hãy quay lại vào ngày mai." if h == "u"
                 else "Mạng bạn đang dùng đã hết lượt AI hôm nay. Hãy quay lại vào ngày mai.")
        return None
    if msgs:
        content = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                 parts=[types.Part(text=m["content"])]) for m in msgs]  # lượt cuối luôn là "user"
    cfg = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=max_tokens + HEAD[think],
        thinking_config=types.ThinkingConfig(thinking_level=think),
        response_mime_type="application/json" if json_mode else None)
    try:
        r = _client(_key()).models.generate_content(model=MODEL, contents=content, config=cfg)
    except errors.APIError as e:
        code = getattr(e, "code", None)
        if code == 429:
            daily = "perday" in str(e).lower().replace(" ", "")
            sh.penalize(3600 if daily else 45, day=daily)
            st.error("AI đã hết hạn mức hôm nay." if daily else "AI đang quá tải. Hãy đợi khoảng 1 phút rồi thử lại.")
        elif code in (400, 401, 403):
            st.error(f"Yêu cầu bị từ chối (mã {code}). Kiểm tra API key hoặc tệp tải lên.")
        else:
            st.error(f"Lỗi kết nối AI (mã {code}). Hãy thử lại.")
    except Exception as e:
        st.error(f"Lỗi kết nối AI ({type(e).__name__}). Hãy thử lại.")
    else:
        used = getattr(getattr(r, "usage_metadata", None), "total_token_count", None)
        if used:
            sh.settle(h, used)
        if r.text:
            return r.text.strip()
        st.error("AI không trả lời (có thể do bộ lọc an toàn hoặc bị cắt). Hãy diễn đạt lại và thử lại.")
    return None


def ask_cached(system, content, max_tokens=3000, think="minimal"):
    """Văn bản trả lời, dùng lại kết quả của người khác nếu đề y hệt."""
    return memo(_digest(system, content, f"t{think}{max_tokens}"),
                lambda: ask(system, content, max_tokens=max_tokens, think=think))


def _fix(o):
    """JSON nuốt \\f \\t \\r \\b của LaTeX (\\frac, \\text, \\right, \\bar): khôi phục lại."""
    if isinstance(o, str):
        return o.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\t", "\\t").replace("\r", "\\r")
    if isinstance(o, list):
        return [_fix(i) for i in o]
    if isinstance(o, dict):
        return {k: _fix(v) for k, v in o.items()}
    return o


def ask_json(system, content, max_tokens=3000, think="minimal", cache=True):
    def run():
        for _ in range(2):
            t = ask(system, content, max_tokens=max_tokens, think=think, json_mode=True)
            if t is None:
                return None
            try:
                return _fix(json.loads(t[t.index("{"): t.rindex("}") + 1], strict=False))
            except ValueError:
                continue
        st.error("AI trả về dữ liệu không đọc được. Hãy thử lại.")
        return None

    return memo(_digest(system, content, f"j{think}{max_tokens}"), run) if cache else run()


# ───────────────────────── FILE / OCR ─────────────────────────
def _T(s):
    return types.Part(text=s)


def fblocks(files):
    """Ảnh -> JPEG thu nhỏ (ảnh điện thoại rất nặng); PDF gửi nguyên. Gemini đọc chữ viết tay (OCR)."""
    out = []
    if files and len(files) > 3:
        st.warning("Chỉ dùng 3 tệp đầu tiên để tiết kiệm lượt AI.")
    for name, data in (files or [])[:3]:
        if name.lower().endswith(".pdf"):
            if len(data) > 10_000_000:
                st.warning(f"{name} lớn hơn 10 MB, bỏ qua.")
                continue
            out.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception:
            st.warning(f"Không đọc được ảnh {name}.")
            continue
        im.thumbnail((1800, 1800))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=85)
        out.append(types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg"))
    return out


def build(prob_txt="", prob_files=None, work_txt="", work_files=None, extra=""):
    c = []
    pf = fblocks(prob_files)
    if prob_txt.strip() or pf:
        c.append(_T("ĐỀ BÀI:"))
        if prob_txt.strip():
            c.append(_T(prob_txt.strip()))
        c += pf
    wf = fblocks(work_files)
    if work_txt.strip() or wf:
        c.append(_T("BÀI LÀM CỦA HỌC SINH:"))
        if work_txt.strip():
            c.append(_T(work_txt.strip()))
        c += wf
    if extra:
        c.append(_T(extra))
    return c


# ───────────────────────── HỒ SƠ ─────────────────────────
def _ppath(name):
    return PROFILE_DIR / (hashlib.sha1(name.strip().lower().encode()).hexdigest()[:12] + ".json")


def load_profile(name):
    try:
        p = json.loads(_ppath(name).read_text("utf-8"))
        if isinstance(p.get("events"), list):
            return p
    except (OSError, ValueError, AttributeError):
        pass
    return {"name": name, "created": dt.date.today().isoformat(), "events": []}


def save_profile(p):
    try:
        PROFILE_DIR.mkdir(exist_ok=True)
        _ppath(p["name"]).write_text(json.dumps(p, ensure_ascii=False), "utf-8")
    except OSError:
        pass  # shortcut: lưu đĩa thất bại thì vẫn giữ trong phiên và cho tải về


def num(x):
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def errs(x):
    out = []
    for e in x if isinstance(x, list) else []:
        if isinstance(e, str):
            e = {"mo_ta": e}
        if isinstance(e, dict):
            out.append({"loai": e.get("loai") if e.get("loai") in ERR else "Khác", "mo_ta": str(e.get("mo_ta", ""))[:300]})
    return out


def log(mode, topic, score, errors, gaps):
    P = S.profile
    gaps = [str(g)[:200] for g in gaps if g] if isinstance(gaps, list) else []
    P["events"].append({"t": dt.datetime.now().isoformat(timespec="seconds"), "mode": mode,
                        "chu_de": str(topic)[:80] or "Chưa phân loại", "diem": round(score, 1),
                        "loi_sai": errs(errors), "lo_hong": gaps})
    P["events"] = P["events"][-500:]
    save_profile(P)


def log_grade(mode, g):
    log(mode, g.get("chu_de") or "Chưa phân loại", num(g.get("diem")), g.get("loi_sai"), g.get("lo_hong"))


@st.dialog("Hồ sơ học XSTK", width="large")
def profile_dialog():
    P = S.profile
    ev = [e for e in P.get("events", []) if isinstance(e, dict)]
    st.markdown(f"**{P['name']}** · {len(ev)} lượt học")
    if not ev:
        st.info("Chưa có dữ liệu. Hãy chấm một bài, làm bài tập hoặc vấn đáp để hồ sơ bắt đầu ghi nhận.")
        return
    sc = [num(e.get("diem")) for e in ev]
    avg, rec, prev = sum(sc) / len(sc), sc[-5:], sc[-10:-5]
    ra = sum(rec) / len(rec)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số lượt", len(ev))
    c2.metric("Điểm trung bình", f"{avg:.1f}")
    c3.metric("5 lượt gần nhất", f"{ra:.1f}", f"{ra - sum(prev) / len(prev):+.1f}" if prev else None)
    st.markdown("**Diễn tiến điểm**")
    st.line_chart(pd.DataFrame({"Điểm": sc}), color="#0F5C5A")

    all_err = [x for e in ev for x in (e.get("loi_sai") or []) if isinstance(x, dict)]
    by_type = Counter(x.get("loai", "Khác") for x in all_err)
    if by_type:
        st.markdown("**Lỗi sai theo nhóm**")
        st.bar_chart(pd.Series(by_type, name="Số lần").sort_values(ascending=False), color="#B4531F")

    df = pd.DataFrame({"Chủ đề": [str(e.get("chu_de", "?")) for e in ev], "Điểm": sc})
    topic = (df.groupby("Chủ đề")["Điểm"].agg(["mean", "count"]).round(1).reset_index()
             .rename(columns={"mean": "Điểm TB", "count": "Lượt"}).sort_values("Điểm TB"))
    st.markdown("**Điểm theo chủ đề** (thấp nhất ở trên)")
    st.dataframe(topic, hide_index=True)

    gaps = Counter(str(g).strip() for e in ev for g in (e.get("lo_hong") or []) if g)
    if gaps:
        st.markdown("**Lỗ hổng kiến thức hay gặp**")
        for g, n in gaps.most_common(8):
            st.markdown(f"- {g}" + (f" (×{n})" if n > 1 else ""))
    if all_err:
        st.markdown("**Sai lầm gần đây**")
        for x in all_err[-5:][::-1]:
            st.markdown(f"- *{x.get('loai', 'Khác')}*: {x.get('mo_ta', '')}")

    st.divider()
    if st.button("Đề xuất phương án học cải thiện", icon=":material/route:", type="primary", key="b_plan"):
        data = {"so_luot": len(ev), "diem_tb": round(avg, 1), "diem_5_gan_nhat": round(ra, 1),
                "loi_theo_nhom": dict(by_type), "chu_de_yeu_nhat": topic.head(3).to_dict("records"),
                "lo_hong_hay_gap": dict(gaps.most_common(8)),
                "sai_lam_gan_day": [x.get("mo_ta", "") for x in all_err[-6:]]}
        with st.spinner("Đang phân tích hồ sơ..."):
            S.r_plan = ask_cached(SYS_PLAN, json.dumps(data, ensure_ascii=False), max_tokens=1500)
    if S.get("r_plan"):
        st.markdown(S.r_plan)


# ───────────────────────── GIAO DIỆN CHUNG ─────────────────────────
def _bell_uri():
    bars = [10, 22, 42, 70, 98, 98, 70, 42, 22, 10]
    r = "".join(f"<rect x='{14 + i * 37}' y='{150 - h}' width='30' height='{h}' rx='4' fill='#0F5C5A' fill-opacity='.13'/>"
                for i, h in enumerate(bars))
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 160'>" + r +
           "<path d='M6 150 C95 150 120 22 200 22 C280 22 305 150 394 150' fill='none' stroke='#B4531F' "
           "stroke-opacity='.55' stroke-width='3' stroke-linecap='round'/></svg>")
    return 'url("data:image/svg+xml,' + quote(svg) + '")'


# Biểu tượng xác suất thống kê: biểu đồ cột và đường cong chuông
ICON = ("<svg class='ico' viewBox='0 0 40 28' width='1.15em' height='.82em' role='img' aria-label='biểu đồ phân phối chuẩn'>"
        "<rect x='2' y='18' width='6' height='9' rx='1.5' fill='#0F5C5A' fill-opacity='.35'/>"
        "<rect x='10' y='11' width='6' height='16' rx='1.5' fill='#0F5C5A' fill-opacity='.55'/>"
        "<rect x='18' y='4' width='6' height='23' rx='1.5' fill='#0F5C5A'/>"
        "<rect x='26' y='11' width='6' height='16' rx='1.5' fill='#0F5C5A' fill-opacity='.55'/>"
        "<rect x='34' y='18' width='4' height='9' rx='1.5' fill='#0F5C5A' fill-opacity='.35'/>"
        "<path d='M1 26 C10 26 12 3 21 3 C30 3 31 26 40 26' fill='none' stroke='#B4531F' stroke-width='2.2' "
        "stroke-linecap='round'/></svg>")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{--paper:#F5F1E8;--card:#FBF9F4;--ink:#1F2A27;--muted:#5E6B66;--line:#D9D2C3;--teal:#0F5C5A;--clay:#B4531F;--amber:#8A6A12}
.stApp{background-color:var(--paper);background-image:radial-gradient(#D9D2C3 1px,transparent 1px);background-size:24px 24px;color:var(--ink);font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',sans-serif}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1100px;padding:4.6rem 1.1rem 5rem}
#MainMenu,footer{visibility:hidden}
h1,h2,h3,h4{font-family:'Newsreader',Georgia,serif!important;color:var(--ink);letter-spacing:-.015em}
h1{font-size:clamp(2rem,5vw,3.2rem)!important;line-height:1.1!important;font-weight:600!important;margin:.2rem 0 .7rem!important;padding:0!important}
h4{font-size:1.25rem!important;font-weight:600!important}
.eyebrow{font-size:.72rem;line-height:1.4;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:600}
.brand{display:flex;align-items:center;gap:.7rem}
.brand-mark{display:inline-flex;align-items:center;justify-content:center;width:2.3rem;height:2.3rem;border-radius:10px;background:var(--teal);color:#fff;font-family:'Newsreader',Georgia,serif;font-size:1.35rem;font-weight:600}
.hero{position:relative;padding:1.1rem 0 .4rem;margin-bottom:.6rem}
.hero::after{content:"";position:absolute;right:0;bottom:-6px;width:min(44%,430px);height:170px;background:__BELL__ no-repeat right bottom/contain;pointer-events:none}
.hero>*{position:relative;z-index:1}
.hero h1{max-width:30rem}
.ico{display:inline-block;vertical-align:-.08em;margin:0 .05em}
.lead{color:var(--muted);font-size:1.05rem;line-height:1.65;max-width:32rem;margin:.2rem 0 0}
.syms{color:var(--teal);opacity:.55;font-family:'Newsreader',Georgia,serif;font-size:1.05rem;letter-spacing:.5em;margin-top:.9rem}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px}
div[data-testid="stElementContainer"]:has(.stButton),div[data-testid="stElementContainer"]:has(.stDownloadButton),div[data-testid="stElementContainer"]:has(.stPopover),div[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]){width:100%!important}
.stButton,.stDownloadButton,.stPopover,div[data-testid="stFormSubmitButton"]{width:100%!important}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button,[data-testid="stPopoverButton"]{width:100%!important;min-height:46px;border-radius:12px;font-weight:500;justify-content:center;text-align:center;transition:transform .15s ease,border-color .15s ease,background .15s ease}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"],.stFormSubmitButton>button[kind="primaryFormSubmit"],.stFormSubmitButton>button[data-testid="stBaseButton-primaryFormSubmit"]{background:var(--teal)!important;border-color:var(--teal)!important;color:#fff!important}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{background:#0B4947!important;border-color:#0B4947!important}
.procdesc{min-height:3.9em;margin-top:.45rem;color:var(--muted);font-size:.86rem;line-height:1.55}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);border-color:var(--teal)}
.stButton>button:active{transform:scale(.98)}
div[data-testid="stButtonGroup"]{width:100%;flex-wrap:wrap;gap:.35rem}
div[data-testid="stLayoutWrapper"]:has([data-testid="stPopover"]),div[data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]){width:100%!important}
div[data-testid="stPopover"]>div,button[data-testid="stPopoverButton"]{width:100%!important}
div[data-testid="stButtonGroup"] [role="radiogroup"]{width:100%!important;max-width:100%!important;display:flex;flex-wrap:wrap;gap:.35rem}
div[data-testid="stButtonGroup"] button{flex:1 1 0;min-width:max-content;min-height:44px;border-radius:10px;justify-content:center}
:focus-visible{outline:2px solid var(--teal)!important;outline-offset:2px}
.step-head{display:flex;align-items:center;gap:1rem;margin:.4rem 0 1rem}
.step-head .num{flex:none;width:3rem;height:3rem;border-radius:50%;border:2px solid var(--teal);color:var(--teal);display:flex;align-items:center;justify-content:center;font-family:'Newsreader',Georgia,serif;font-size:1.6rem;font-weight:600;background:var(--card)}
.step-head .t{font-family:'Newsreader',Georgia,serif;font-size:1.45rem;font-weight:600;line-height:1.2}
.step-head .d{color:var(--muted);font-size:.95rem;line-height:1.5;margin-top:.15rem}
.score{display:flex;align-items:baseline;gap:.3rem;font-family:'Newsreader',Georgia,serif;margin:.1rem 0 .5rem}
.score b{font-size:3.6rem;line-height:1;font-weight:600}.score span{color:var(--muted);font-size:1.2rem}
.score.teal b{color:var(--teal)}.score.amber b{color:var(--amber)}.score.clay b{color:var(--clay)}
.foot{margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid var(--line);text-align:center;color:var(--muted);font-size:.88rem;line-height:1.6}
.foot b{color:var(--teal);font-family:'Newsreader',Georgia,serif;letter-spacing:.3em;font-weight:600}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.block-container{animation:rise .45s ease both}
@media (max-width:640px){.hero::after{width:78%;opacity:.45}.block-container{padding-top:4.2rem}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>""".replace("__BELL__", _bell_uri())

PROCS = {
    "solve": ("Quy trình giải toán", ":material/calculate:",
              "Tìm hiểu đề, xây dựng chương trình giải, trình bày và kiểm tra lời giải.",
              ["1 · Tìm hiểu đề", "2 · Chương trình giải", "3 · Lời giải", "4 · Kiểm tra"]),
    "concept": ("Quy trình học khái niệm", ":material/lightbulb:",
                "Trải nghiệm, hình thành định nghĩa, củng cố và vận dụng khái niệm.",
                ["1 · Trải nghiệm", "2 · Hình thành", "3 · Củng cố", "4 · Vận dụng"]),
    "theorem": ("Quy trình học định lí", ":material/menu_book:",
                "Trải nghiệm, hình thành định lí, củng cố và vận dụng định lí.",
                ["1 · Trải nghiệm", "2 · Hình thành", "3 · Củng cố", "4 · Vận dụng"]),
}


def hero():
    st.markdown(
        '<div class="hero">'
        f'<h1>Hiểu bản chất {ICON} giải đúng từng bài toán</h1>'
        '<p class="lead">Học khái niệm, học định lí và giải toán theo quy trình 4 bước, có trợ giảng AI đồng hành.</p>'
        '<div class="syms" aria-hidden="true">Σ μ σ P(A)</div></div>', unsafe_allow_html=True)


def step_head(n, title, desc):
    st.markdown(f'<div class="step-head"><div class="num">{n}</div><div><div class="t">{title}</div>'
                f'<div class="d">{desc}</div></div></div>', unsafe_allow_html=True)


def mark(proc, i):
    S.setdefault("done", set()).add((proc, i))


def keep_text(label, key, **kw):
    """Ô nhập giữ nguyên nội dung khi chuyển bước (widget không hiển thị sẽ bị Streamlit xoá trạng thái)."""
    wk = f"w_{key}"
    if wk not in S:  # đã có trạng thái widget thì không truyền value (tránh cảnh báo của Streamlit)
        kw["value"] = S.get(key, "")
    v = st.text_area(label, key=wk, **kw)
    S[key] = v
    return v


def _sync_files(key, wk):
    S[key] = [(f.name, f.getvalue()) for f in S.get(wk) or []]


def keep_files(label, key):
    wk = f"w_{key}{S.get('ver', 0)}"
    up = st.file_uploader(label, type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True, key=wk,
                          on_change=_sync_files, args=(key, wk))
    if not up and S.get(key):
        st.caption("Đã lưu: " + ", ".join(n for n, _ in S[key]))
    return S.get(key, [])


def show_score(d):
    col = "teal" if d >= 7 else "amber" if d >= 5 else "clay"
    st.markdown(f'<div class="score {col}"><b>{d:g}</b><span>/10</span></div>', unsafe_allow_html=True)


def bullets(title, items):
    if items:
        st.markdown(f"**{title}**")
        for x in items if isinstance(items, list) else [items]:
            st.markdown(f"- {x}")


def show_issues(g):
    if g.get("loi_sai"):
        st.markdown("**Sai lầm phát hiện**")
        for e in errs(g["loi_sai"]):
            st.markdown(f"- *{e['loai']}*: {e['mo_ta']}")
    bullets("Lỗ hổng kiến thức", g.get("lo_hong"))


def show_grade(g):
    show_score(num(g.get("diem")))
    st.markdown(str(g.get("nhan_xet", "")))
    for s in g.get("buoc") or []:
        if isinstance(s, dict):
            tag = ":green[Đúng]" if s.get("dung") else ":red[Cần sửa]"
            st.markdown(f"{tag} · **{s.get('ten', '')}**: {s.get('ghi_chu', '')}")
    show_issues(g)
    if g.get("van_ban_chep_lai"):
        with st.expander("Nội dung AI đọc được từ bài làm (OCR)"):
            st.markdown(str(g["van_ban_chep_lai"]))


def result(title, body_fn):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        body_fn()


# ───────────────────────── QUY TRÌNH GIẢI TOÁN ─────────────────────────
def need_problem():
    if S.get("prob", "").strip() or S.get("prob_f"):
        return True
    st.warning("Hãy quay lại bước 1 để nhập hoặc tải đề bài.")
    return False


def prob_preview():
    with st.expander("Đề bài đang làm"):
        st.markdown(S.get("prob") or "(Đề bài ở dạng ảnh hoặc PDF đã tải lên.)")


def use_problem(txt):
    """Nạp một bài tập tương tự làm đề mới và quay về bước 1 (chạy trong callback nên được phép đặt lại widget)."""
    S["prob"] = S["w_prob"] = txt
    S["work"] = S["w_work"] = ""
    S["prob_f"], S["work_f"] = [], []
    S["ver"] = S.get("ver", 0) + 1
    for k in ("r_ana", "r_hint", "r_grade", "r_sol", "r_alt"):
        S.pop(k, None)
    S["cur_solve"] = S["w_step_solve"] = 0
    S["done"] = {d for d in S.get("done", set()) if d[0] != "solve"}


def solve_1():
    step_head(1, "Tìm hiểu đề", "Phân biệt cái đã cho, cái phải tìm; diễn tả đề bằng công thức, kí hiệu.")
    with st.container(border=True):
        keep_text("Nhập đề bài", "prob", height=150, max_chars=4000)
        keep_files("Hoặc tải ảnh/PDF đề bài", "prob_f")
        if st.button("Phân tích đề", icon=":material/search:", type="primary", key="b_ana"):
            if need_problem():
                with st.spinner("Đang phân tích đề..."):
                    S.r_ana = ask_cached(SYS_ANALYZE, build(S.get("prob", ""), S.get("prob_f", [])), max_tokens=1500, think="medium")
                if S.r_ana:
                    mark("solve", 0)
    if S.get("r_ana"):
        result("Phân tích đề", lambda: st.markdown(S.r_ana))


def solve_2():
    step_head(2, "Xây dựng chương trình giải", "Quy về bài toán đơn giản quen thuộc, chọn kiến thức gần gũi nhất với dữ kiện.")
    if not need_problem():
        return
    prob_preview()
    if st.button("Gợi ý chiến lược giải (không có đáp án)", icon=":material/lightbulb:", type="primary", key="b_hint"):
        with st.spinner("Đang xây dựng chiến lược..."):
            S.r_hint = ask_cached(SYS_HINT, build(S.get("prob", ""), S.get("prob_f", [])), max_tokens=1800, think="medium")
        if S.r_hint:
            mark("solve", 1)
    if S.get("r_hint"):
        result("Chiến lược giải", lambda: st.markdown(S.r_hint))


def solve_3():
    step_head(3, "Trình bày lời giải", "Trình bày lại lời giải sau khi điều chỉnh những chỗ cần thiết. AI chỉ chấm điểm, không giải hộ.")
    if not need_problem():
        return
    prob_preview()
    with st.container(border=True):
        keep_text("Gõ bài làm của bạn (không bắt buộc)", "work", height=150, max_chars=4000)
        keep_files("Tải ảnh chụp hoặc PDF bài làm (AI đọc bằng OCR)", "work_f")
        if st.button("Chấm bài", icon=":material/task_alt:", type="primary", key="b_grade"):
            if not (S.get("work", "").strip() or S.get("work_f")):
                st.warning("Hãy gõ hoặc tải bài làm của bạn.")
            else:
                with st.spinner("Đang đọc bài làm và chấm điểm..."):
                    g = ask_json(SYS_GRADE, build(S.get("prob", ""), S.get("prob_f", []), S.get("work", ""), S.get("work_f", [])), max_tokens=3500, think="medium")
                if g:
                    if g != S.get("r_grade"):  # bấm chấm lại y hệt: dùng kết quả cũ, không ghi hồ sơ lần nữa
                        log_grade("tutor", g)
                    S.r_grade = g
                    mark("solve", 2)
    if S.get("r_grade"):
        result("Kết quả chấm", lambda: show_grade(S.r_grade))


def solve_4():
    step_head(4, "Kiểm tra và nghiên cứu lời giải",
              "Kiểm tra kết quả và lập luận, rút phương pháp, tìm cách giải khác, khai thác và đề xuất bài tương tự.")
    if not need_problem():
        return
    prob_preview()
    c1, c2, c3 = st.columns(3)
    pb = build(S.get("prob", ""), S.get("prob_f", []))
    if c1.button("Hiện bài giải chi tiết", icon=":material/menu_book:", type="primary", key="b_sol"):
        with st.spinner("Đang viết bài giải..."):
            S.r_sol = ask_cached(SYS_SOL, pb, max_tokens=3500, think="medium")
        if S.r_sol:
            mark("solve", 3)
    if c2.button("Cách giải khác và rút phương pháp", icon=":material/alt_route:", key="b_alt"):
        with st.spinner("Đang kiểm tra và khai thác bài toán..."):
            S.r_alt = ask_cached(SYS_ALT, build(S.get("prob", ""), S.get("prob_f", []), S.get("work", ""), S.get("work_f")), max_tokens=3000, think="medium")
        if S.r_alt:
            mark("solve", 3)
    if c3.button("Tạo bài tập tương tự", icon=":material/library_add:", key="b_sim"):
        with st.spinner("Đang soạn bài tập tương tự..."):
            S.r_sim = ask_json(SYS_SIMILAR, pb, max_tokens=2000, think="medium")
        if S.r_sim:
            mark("solve", 3)
    if S.get("r_sol"):
        result("Bài giải chi tiết", lambda: st.markdown(S.r_sol))
    if S.get("r_alt"):
        result("Kiểm tra, cách giải khác và khai thác", lambda: st.markdown(S.r_alt))
    sim = (S.get("r_sim") or {}).get("bai")
    if sim:
        with st.container(border=True):
            st.markdown("#### Bài tập tương tự")
            for i, b in enumerate(x for x in sim if isinstance(x, dict) and x.get("de")):
                st.markdown(f"**{b.get('muc', 'Bài tập')}.** {b['de']}")
                st.button("Làm bài này", icon=":material/play_arrow:", key=f"b_use{i}", on_click=use_problem, args=(b["de"],))


# ───────────────────────── LUYỆN TẬP, VẤN ĐÁP, SƠ ĐỒ (DÙNG CHO KHÁI NIỆM VÀ ĐỊNH LÍ) ─────────────────────────
def ok_mc(q):
    return (isinstance(q, dict) and q.get("de") and isinstance(q.get("lua_chon"), dict)
            and all(x in q["lua_chon"] for x in "ABCD") and q.get("dap_an") in list("ABCD"))


def toggle(k):
    S[k] = not S.get(k)


def get_practice(proc, stage, kind, topic, level):
    """Kho đề dùng chung: mỗi (quy trình, bước, loại, chủ đề, mức độ) chỉ sinh tối đa POOL_SIZE bộ; người sau nhận bộ chưa làm."""
    sh, key = shared(), (proc, stage, kind, topic.strip().lower(), level)
    seen = S.setdefault("pool_seen", set())
    with sh.flight_lock(("gen", key)):
        pool = sh.pool_get(key)
        fresh = [i for i in range(len(pool)) if (key, i) not in seen]
        if fresh:
            i = random.choice(fresh)
        elif len(pool) >= POOL_SIZE:
            i = random.randrange(len(pool))
        else:
            d = ask_json(sys_practice(kind, proc, stage), f"Chủ đề: {topic}. Mức độ: {level}.",
                         max_tokens=5500, think="medium", cache=False) or {}
            qs = [q for q in (d.get("cau") or []) if (ok_mc(q) if kind == "mc" else isinstance(q, dict) and q.get("de"))][:5]
            if not qs:
                return None
            sh.pool_add(key, qs)
            i = len(sh.pool_get(key)) - 1
        seen.add((key, i))
        return copy.deepcopy(sh.pool_get(key)[i])


def mc_card(pre, i, q, topic):
    k = f"pq_{pre}{i}"
    with st.container(border=True):
        st.markdown(f"**Câu {i + 1}.** {q['de']}")
        pick = st.radio("Chọn đáp án", list("ABCD"), index=None, key=k, label_visibility="collapsed",
                        format_func=lambda x: f"{x}. {q['lua_chon'][x]}")
        a, b, c = st.columns(3)
        a.button("Gợi ý chiến lược", icon=":material/lightbulb:", key=f"b_{k}_h", on_click=toggle, args=(f"{k}_h",))
        if b.button("Chấm bài", icon=":material/task_alt:", type="primary", key=f"b_{k}_g"):
            if pick is None:
                st.warning("Hãy chọn một đáp án.")
            else:
                S[f"{k}_g"] = pick
                if not S.get(f"{k}_l"):  # chỉ ghi hồ sơ một lần cho mỗi câu
                    S[f"{k}_l"] = 1
                    m = (q.get("sai_lam") or {}).get(pick)
                    log("practice", topic, 10.0 if pick == q["dap_an"] else 0.0,
                        [] if pick == q["dap_an"] else [m if m else {"loai": "Khác", "mo_ta": "Chọn sai đáp án"}], [])
        c.button("Hiện lời giải", icon=":material/menu_book:", key=f"b_{k}_s", on_click=toggle, args=(f"{k}_s",))
        gp = S.get(f"{k}_g")
        if gp:
            if gp == q["dap_an"]:
                st.success(f"Chính xác. Bạn đã chọn {gp}.")
            else:
                m = (q.get("sai_lam") or {}).get(gp)
                why = m.get("mo_ta", "") if isinstance(m, dict) else str(m or "")
                st.error(f"Chưa đúng (bạn chọn {gp}). {why}")
        if S.get(f"{k}_h"):
            st.info(q.get("chien_luoc", ""))
        if S.get(f"{k}_s"):
            st.success(f"**Đáp án: {q['dap_an']}**\n\n{q.get('loi_giai', '')}")


def es_card(pre, i, q, topic):
    k = f"pq_{pre}{i}"
    with st.container(border=True):
        st.markdown(f"**Câu {i + 1}.** {q['de']}")
        ans = st.text_area("Bài làm của bạn", key=f"{k}_a", height=130, max_chars=4000)
        with st.expander("Hoặc tải ảnh/PDF bài làm"):
            up = st.file_uploader("Ảnh/PDF", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True,
                                  key=f"{k}_f", label_visibility="collapsed")
        a, b, c = st.columns(3)
        a.button("Gợi ý chiến lược", icon=":material/lightbulb:", key=f"b_{k}_h", on_click=toggle, args=(f"{k}_h",))
        if b.button("Chấm bài", icon=":material/task_alt:", type="primary", key=f"b_{k}_g"):
            if not (ans.strip() or up):
                st.warning("Hãy nhập hoặc tải bài làm.")
            else:
                ref = f"ĐÁP ÁN THAM KHẢO VÀ THANG ĐIỂM (chỉ để chấm, không tiết lộ):\n{q.get('loi_giai', '')}\n{q.get('tieu_chi', '')}"
                fl = [(f.name, f.getvalue()) for f in up]
                with st.spinner("Đang chấm..."):
                    g = ask_json(SYS_GRADE, build(q["de"], None, ans, fl, ref), max_tokens=3500, think="medium")
                if g:
                    g.setdefault("chu_de", topic)
                    if g != S.get(f"{k}_g"):
                        log_grade("practice", g)
                    S[f"{k}_g"] = g
        c.button("Hiện lời giải", icon=":material/menu_book:", key=f"b_{k}_s", on_click=toggle, args=(f"{k}_s",))
        if S.get(f"{k}_g"):
            show_grade(S[f"{k}_g"])
        if S.get(f"{k}_h"):
            st.info(q.get("chien_luoc", ""))
        if S.get(f"{k}_s"):
            st.success(q.get("loi_giai", ""))


def practice_panel(proc, stage, topic, step_idx):
    pre = f"{proc}_{stage}"
    st.markdown("**Luyện tập**")
    level = st.radio("Mức độ", LEVELS, index=1, horizontal=True, key=f"w_lv_{pre}")
    c1, c2 = st.columns(2)
    for col, kind, label, ic in ((c1, "mc", "Tạo 5 câu trắc nghiệm", ":material/checklist:"),
                                 (c2, "es", "Tạo 5 câu tự luận", ":material/edit_note:")):
        if col.button(label, icon=ic, type="primary" if kind == "mc" else "secondary", key=f"b_gen_{pre}_{kind}"):
            for kk in [x for x in S if x.startswith(f"pq_{pre}_{kind}")]:
                del S[kk]
            with st.spinner("Đang chuẩn bị bộ bài tập..."):
                qs = get_practice(proc, stage, kind, topic, level)
            if qs:
                S.setdefault("r_pq", {})[f"{pre}_{kind}"] = {"topic": topic, "qs": qs}
                mark(proc, step_idx)
            else:
                st.error("Không tạo được bài tập. Hãy thử lại.")
    for kind, title, card in (("mc", "Trắc nghiệm", mc_card), ("es", "Tự luận", es_card)):
        d = S.get("r_pq", {}).get(f"{pre}_{kind}")
        if d:
            st.markdown(f"#### {title} · {d['topic']}")
            for i, q in enumerate(d["qs"]):
                card(f"{pre}_{kind}", i, q, d["topic"])


def oral_first(topic, kind):
    """Câu 1 lấy từ mẫu có sẵn, không tốn lượt AI."""
    if kind.startswith("Khái"):
        return f"Câu 1/5. Hãy phát biểu định nghĩa của «{topic}» bằng lời của bạn và nêu các dấu hiệu bản chất của khái niệm."
    return f"Câu 1/5. Hãy phát biểu định lí hoặc công thức quan trọng nhất của chủ đề «{topic}»."


def oral_sys(O):
    n = sum(m["role"] == "assistant" for m in O["msgs"])
    plan = ORAL_CONCEPT if O["kind"].startswith("Khái") else ORAL_THEOREM
    return BASE + (
        f"Bạn là giám khảo vấn đáp lí thuyết XSTK, chủ đề: {O['topic']}. {plan} "
        f"Học sinh vừa trả lời câu {n}/5. Hãy nhận xét 1-2 câu (đúng, chưa đủ hay sai ở đâu, có gợi mở), KHÔNG giảng đáp án "
        f"đầy đủ, rồi hỏi duy nhất câu {n + 1}/5 theo đúng thứ tự trên, bắt đầu câu hỏi bằng \"Câu {n + 1}/5.\". "
        "Không cho điểm số trong lúc hỏi."
    )


def oral_panel(proc, topic, step_idx):
    kind = "Khái niệm" if proc == "concept" else "Định lí, công thức"
    ok = f"oral_{proc}"
    st.markdown("**Vấn đáp lí thuyết**")
    st.caption("Giám khảo AI hỏi 5 câu theo đúng quy trình. Cuối buổi AI chấm điểm và chỉ ra lỗ hổng kiến thức, sai lầm của bạn.")
    if st.button("Bắt đầu vấn đáp mới", icon=":material/record_voice_over:", type="primary", key=f"b_o_start_{proc}"):
        S[ok] = {"topic": topic, "kind": kind, "res": None,
                 "msgs": [{"role": "user", "content": "Hãy bắt đầu vấn đáp."},
                          {"role": "assistant", "content": oral_first(topic, kind)}]}
    O = S.get(ok)
    if not O:
        return
    st.caption(f"Chủ đề: {O['topic']}")
    for m in O["msgs"][1:]:
        st.chat_message(m["role"]).markdown(m["content"])
    done = bool(O.get("res"))
    answered = sum(m["role"] == "user" for m in O["msgs"]) - 1
    p = st.chat_input("Nhập câu trả lời của bạn...", disabled=done or answered >= 5, max_chars=1200)
    if p:
        O["msgs"].append({"role": "user", "content": p})
        st.chat_message("user").markdown(p)
        answered += 1
        if answered < 5:  # câu trả lời thứ 5 không cần nhận xét riêng, để dành lượt AI cho phần chấm điểm
            with st.chat_message("assistant"):
                with st.spinner("Giám khảo đang nhận xét..."):
                    r = ask(oral_sys(O), msgs=O["msgs"], max_tokens=600)
                if r:
                    st.markdown(r)
                    O["msgs"].append({"role": "assistant", "content": r})
                else:
                    O["msgs"].pop()
                    answered -= 1
    if answered >= 5 and not done:
        st.info("Bạn đã trả lời đủ 5 câu. Bấm “Kết thúc và chấm điểm” để nhận kết quả.")
    if not done and answered >= 1 and st.button("Kết thúc và chấm điểm", icon=":material/fact_check:",
                                                  type="primary", key=f"b_o_end_{proc}"):
        tr = "\n".join(f"{'Giám khảo' if m['role'] == 'assistant' else 'Học sinh'}: {m['content']}" for m in O["msgs"][1:])
        with st.spinner("Đang chấm phần vấn đáp..."):
            res = ask_json(SYS_ORAL_GRADE, f"Chủ đề: {O['topic']}\nDạng: {O['kind']}\nBản ghi:\n{tr}",
                           max_tokens=2500, think="medium")
        if res:
            O["res"] = res
            res.setdefault("chu_de", O["topic"])
            log_grade("oral", res)
            mark(proc, step_idx)
            st.rerun()
    if done:
        res = O["res"]
        with st.container(border=True):
            st.markdown("#### Kết quả vấn đáp")
            show_score(num(res.get("diem")))
            st.markdown(str(res.get("nhan_xet", "")))
            for c in res.get("tung_cau") or []:
                if isinstance(c, dict):
                    st.markdown(f"- **{c.get('cau', '')}** ({num(c.get('diem')):g}/10): {c.get('nhan_xet', '')}")
            show_issues(res)
            bullets("Gợi ý ôn tập", res.get("goi_y_on_tap"))


# ───────────────────────── NHÚNG HTML TƯƠNG TÁC (SƠ ĐỒ TƯ DUY, BÀN PHÍM TOÁN) ─────────────────────────
def embed(html, height):
    """st.iframe ở Streamlit mới; components.html ở bản cũ."""
    if hasattr(st, "iframe"):
        return st.iframe(html, height=height)
    import streamlit.components.v1 as components
    return components.html(html, height=height)


MINDMAP_HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&display=swap');
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:transparent;color:#1F2A27;font-family:'Be Vietnam Pro',-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,'Noto Sans',sans-serif}
#vp{position:relative;height:100%;overflow:hidden;border:1px solid #D9D2C3;border-radius:16px;background:#FBF9F4;cursor:grab;user-select:none;-webkit-user-select:none}
#vp.drag{cursor:grabbing}
#stage{position:absolute;left:0;top:0;width:max-content;padding:28px;transform-origin:0 0}
#links{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}
.row{display:flex;align-items:center}
.kids{display:flex;flex-direction:column;gap:10px;margin-left:54px}
.node{position:relative;z-index:1;max-width:230px;padding:9px 14px;border-radius:12px;font-size:14px;line-height:1.4;font-weight:500;border:1.5px solid transparent}
.node.has{cursor:pointer}
.node:focus-visible{outline:2px solid #0F5C5A;outline-offset:2px}
.d0{background:#1F2A27;color:#fff;font-size:16px;font-weight:600;max-width:240px}
.d1{color:#fff;font-weight:600}
.badge{display:inline-block;margin-left:8px;padding:0 7px;border-radius:99px;background:rgba(255,255,255,.28);font-size:12px;line-height:18px}
.d2 .badge{background:rgba(0,0,0,.1)}
.tb{position:absolute;right:10px;top:10px;display:flex;gap:6px;z-index:5;flex-wrap:wrap;justify-content:flex-end;max-width:85%}
.tb button{min-width:40px;height:40px;padding:0 12px;border:1px solid #D9D2C3;border-radius:10px;background:#fff;color:#1F2A27;font:inherit;font-size:13px;cursor:pointer}
.tb button:hover{border-color:#0F5C5A;color:#0F5C5A}
.hint{position:absolute;left:14px;right:14px;bottom:10px;font-size:12px;line-height:1.4;color:#5E6B66;pointer-events:none}
</style></head><body>
<div id="vp">
 <div class="tb">
  <button id="zi" aria-label="Phóng to">+</button><button id="zo" aria-label="Thu nhỏ">&minus;</button>
  <button id="fit">Vừa khung</button><button id="all">Mở hết</button><button id="less">Thu gọn</button><button id="pan" aria-pressed="false" style="display:none" title="Bật để kéo bản đồ theo mọi hướng bằng một ngón tay">Kéo tự do</button>
 </div>
 <div id="stage"><svg id="links"></svg><div id="tree"></div></div>
 <div class="hint">Chạm vào nhánh để mở hoặc thu gọn. Kéo để di chuyển. Chụm hai ngón hoặc giữ Ctrl và cuộn để phóng to.</div>
</div>
<script>
(function(){
var DATA=__DATA__;
var PAL=["#0F5C5A","#B4531F","#4F6D3A","#8A6A12","#2F5D8C","#8A3B4E"];
var vp=document.getElementById('vp'),stage=document.getElementById('stage'),svg=document.getElementById('links'),tree=document.getElementById('tree');
var tx=0,ty=0,s=1,nodes=[],n=0,NS='http://www.w3.org/2000/svg';
function rgba(h,a){var v=parseInt(h.slice(1),16);return 'rgba('+(v>>16&255)+','+(v>>8&255)+','+(v&255)+','+a+')'}
function build(d,depth,color,parent){
  if(n>=80||!d||typeof d!=='object')return null;n++;
  var row=document.createElement('div');row.className='row';
  var box=document.createElement('div');box.className='node d'+Math.min(depth,2);
  box.appendChild(document.createTextNode(String(d.ten||'?')));
  if(depth===1)box.style.background=color;
  else if(depth>=2){box.style.background=rgba(color,.14);box.style.borderColor=rgba(color,.5)}
  box._c=color;box._p=parent;box._d=depth;nodes.push(box);row.appendChild(box);
  var kids=Array.isArray(d.con)?d.con:[];
  if(kids.length&&depth<3){
    var ke=document.createElement('div');ke.className='kids';
    kids.forEach(function(k,i){var r=build(k,depth+1,depth===0?PAL[i%PAL.length]:color,box);if(r)ke.appendChild(r)});
    if(ke.children.length){
      row.appendChild(ke);box._k=ke;box.classList.add('has');box.setAttribute('role','button');box.tabIndex=0;box.setAttribute('aria-expanded','true');
      var b=document.createElement('span');b.className='badge';b.textContent='+'+ke.children.length;b.style.display='none';box.appendChild(b);box._b=b;
    }
  }
  return row;
}
function setC(b,c){b._k.style.display=c?'none':'';b._b.style.display=c?'':'none';b.setAttribute('aria-expanded',String(!c))}
function apply(){stage.style.transform='translate('+tx+'px,'+ty+'px) scale('+s+')'}
function links(){
  svg.innerHTML='';svg.setAttribute('width',stage.offsetWidth);svg.setAttribute('height',stage.offsetHeight);
  var sr=stage.getBoundingClientRect();
  nodes.forEach(function(b){
    if(!b._p||b.offsetParent===null)return;
    var a=b._p.getBoundingClientRect(),c=b.getBoundingClientRect();
    var x1=(a.right-sr.left)/s,y1=(a.top+a.height/2-sr.top)/s,x2=(c.left-sr.left)/s,y2=(c.top+c.height/2-sr.top)/s,dx=(x2-x1)/2;
    var p=document.createElementNS(NS,'path');
    p.setAttribute('d','M'+x1+' '+y1+' C'+(x1+dx)+' '+y1+' '+(x2-dx)+' '+y2+' '+x2+' '+y2);
    p.setAttribute('fill','none');p.setAttribute('stroke',b._c);p.setAttribute('stroke-width','2.4');
    p.setAttribute('stroke-opacity','.75');p.setAttribute('stroke-linecap','round');svg.appendChild(p);
  });
}
function fit(){
  var w=stage.offsetWidth,h=stage.offsetHeight,vw=vp.clientWidth,vh=vp.clientHeight;
  if(!vw||!vh)return;
  s=Math.max(.35,Math.min(vw/w,(vh-24)/h,1.1));tx=(vw-w*s)/2;ty=Math.max(8,(vh-h*s)/2);apply();
}
function relayout(){links();fit()}
function toggle(b){setC(b,b._k.style.display!=='none');requestAnimationFrame(links)}
function zoomAt(cx,cy,ns){ns=Math.max(.3,Math.min(2.5,ns));tx=cx-(cx-tx)*ns/s;ty=cy-(cy-ty)*ns/s;s=ns;apply()}
tree.appendChild(build(DATA,0,PAL[0],null));
nodes.forEach(function(b){if(b._k&&b._d>=1)setC(b,true)});
var pts=new Map(),down=null,moved=0,d0=1,s0=1;
vp.addEventListener('pointerdown',function(e){
  if(e.target.closest('.tb'))return;
  vp.setPointerCapture(e.pointerId);pts.set(e.pointerId,{x:e.clientX,y:e.clientY});
  if(pts.size===1){down=e.target;moved=0;vp.classList.add('drag')}
  else if(pts.size===2){var v=Array.from(pts.values());d0=Math.hypot(v[0].x-v[1].x,v[0].y-v[1].y)||1;s0=s;down=null}
});
vp.addEventListener('pointermove',function(e){
  var p=pts.get(e.pointerId);if(!p)return;
  if(pts.size===1){var dx=e.clientX-p.x,dy=e.clientY-p.y;tx+=dx;ty+=dy;moved+=Math.abs(dx)+Math.abs(dy);p.x=e.clientX;p.y=e.clientY;apply()}
  else if(pts.size===2){
    p.x=e.clientX;p.y=e.clientY;var v=Array.from(pts.values()),r=vp.getBoundingClientRect();
    zoomAt((v[0].x+v[1].x)/2-r.left,(v[0].y+v[1].y)/2-r.top,s0*Math.hypot(v[0].x-v[1].x,v[0].y-v[1].y)/d0);
  }
});
function end(e){
  if(!pts.has(e.pointerId))return;var was=pts.size;pts.delete(e.pointerId);
  if(pts.size===0){
    vp.classList.remove('drag');
    if(was===1&&moved<6&&down&&down.closest){var b=down.closest('.node.has');if(b)toggle(b)}
    down=null;
  }
}
vp.addEventListener('pointerup',end);vp.addEventListener('pointercancel',end);
vp.addEventListener('wheel',function(e){
  if(!(e.ctrlKey||e.metaKey))return;e.preventDefault();
  var r=vp.getBoundingClientRect();zoomAt(e.clientX-r.left,e.clientY-r.top,s*(e.deltaY<0?1.12:.89));
},{passive:false});
tree.addEventListener('keydown',function(e){
  if(e.key!=='Enter'&&e.key!==' ')return;var b=e.target.closest&&e.target.closest('.node.has');
  if(b){e.preventDefault();toggle(b)}
});
document.getElementById('zi').onclick=function(){zoomAt(vp.clientWidth/2,vp.clientHeight/2,s*1.2)};
document.getElementById('zo').onclick=function(){zoomAt(vp.clientWidth/2,vp.clientHeight/2,s/1.2)};
document.getElementById('fit').onclick=fit;
document.getElementById('all').onclick=function(){nodes.forEach(function(b){if(b._k)setC(b,false)});requestAnimationFrame(relayout)};
document.getElementById('less').onclick=function(){nodes.forEach(function(b){if(b._k)setC(b,b._d>=1)});requestAnimationFrame(relayout)};
var coarse=window.matchMedia&&matchMedia('(pointer:coarse)').matches,pan=document.getElementById('pan');
function tmode(){vp.style.touchAction=(!coarse||pan.getAttribute('aria-pressed')==='true')?'none':'pan-y'}
if(coarse)pan.style.display='';
pan.onclick=function(){var on=pan.getAttribute('aria-pressed')!=='true';pan.setAttribute('aria-pressed',String(on));pan.style.background=on?'#0F5C5A':'';pan.style.color=on?'#fff':'';tmode()};
tmode();
requestAnimationFrame(relayout);
if(document.fonts&&document.fonts.ready)document.fonts.ready.then(relayout);
if(window.ResizeObserver)new ResizeObserver(relayout).observe(vp);
})();
</script></body></html>"""

MATH_HTML = r"""<!doctype html><html><head><meta charset="utf-8"></head><body>
<script>
(function(){
var P,D;
try{P=window.parent;D=P.document;void D.body}catch(e){return}
var SETS=[
 ["Kí hiệu",["±","×","÷","√","²","³","ⁿ","₁","₂","ₙ","≤","≥","≠","≈","∞","∑","∏","∫","∂","∈","∉","⊂","∪","∩","∅","→","⇒","⇔","∴","∀","∃","°","‰","|","Ā","x̄","p̂"]],
 ["Hy Lạp",["α","β","γ","δ","ε","θ","λ","μ","ν","π","ρ","σ","τ","φ","χ","ψ","ω","Δ","Σ","Ω","Φ","Π","Θ","Λ","χ²"]],
 ["Công thức",[
  ["phân số","$\\frac{a}{b}$"],["căn","$\\sqrt{x}$"],["mũ","$x^{2}$"],["chỉ số","$x_{i}$"],["tổ hợp","$C_n^k$"],["chỉnh hợp","$A_n^k$"],
  ["n!","n!"],["P(A|B)","P(A|B)"],["P(A∩B)","P(A∩B)"],["P(A∪B)","P(A∪B)"],["E(X)","E(X)"],["Var(X)","Var(X)"],
  ["X~B(n;p)","X ~ B(n; p)"],["X~N(μ;σ²)","X ~ N(μ; σ²)"],["tổng","$\\sum_{i=1}^{n} x_i$"],["tích phân","$\\int_{a}^{b} f(x)\\,dx$"],
  ["xác suất toàn phần","$P(B)=\\sum_{i} P(A_i)P(B|A_i)$"],["Bayes","$P(A|B)=\\frac{P(B|A)P(A)}{P(B)}$"],
  ["nhị thức","$P(X=k)=C_n^k p^k (1-p)^{n-k}$"]
 ]]
];
["xstk-math","xstk-math-css"].forEach(function(id){var o=D.getElementById(id);if(o)o.remove()});
if(P.__xstkFocus)D.removeEventListener('focusin',P.__xstkFocus,true);
P.__xstkField=null;
P.__xstkFocus=function(ev){var t=ev.target;if(t&&(t.tagName==='TEXTAREA'||(t.tagName==='INPUT'&&(!t.type||t.type==='text'||t.type==='search'))))P.__xstkField=t};
D.addEventListener('focusin',P.__xstkFocus,true);
var css=D.createElement('style');css.id='xstk-math-css';
css.textContent="#xstk-math{position:fixed;left:14px;bottom:90px;z-index:999990;font-family:'Be Vietnam Pro',-apple-system,'Segoe UI',Roboto,Arial,sans-serif}"
+"#xstk-math .fab{width:50px;height:50px;border-radius:50%;border:0;background:#0F5C5A;color:#fff;font-size:24px;font-family:Georgia,'Times New Roman',serif;cursor:pointer;box-shadow:0 4px 14px rgba(15,92,90,.38)}"
+"#xstk-math .fab:focus-visible,#xstk-math button:focus-visible{outline:2px solid #B4531F;outline-offset:2px}"
+"#xstk-math .panel{display:none;position:absolute;left:0;bottom:60px;width:min(92vw,400px);max-height:min(58vh,430px);overflow:auto;background:#FBF9F4;color:#1F2A27;border:1px solid #D9D2C3;border-radius:16px;padding:12px;box-shadow:0 12px 34px rgba(31,42,39,.22)}"
+"#xstk-math.open .panel{display:block}"
+"#xstk-math .hd{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:4px;font-weight:600;font-size:15px}"
+"#xstk-math .hd button{border:0;background:transparent;font-size:20px;line-height:1;cursor:pointer;color:#5E6B66;padding:4px 8px}"
+"#xstk-math .sub{font-size:12px;color:#5E6B66;margin-bottom:8px;line-height:1.4}"
+"#xstk-math .tabs{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}"
+"#xstk-math .tab{border:1px solid #D9D2C3;background:#fff;border-radius:99px;padding:6px 12px;font-size:13px;cursor:pointer;font-family:inherit;color:#1F2A27}"
+"#xstk-math .tab.on{background:#0F5C5A;border-color:#0F5C5A;color:#fff}"
+"#xstk-math .grid{display:flex;flex-wrap:wrap;gap:6px}"
+"#xstk-math .k{min-width:42px;height:42px;padding:0 10px;border:1px solid #D9D2C3;border-radius:10px;background:#fff;color:#1F2A27;font-size:17px;cursor:pointer;font-family:inherit}"
+"#xstk-math .k.f{font-size:13px;height:auto;min-height:42px;padding:6px 12px}"
+"#xstk-math .k:hover{border-color:#0F5C5A;color:#0F5C5A}"
+"#xstk-math .msg{min-height:18px;margin-top:8px;font-size:12px;color:#0F5C5A}";
D.head.appendChild(css);
function el(tag,cls,txt){var e=D.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e}
var root=el('div');root.id='xstk-math';
var fab=el('button','fab','Σ');fab.type='button';fab.setAttribute('aria-label','Mở bàn phím kí hiệu toán học');fab.setAttribute('aria-expanded','false');
var panel=el('div','panel');panel.setAttribute('role','dialog');panel.setAttribute('aria-label','Bàn phím kí hiệu toán học');
var hd=el('div','hd');hd.appendChild(el('span',null,'Kí hiệu toán học'));
var cl=el('button',null,'×');cl.type='button';cl.setAttribute('aria-label','Đóng');hd.appendChild(cl);
var sub=el('div','sub','Nhấp vào ô nhập trước, rồi chọn kí hiệu để chèn tại vị trí con trỏ.');
var tabs=el('div','tabs'),grid=el('div','grid'),msg=el('div','msg');msg.setAttribute('role','status');
panel.appendChild(hd);panel.appendChild(sub);panel.appendChild(tabs);panel.appendChild(grid);panel.appendChild(msg);
root.appendChild(panel);root.appendChild(fab);D.body.appendChild(root);
var timer=0;
function say(t){msg.textContent=t;clearTimeout(timer);timer=setTimeout(function(){msg.textContent=''},1800)}
function insert(txt){
  var f=P.__xstkField;
  if(!f||!D.contains(f)){say('Hãy nhấp vào một ô nhập trước.');return}
  f.focus();
  var a=f.selectionStart==null?f.value.length:f.selectionStart,b=f.selectionEnd==null?a:f.selectionEnd;
  var setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(f),'value').set;
  setter.call(f,f.value.slice(0,a)+txt+f.value.slice(b));
  var pos=a+txt.length;try{f.setSelectionRange(pos,pos)}catch(e){}
  f.dispatchEvent(new P.Event('input',{bubbles:true}));
  say('Đã chèn '+txt.slice(0,20));
}
function show(i){
  Array.prototype.forEach.call(tabs.children,function(t,j){t.classList.toggle('on',i===j)});
  grid.textContent='';
  SETS[i][1].forEach(function(it){
    var lab=Array.isArray(it)?it[0]:it,txt=Array.isArray(it)?it[1]:it;
    var k=el('button','k'+(Array.isArray(it)?' f':''),lab);k.type='button';
    k.onclick=function(){insert(txt)};grid.appendChild(k);
  });
}
SETS.forEach(function(s,i){var t=el('button','tab',s[0]);t.type='button';t.onclick=function(){show(i)};tabs.appendChild(t)});
show(0);
function toggle(o){root.classList.toggle('open',o);fab.setAttribute('aria-expanded',String(o))}
fab.onclick=function(){toggle(!root.classList.contains('open'))};
cl.onclick=function(){toggle(false)};
root.addEventListener('mousedown',function(e){e.preventDefault()});
function place(){
  var v=P.visualViewport;if(!v)return;
  var off=Math.max(0,P.innerHeight-v.height-v.offsetTop);
  root.style.bottom=(off>0?off+12:90)+'px';
}
if(P.visualViewport){P.visualViewport.addEventListener('resize',place);P.visualViewport.addEventListener('scroll',place)}
})();
</script></body></html>"""


def clean_tree(n, depth=0, cnt=None):
    """Làm sạch cây do AI trả về trước khi nhúng vào HTML: chỉ giữ chữ, tối đa 3 cấp và 80 nút."""
    cnt = cnt if cnt is not None else [0]
    if not isinstance(n, dict) or cnt[0] >= 80:
        return None
    cnt[0] += 1
    ten = re.sub(r"[\x00-\x1f$\\]", "", str(n.get("ten", "")))[:80].strip() or "?"
    kids = []
    if depth < 3 and isinstance(n.get("con"), list):
        kids = [c for c in (clean_tree(k, depth + 1, cnt) for k in n["con"]) if c]
    return {"ten": ten, "con": kids}


def mindmap_html(tree):
    data = json.dumps(clean_tree(tree), ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return MINDMAP_HTML.replace("__DATA__", data)


def outline(n, d=0):
    return "  " * d + f"- {n['ten']}\n" + "".join(outline(c, d + 1) for c in n["con"])


def map_panel(proc, topic):
    mk = f"r_map_{proc}"
    if st.button("Tạo bản đồ tư duy", icon=":material/account_tree:", key=f"b_map_{proc}"):
        with st.spinner("Đang vẽ bản đồ..."):
            t = ask_json(SYS_MAP, f"Chủ đề: {topic}", max_tokens=2500)
        tree = clean_tree(t)
        if tree and tree["con"]:
            S[mk] = {"topic": topic, "tree": tree}
        elif t is not None:
            st.error("Không tạo được bản đồ. Hãy thử lại.")
    d = S.get(mk)
    if d:
        st.markdown(f"#### Bản đồ tư duy · {d['topic']}")
        embed(mindmap_html(d["tree"]), 560)
        with st.expander("Dạng danh sách (dễ đọc trên điện thoại)"):
            st.markdown(outline(d["tree"]))


# ───────────────────────── QUY TRÌNH HỌC KHÁI NIỆM / ĐỊNH LÍ ─────────────────────────
def topic_box(proc):
    opts = CONCEPTS if proc == "concept" else THEOREMS
    what = "khái niệm" if proc == "concept" else "định lí, công thức"
    cur = S.get(f"sel_{proc}", opts[0])
    with st.container(border=True):
        c1, c2 = st.columns([3, 2], gap="large")
        sel = c1.selectbox(f"Chọn {what} cần học", opts, index=opts.index(cur) if cur in opts else 0, key=f"w_sel_{proc}")
        own = c2.text_input("Hoặc nhập chủ đề riêng", value=S.get(f"own_{proc}", ""), key=f"w_own_{proc}",
                            placeholder="Ví dụ: Phân phối chuẩn tắc", max_chars=80)
    S[f"sel_{proc}"], S[f"own_{proc}"] = sel, own
    return own.strip() or sel


def learn_1(proc, topic):
    is_c = proc == "concept"
    step_head(1, "Trải nghiệm",
              "Tiếp cận dấu hiệu bản chất của khái niệm qua tình huống cụ thể." if is_c
              else "Tiếp cận giả thiết và kết luận, khơi gợi nhu cầu tự kiểm chứng.")
    sk = f"r_sit_{proc}"
    if st.button("Tạo tình huống trải nghiệm", icon=":material/science:", type="primary", key=f"b_sit_{proc}"):
        with st.spinner("Đang soạn tình huống..."):
            d = ask_json(sys_situation(proc), f"Chủ đề: {topic}", max_tokens=1500, think="medium")
        if isinstance(d, dict) and d.get("tinh_huong"):
            S[sk] = {"topic": topic, **d}
            S.pop(f"r_obs_{proc}", None)
            mark(proc, 0)
    sit = S.get(sk)
    if not sit:
        return
    with st.container(border=True):
        st.markdown(f"#### Tình huống · {sit['topic']}")
        st.markdown(str(sit["tinh_huong"]))
        bullets("Câu hỏi dẫn dắt", [str(x) for x in sit.get("cau_hoi") or []])
        if sit.get("goi_y"):
            st.caption(f"Gợi ý quan sát: {sit['goi_y']}")
        obs = keep_text("Nhận xét" + (" của bạn" if is_c else " và dự đoán của bạn"), f"obs_{proc}", height=120, max_chars=2000)
        if st.button("AI nhận xét", icon=":material/rate_review:", key=f"b_obs_{proc}"):
            if not obs.strip():
                st.warning("Hãy ghi nhận xét của bạn trước.")
            else:
                with st.spinner("Đang nhận xét..."):
                    S[f"r_obs_{proc}"] = ask_cached(
                        SYS_OBS, f"Chủ đề: {sit['topic']}\nTình huống: {sit['tinh_huong']}\nCâu hỏi: {sit.get('cau_hoi')}\n"
                                 f"Nhận xét của học sinh: {obs}", max_tokens=700)
        if S.get(f"r_obs_{proc}"):
            st.info(S[f"r_obs_{proc}"])


def learn_2(proc, topic):
    is_c = proc == "concept"
    step_head(2, "Hình thành định nghĩa" if is_c else "Hình thành định lí",
              "Nhận biết dấu hiệu bản chất, lĩnh hội thuật ngữ và kí hiệu, phát biểu và ghi nhớ định nghĩa." if is_c
              else "Phân biệt giả thiết và kết luận, nhận biết luận cứ và cách chứng minh, hiểu cấu trúc logic.")
    fk = f"r_frame_{proc}"
    if st.button("Hiện khung định nghĩa" if is_c else "Hiện khung định lí", icon=":material/menu_book:",
                 type="primary", key=f"b_frame_{proc}"):
        with st.spinner("Đang lập khung..."):
            d = ask_json(sys_frame(proc), f"Chủ đề: {topic}", max_tokens=3000, think="medium")
        if isinstance(d, dict):
            S[fk] = {"topic": topic, **d}
            mark(proc, 1)
    f = S.get(fk)
    if f:
        with st.container(border=True):
            st.markdown(f"#### {'Định nghĩa' if is_c else 'Định lí'} · {f['topic']}")
            if is_c:
                st.markdown(f"**Định nghĩa.** {f.get('dinh_nghia', '')}")
                bullets("Dấu hiệu bản chất", f.get("dau_hieu"))
                bullets("Thuật ngữ và kí hiệu", [f"{t.get('ky_hieu', '')}: {t.get('y_nghia', '')}" if isinstance(t, dict) else str(t)
                                                 for t in f.get("thuat_ngu") or []])
                bullets("Ví dụ", f.get("vi_du"))
                bullets("Phản ví dụ", f.get("phan_vi_du"))
                if f.get("luu_y"):
                    st.info(f"Lưu ý: {f['luu_y']}")
            else:
                st.markdown(f"**Phát biểu.** {f.get('phat_bieu', '')}")
                bullets("Giả thiết", f.get("gia_thiet"))
                bullets("Kết luận", f.get("ket_luan"))
                if f.get("dieu_kien"):
                    st.markdown(f"**Điều kiện áp dụng.** {f['dieu_kien']}")
                bullets("Luận cứ", f.get("luan_cu"))
                if f.get("y_tuong"):
                    st.markdown(f"**Ý tưởng chứng minh.** {f['y_tuong']}")
                if f.get("chung_minh"):
                    with st.expander("Chứng minh"):
                        st.markdown(str(f["chung_minh"]))
                if f.get("cau_truc"):
                    st.info(f"Cấu trúc logic: {f['cau_truc']}")
    st.markdown("**Hệ thống hoá**")
    map_panel(proc, topic)
    if S.get(f"r_map_{proc}"):
        mark(proc, 1)


def learn_3(proc, topic):
    is_c = proc == "concept"
    step_head(3, "Củng cố",
              "Nhận diện khái niệm trong trường hợp đơn giản; thể hiện khái niệm trong các ngữ cảnh và mối liên hệ logic." if is_c
              else "Thực hành vận dụng định lí trong những trường hợp đơn giản có tính chất đặc trưng.")
    t1, t2 = st.tabs(["Luyện tập", "Vấn đáp"])
    with t1:
        practice_panel(proc, "cung_co", topic, 2)
    with t2:
        oral_panel(proc, topic, 2)


def learn_4(proc, topic):
    is_c = proc == "concept"
    step_head(4, "Vận dụng",
              "Vận dụng khái niệm trong các tình huống gián tiếp, phức hợp và giải quyết vấn đề thực tiễn." if is_c
              else "Vận dụng định lí giải quyết vấn đề toán học hoặc vấn đề gắn với thực tiễn.")
    practice_panel(proc, "van_dung", topic, 3)


# ───────────────────────── KHUNG CHÍNH ─────────────────────────
STEPS = {"solve": (solve_1, solve_2, solve_3, solve_4),
         "concept": (learn_1, learn_2, learn_3, learn_4),
         "theorem": (learn_1, learn_2, learn_3, learn_4)}


def set_proc(k):
    S.proc = k


def proc_nav():
    cur = S.get("proc", "solve")
    for col, (k, (title, icon, desc, _)) in zip(st.columns(3, gap="medium"), PROCS.items()):
        with col:
            st.button(title, icon=icon, key=f"b_proc_{k}", type="primary" if k == cur else "secondary",
                      on_click=set_proc, args=(k,))
            st.markdown(f'<div class="procdesc">{desc}</div>', unsafe_allow_html=True)


def render_proc(proc):
    done = S.setdefault("done", set())
    labels = PROCS[proc][3]
    topic = topic_box(proc) if proc != "solve" else None
    n = sum((proc, i) in done for i in range(4))
    st.progress(n / 4, text=f"Tiến độ quy trình: đã hoàn thành {n}/4 bước")
    wk = f"w_step_{proc}"
    if wk not in S:
        S[wk] = S.get(f"cur_{proc}", 0)
    sel = st.segmented_control("Bước", [0, 1, 2, 3], key=wk, label_visibility="collapsed",
                               format_func=lambda i: ("✓ " if (proc, i) in done else "") + labels[i])
    if sel is not None:
        S[f"cur_{proc}"] = sel
    fn = STEPS[proc][S.get(f"cur_{proc}", 0)]
    fn() if proc == "solve" else fn(proc, topic)


def topbar():
    a, b, c = st.columns([5, 1.8, 1.6], vertical_alignment="center")
    a.markdown('<div class="brand"><span class="brand-mark" aria-hidden="true">Σ</span>'
               '<span class="eyebrow">Xác suất thống kê · Trợ lí học tập</span></div>', unsafe_allow_html=True)
    if b.button("Hồ sơ học tập", icon=":material/person:", type="primary", key="b_prof"):
        profile_dialog()
    with c.popover(S.user, icon=":material/account_circle:"):
        st.download_button("Tải hồ sơ (.json)", json.dumps(S.profile, ensure_ascii=False),
                           file_name="ho_so_xstk.json", mime="application/json")
        if st.button("Đổi người dùng", key="b_out"):
            for k in list(S.keys()):
                del S[k]
            st.rerun()


def login():
    l, r = st.columns([1.2, 1], gap="large", vertical_alignment="center")
    with l:
        st.markdown('<div class="brand"><span class="brand-mark" aria-hidden="true">Σ</span>'
                    '<span class="eyebrow">Xác suất thống kê · Trợ lí học tập</span></div>', unsafe_allow_html=True)
        hero()
    with r, st.container(border=True):
        st.markdown("#### Bắt đầu học")
        with st.form("login"):
            name = st.text_input("Tên của bạn", max_chars=40)
            go = st.form_submit_button("Vào học", type="primary")
        if go:
            if name.strip():
                S.user, S.profile = name.strip(), load_profile(name.strip())  # shortcut: chỉ dùng tên, chưa có mật khẩu
                st.rerun()
            st.warning("Hãy nhập tên của bạn.")
        with st.expander("Khôi phục hồ sơ đã tải về"):
            up = st.file_uploader("Tệp .json", type=["json"], key="restore", label_visibility="collapsed")
            if up:
                try:
                    p = json.load(up)
                    assert isinstance(p.get("events"), list) and str(p.get("name", "")).strip()
                except (ValueError, AssertionError, AttributeError):
                    st.error("Tệp hồ sơ không hợp lệ.")
                else:
                    p["name"] = str(p["name"]).strip()[:40]
                    save_profile(p)
                    S.user, S.profile = p["name"], p
                    st.rerun()
    footer()


def footer():
    st.markdown('<div class="foot"><b>Σ μ σ</b><br>Đây là nội dung do AI tạo ra, xin hãy đối chiếu với đề cương.</div>',
                unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="XSTK Tutor", page_icon=":material/functions:", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    if not S.get("user"):
        login()
        return
    topbar()
    hero()
    embed(MATH_HTML, 1)  # bàn phím kí hiệu toán học nổi, dùng cho mọi ô nhập
    if not _has_key():
        st.warning("Chưa cấu hình API key: các tính năng AI tạm thời chưa hoạt động.")
    proc_nav()
    st.divider()
    render_proc(S.get("proc", "solve"))
    footer()


main()
