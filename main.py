import json
import os
import threading
import time
import websocket
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép mọi nơi gọi API

# --- LƯU TRỮ DỮ LIỆU ---
md5_storage = {"htr": []}
MAX_HISTORY = 200
data_lock = threading.Lock()

MD5_URL = "THAY_LINK_WEBSOCKET_O_DAY"


def extract_entry(payload):
    """Hàm phụ: Rút gọn việc lấy thông tin xí ngầu và mã phiên từ payload."""
    res = payload.get("Result") or payload.get("result") or payload

    # Kiểm tra xem res có phải là dict không trước khi lấy value
    if not isinstance(res, dict):
        return None

    # Lấy giá trị xí ngầu (d1, d2, d3) và session id (sid)
    sid = payload.get("SessionID") or payload.get("sid") or payload.get("SessionId")
    d1 = res.get("Dice1") or res.get("d1")
    d2 = res.get("Dice2") or res.get("d2")
    d3 = res.get("Dice3") or res.get("d3")

    # Kiểm tra hợp lệ: sid phải tồn tại và xí ngầu phải từ 1 đến 6
    dices = [d1, d2, d3]
    if sid and all(isinstance(v, int) and 1 <= v <= 6 for v in dices):
        return {"sid": sid, "d1": d1, "d2": d2, "d3": d3}

    return None


def parse_md5_message(raw_message):
    """Xử lý tin nhắn thô từ WebSocket trả về."""
    try:
        # Xóa ký tự phân đoạn đặc biệt (\x1e) của SignalR
        clean_msg = raw_message.replace("\x1e", "").strip()
        if not clean_msg:
            return

        data = json.loads(clean_msg)

        # Lặp qua các tin nhắn SignalR (thường nằm trong mảng 'M')
        for msg in data.get("M", []):
            if "A" in msg and msg["A"]:
                payload = msg["A"][0]
                entry = extract_entry(payload)

                if entry:
                    with data_lock:
                        # Tránh lưu trùng phiên (SessionID)
                        if not any(
                            item["sid"] == entry["sid"]
                            for item in md5_storage["htr"]
                        ):
                            md5_storage["htr"].insert(0, entry)

                            # Xóa bớt nếu vượt quá giới hạn bộ nhớ
                            if len(md5_storage["htr"]) > MAX_HISTORY:
                                md5_storage["htr"].pop()

                            print(f"✅ Đã thêm phiên: {entry['sid']}")

    except Exception as e:
        print(f"❌ Lỗi xử lý tin nhắn: {e}")


def start_md5_ws():
    """Khởi tạo và duy trì kết nối WebSocket."""

    def on_message(ws, message):
        parse_md5_message(message)

    def on_error(ws, error):
        print(f"⚠️ WebSocket Lỗi: {error}")

    def on_close(ws, *args):
        print("🔄 WebSocket bị ngắt. Đang kết nối lại sau 5s...")
        time.sleep(5)
        threading.Thread(target=start_md5_ws, daemon=True).start()

    ws = websocket.WebSocketApp(
        MD5_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()


# --- API FLASK ---
@app.route("/api/tx")
def get_md5_data():
    with data_lock:
        return jsonify(
            {
                "status": "success",
                "total": len(md5_storage["htr"]),
                "htr": md5_storage["htr"],
            }
        )


if __name__ == "__main__":
    # Chạy WebSocket ở luồng ngầm (Background Thread)
    threading.Thread(target=start_md5_ws, daemon=True).start()

    # Chạy Web App Flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
