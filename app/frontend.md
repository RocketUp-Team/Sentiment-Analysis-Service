# Frontend API Specifications

Tài liệu này cung cấp chi tiết về **Contract API** mà Frontend Chatbot đang gọi xuống Server. Các kỹ sư Cloud / Backend đọc kỹ tài liệu này để implement các Endpoint tương ứng trong `main.py` của Backend.

---

## 🏗️ 1. Cấu hình Cơ bản (Base Settings)
- **Base URL:** `http://localhost:8000`
- **CORS:** Backend cần cấu hình CORS cho phép `localhost:4200` gọi tới.

---

## ⚡ 2. API Endpoints Frontend đang sử dụng

Dưới đây là 3 API mà Frontend gọi xuống Backend. Bất cứ khi nào có lỗi HTTP (400, 404, 500) hoặc Backend chưa support, Frontend đều tự động *Fallback về giả lập Mock UI (fake result)* để không sập app. 

### A. Phân tích Text Trực tiếp (Có sẵn hiện tại)
Frontend truyền một đoạn text bất kỳ xuống nhờ Backend Inference Model nhận diện:
- **Endpoint:** `POST /predict`
- **Content-Type:** `application/json`
- **Request Body Payload:**
  ```json
  {
    "text": "Đoạn văn bản người dùng gõ từ bàn phím hoặc hệ thống đã phân tích từ giọng nói Web API...",
    "lang": "vi" // hoặc "en"
  }
  ```
- **Response Format Kì vọng (HTTP 200)** (Mọi API dưới đây đều dùng format này):
  ```json
  {
    "text": "Nội dung ban đầu (tuỳ chọn)",
    "sentiment": "positive", // hoặc "negative" hoặc "neutral"
    "confidence": 0.95,
    "latency_ms": 150
  }
  ```

---

### B. Upload Tài liệu Document (Tính năng mới)
Frontend hỗ trợ người dùng upload file đính kèm (`.pdf` hoặc `.docx`). Backend team cần nhận file này, dùng `PyPDF2` / `python-docx` để đọc raw text từ file rồi chạy Inference y như gọi `/predict`. 

- **Endpoint:** `POST /upload/document`
- **Content-Type:** `multipart/form-data`
- **Request Payload:**
  Frontend gửi `FormData` chứa:
  - `file`: (Binary File, loại MIME `.pdf` hoặc `.docx`)
  - `lang`: `"vi"` (String)
- **Response Format Kì vọng:** Trống y hệt `/predict` (trả về Sentiment của nội dung bóc cục từ PDF).

---

### C. Ghi âm Audio File (Tính năng mới - Lưu ý)
Hiện tại tính năng Micro của Frontend đang sử dụng luồng: **Web Speech API trực tiếp dịch lời nói ra Chữ (Real-time Transcription)**. Nếu Frontend lấy được chữ bằng Web Speech API nó sẽ bắn thẳng qua Endpoint **[A] `/predict`**. 
Chỉ trong trường hợp Frontend *không đọc được chữ* nhưng lại có file âm thanh ghi từ `MediaRecorder` thì sẽ chạy Endpoint sau. 

*(Tuy hiện tại Frontend đã tích hợp logic Web Speech Auto-Transcribe thay vì gửi file, Backend Team VẪN nên build Endpoint Audio phục vụ cho Mobile API hoặc API chéo).*
- **Endpoint:** `POST /upload/audio`
- **Content-Type:** `multipart/form-data`
- **Request Payload:**
  - `file`: (Audio Blob, loại `audio/webm` chuẩn từ Browser). Cần dùng `pydub` decode.
  - `lang`: `"vi"`
- **Response Kì vọng:** Y hệt `/predict`.
