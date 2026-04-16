# Frontend Chatbot - Sentiment Analysis Pro AI 🚀

Dự án này là giao diện tương tác Frontend cho dịch vụ **Sentiment Analysis** 

## 🌟 Chức năng nổi bật
- **Giao diện Premium SaaS:** Giao diện tối giản hiện đại với chế độ Light Mode / Dark Mode siêu mượt.
- **Glassmorphism Design:** Tích hợp hiệu ứng mờ kính và hiệu ứng gradient animation.
- **Tương tác linh hoạt:** 
  - Chat text truyền thống.
  - Tích hợp File Attachment (PDF/DOCX).
  - Tích hợp Web Speech API (nhận diện giọng nói) ngay trên trình duyệt (Kiểu giao diện Hold-to-Talk như Telegram/Zalo).
- **Graceful Degradation:** Tự động sử dụng Mock API giả lập nếu Backend Server (trên `localhost:8000`) bị mất kết nối hoặc chưa hoàn thành tính năng.

## 🛠️ Stack & Môi trường
- **Framework:** Angular 19+ (Standalone Components).
- **Styling:** CSS + Tailwind CSS utilities (tự động biên dịch classes).
- **Icon:** `lucide-angular` (đã tối ưu hóa Config Providers tại Root).

## 🚀 Hướng dẫn Khởi chạy (Local Development)

### Lưu ý quan trọng
Để chạy được Frontend độc lập, bạn cần cài đặt **Node.js (>= 18.x)**.

### Các bước cài đặt & Triển khai (CI/CD)
   ```bash
   # Di chuyển vào thư mục frontend
   cd app/sentiment-analysis-chatbot
   
   # Cài đặt toàn bộ packets phụ thuộc
   npm install
   
   # 1. Chạy ở môi trường Local (Development)
   # Dùng để code và xem thay đổi ngay lập tức (Live Reload)
   npm run start
   
   # 2. Build dự án để triển khai (Production / CI-CD)
   # Lệnh này sẽ biên dịch toàn bộ source code thành các file tĩnh (HTML/JS/CSS) 
   # nằm trong thư mục: app/sentiment-analysis-chatbot/dist/
   # Team Cloud CHỈ CẦN lấy nội dung trong thư mục 'dist' này để chạy trên Web Server (Nginx, S3, vv.)
   npm run build
   ```

Giao diện sẽ lập tức hiển thị sau vài giây compile, bạn có thể truy cập qua: `http://localhost:4200`.

## 🤝 Ghi chú dành cho Backend (Cloud Team)
Frontend hoàn toàn phụ thuộc vào danh sách API được cung cấp tại Backend. Vui lòng tham khảo tệp `frontend.md` nằm cùng thư mục này để đọc các API Contracts mà Frontend đang gọi (bao gồm URL, Parameters và Format Payload) để phát triển Backend và Integration cho chính xác.
