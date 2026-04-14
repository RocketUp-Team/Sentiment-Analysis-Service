# **KẾ HOẠCH TRIỂN KHAI DỰ ÁN (TASK BREAKDOWN)**

**Dự án:** Sentiment Analysis Service (Topic 8\)

**Mục tiêu:** Hoàn thành End-to-End ML System đáp ứng 100% Rubric môn DDM501.

## **MỤC TIÊU CHIẾN LƯỢC**

1. **MVP First:** Ưu tiên xây dựng một luồng chạy được xuyên suốt từ UI \-\> API \-\> Model \-\> UI càng sớm càng tốt. KHÔNG sa đà vào train model điểm cao ngay từ đầu.  
2. **Mocking:** Quân và Long sẽ dùng dữ liệu giả (Mock data) dựa trên API Contract để code ngay lập tức, không chờ Trung train xong model.  
3. **Pre-trained Models:** Trung nên dùng model có sẵn (VD: xlm-roberta-base-sentiment hoặc philschmid/distilbert-base-multilingual-cased-sentiment-2) để làm Baseline, sau đó mới fine-tune (PEFT/LoRA) trên tập data dự án.

## **GIAI ĐOẠN 1: KICK-OFF & CHỐT API CONTRACT**

*Tất cả 3 thành viên họp để chốt các vấn đề sau:*

1. **Chốt Architecture & Tech Stack:**  
   * Model: HuggingFace Transformers (PyTorch).  
   * Backend: FastAPI (có sẵn Swagger UI cho API Docs).  
   * Frontend: Streamlit (code bằng Python, cực nhanh) hoặc React/HTML đơn giản.  
   * MLOps/DevOps: MLflow (Tracking), Docker Compose, Prometheus \+ Grafana.  
2. **Chốt API Contract (Cực kỳ quan trọng):**  
   * Định nghĩa chính xác Request/Response cho 2 Endpoints chính: Real-time và Batch.  
   * *Ví dụ Request Real-time:* {"text": "Món ăn quá tệ, phục vụ chậm\!", "lang": "vi"}  
   * *Ví dụ Response:* {"sentiment": "negative", "confidence": 0.95, "aspects": \[{"aspect": "food", "sentiment": "negative"}, {"aspect": "service", "sentiment": "negative"}\], "sarcasm\_flag": false, "latency\_ms": 45}

## **GIAI ĐOẠN 2: PHÂN CÔNG NHIỆM VỤ CHI TIẾT**

### **1\. Trung (AI Core & Modeling) \- *Trọng số Rubric: 25% (ML Pipeline & Responsible AI)***

**Nhiệm vụ trọng tâm:** Xử lý dữ liệu, huấn luyện mô hình, xuất model weights và viết hàm xử lý core AI.

* **Dữ liệu & Baseline Model:**  
  * Tải subset nhỏ của Amazon/Yelp/Twitter (không dùng full data để tiết kiệm thời gian train ban đầu).  
  * Viết script Preprocessing (làm sạch emoji, text, tokenizer).  
  * Khởi tạo Baseline model bằng pre-trained model. Gói model thành 1 class Python (ModelInference) có hàm predict\_single(text) và predict\_batch(list\_texts).  
* **Fine-tuning & Tracking:**  
  * Tích hợp **MLflow** để track metrics (F1-score, Accuracy) và params.  
  * Áp dụng PEFT/LoRA để fine-tune nhẹ trên tập data dự án nhằm xử lý Sarcasm hoặc đa ngôn ngữ.  
  * Viết script xử lý Batch Job (đọc từ file CSV/JSON, chạy qua model, xuất ra file kết quả).  
* **Responsible AI & Tối ưu hóa:**  
  * Viết code chạy **SHAP hoặc LIME** để giải thích mô hình (Explainability \- Yêu cầu bắt buộc của Rubric). Xuất plot ảnh đưa cho Long.  
  * (Optional) Export model sang ONNX hoặc dùng Quantization INT8 để tối ưu Inference Latency.

### **2\. Quân (Backend, DevOps & MLOps) \- *Trọng số Rubric: 40% (Deployment, Monitoring, CI/CD)***

**Nhiệm vụ trọng tâm:** Xây dựng cầu nối, triển khai hệ thống và giám sát.

* **API & Containerization:**  
  * Khởi tạo FastAPI project. Viết các routers /predict và /batch\_predict trả về Mock Data (theo API Contract) để Long làm UI.  
  * Viết Dockerfile tối ưu cho FastAPI và docker-compose.yml gồm các service: fastapi\_app, prometheus, grafana, mlflow.  
* **Integration & Monitoring:**  
  * Tích hợp class ModelInference của Trung vào FastAPI. Chạy thử luồng thật nghiệm thu Latency.  
  * Gắn middleware vào FastAPI để export metrics (Request count, Error rate, Model Inference Latency) ra định dạng cho Prometheus.  
  * Cấu hình Dashboard trên **Grafana** (Vẽ biểu đồ request/giây, độ trễ, phân phối cảm xúc). Thiết lập Alert rule cơ bản.  
* **Testing & CI/CD Pipeline:**  
  * Viết Unit Tests (pytest) cho API endpoints và logic xử lý dữ liệu.  
  * Viết GitHub Actions file (.github/workflows/main.yml) để chạy test và linting (flake8/black) mỗi khi có push code.

### **3\. Long (Frontend, Documentation & Report) \- *Trọng số Rubric: 35% (Problem Def, Design, UI, Docs)***

**Nhiệm vụ trọng tâm:** Giao diện người dùng, Tài liệu hóa hệ thống và Slide thuyết trình.

* **Problem Definition & Architecture Design:**  
  * Viết phần Problem Statement, Requirements, Success Metrics, Data Flow diagram (dùng Draw.io/Lucidchart).  
  * Khảo sát các vấn đề Đạo đức AI (Fairness, Data Privacy) để viết phần Responsible AI.  
* **Frontend UI (Streamlit/React):**  
  * Build UI kết nối API Backend của Quân.  
  * Thiết kế 2 luồng tính năng chính:  
    1. *Real-time:* Ô text nhập liệu \-\> Nút submit \-\> Hiển thị kết quả (Bảng cảm xúc, biểu đồ SHAP giải thích từ Trung).  
    2. *Batch Job:* Nút upload file CSV \-\> Nút process \-\> Nút download kết quả \+ Hiển thị biểu đồ thống kê tổng quan (Pie chart).  
* **Documentation & Report (LaTeX):**  
  * Tạo cấu trúc repository chuẩn (có .gitignore, requirements.txt).  
  * Viết README.md cực kỳ chi tiết (Hướng dẫn setup từng bước, run docker-compose).  
  * Tổng hợp tất cả nội dung vào template LaTeX thành Report hoàn chỉnh. Đảm bảo format theo chuẩn học thuật/ngành.

## **GIAI ĐOẠN 3: TÍCH HỢP, KIỂM THỬ & NGHIỆM THU**

* **End-to-End Testing & Bug Fixing:**  
  * Ba người cùng test toàn bộ luồng: Upload CSV \-\> FE gọi BE \-\> BE gọi Model \-\> Trả kết quả FE.  
  * Quân check lại Grafana xem metrics có nhảy không khi test.  
  * Trung đẩy model weights nhẹ nhất lên repo (hoặc load qua HuggingFace) để Docker build không bị lỗi quá tải.  
* **Slide & Presentation Rehearsal:**  
  * Long hoàn thiện Slide (Canva/PPT).  
  * Chuẩn bị Kịch bản Live Demo (Tránh rủi ro lúc demo thật).  
  * Rehearsal Q\&A: Chia nhau trả lời câu hỏi dựa trên Rubric (Trung trả lời về Model/Sarcasm, Quân trả lời về Latency/Docker/CI/CD, Long trả lời về UI/Ethics/Business flow).

## **CHECKLIST KIỂM TRA CHÉO THEO RUBRIC DDM501**

* \[ \] **Problem & Req (10%):** Có mục tiêu kinh doanh, metrics rõ ràng chưa? (Long)  
* \[ \] **Architecture (15%):** Có sơ đồ luồng dữ liệu, giải thích trade-off tech stack chưa? (Long \+ Quân)  
* \[ \] **Implementation \- ML (15%):** Có MLflow tracking params/metrics không? (Trung)  
* \[ \] **Implementation \- Deploy (15%):** Có docker-compose chạy 1 lệnh là lên toàn bộ không? Swagger API docs có không? (Quân)  
* \[ \] **Implementation \- Monitor (10%):** Có Grafana Dashboard và Alert không? (Quân)  
* \[ \] **Testing & CI/CD (15%):** Có GitHub Actions tích xanh và unit tests (coverage \>50%) không? (Quân)  
* \[ \] **Responsible AI (10%):** Có biểu đồ SHAP/LIME giải thích model và thảo luận về Bias/Ethics không? (Trung \+ Long)  
* \[ \] **Docs (10%):** README chuẩn chỉnh không? Có file ARCHITECTURE.md không? (Long)