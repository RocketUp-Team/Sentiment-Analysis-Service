# **KẾ HOẠCH TRIỂN KHAI DỰ ÁN (TASK BREAKDOWN)**

**Dự án:** Sentiment Analysis Service (Topic 8\)

**Mục tiêu:** Hoàn thành End-to-End ML System đáp ứng 100% Rubric môn DDM501.

---

## **MỤC TIÊU CHIẾN LƯỢC**

1. **MVP First:** Ưu tiên xây dựng một luồng chạy được xuyên suốt từ UI \-> API \-> Model \-> UI càng sớm càng tốt. KHÔNG sa đà vào train model điểm cao ngay từ đầu.
2. **Mocking:** Quân và Long sẽ dùng dữ liệu giả (Mock data) dựa trên API Contract để code ngay lập tức, không chờ Trung train xong model.
3. **Pre-trained Models:** Trung nên dùng model có sẵn (VD: xlm-roberta-base-sentiment hoặc philschmid/distilbert-base-multilingual-cased-sentiment-2) để làm Baseline, sau đó mới fine-tune (PEFT/LoRA) trên tập data dự án.

---

## **SUCCESS METRICS (ĐA TẦNG)**

> Phụ trách: Long & Quân — cần định nghĩa rõ ngay từ đầu để đưa vào Document & Slide.

### Business Level (Kinh doanh)
| Metric | Target |
|--------|--------|
| Tỷ lệ phân loại đúng sentiment tổng thể | ≥ 85% |
| Tỷ lệ phát hiện đúng khía cạnh (ABSA) | ≥ 75% |
| Phần trăm review tiêu cực được phát hiện đúng (Recall) | ≥ 90% |

### System Level (Hệ thống)
| Metric | Target |
|--------|--------|
| Inference Latency (Real-time, 1 request) | ≤ 200ms (p95) |
| Batch Throughput | ≥ 100 samples/giây |
| API Uptime | ≥ 99.5% |
| Test Coverage | > 80% |

### Model Level (Mô hình)
| Metric | Target |
|--------|--------|
| F1-score (Macro) | ≥ 0.80 |
| Accuracy (3-class: pos/neg/neu) | ≥ 85% |
| ABSA Aspect F1 | ≥ 0.70 |
| Sarcasm Detection Precision | ≥ 0.70 |

---

## **GIAI ĐOẠN 1: KICK-OFF & CHỐT API CONTRACT**

*Tất cả 3 thành viên họp để chốt các vấn đề sau:*

1. **Chốt Architecture & Tech Stack (kèm Trade-off Analysis):**
   * Model: HuggingFace Transformers (PyTorch).
   * Backend: FastAPI (có sẵn Swagger UI cho API Docs).
   * Frontend: Streamlit (code bằng Python, cực nhanh) hoặc React/HTML đơn giản.
   * MLOps/DevOps: MLflow (Tracking), Docker Compose, Prometheus \+ Grafana.
   * Message Broker (Real-time pipeline): Cân nhắc đưa **Kafka hoặc RabbitMQ** vào luồng xử lý real-time để thể hiện tính chuyên nghiệp của hệ thống.
   * **⚠️ Bắt buộc:** Long tổng hợp phần **Trade-off Analysis** dưới đây vào tài liệu.

2. **Chốt API Contract (Cực kỳ quan trọng):**
   * Định nghĩa chính xác Request/Response cho 2 Endpoints chính: Real-time và Batch.
   * *Ví dụ Request Real-time:* `{"text": "Món ăn quá tệ, phục vụ chậm!", "lang": "vi"}`
   * *Ví dụ Response:* `{"sentiment": "negative", "confidence": 0.95, "aspects": [{"aspect": "food", "sentiment": "negative"}, {"aspect": "service", "sentiment": "negative"}], "sarcasm_flag": false, "latency_ms": 45}`

---

## **GIAI ĐOẠN 2: PHÂN CÔNG NHIỆM VỤ CHI TIẾT**

### **1\. Trung (AI Core & Modeling) — *Trọng số Rubric: 25% (ML Pipeline & Responsible AI)***

**Nhiệm vụ trọng tâm:** Xử lý dữ liệu, huấn luyện mô hình, xuất model weights và viết hàm xử lý core AI.

#### Dữ liệu & Tiền xử lý
* Tải subset nhỏ của Amazon/Yelp/Twitter (không dùng full data để tiết kiệm thời gian train ban đầu).
* Viết script **Preprocessing chuyên sâu cho Social Media/Review data**:
  * Xử lý emoji (chuyển thành text có nghĩa hoặc loại bỏ).
  * Chuẩn hóa slang/tiếng lóng (mapping dictionary tiếng Việt & tiếng Anh).
  * Xử lý hashtag (tách từ, loại bỏ ký hiệu `#`).
  * Loại bỏ URL, mention (`@user`), ký tự lặp, khoảng trắng thừa.
* Viết **Data Quality Validation script**: kiểm tra missing values, nhãn không hợp lệ, phân phối lớp (class imbalance). Cân nhắc dùng **DVC** hoặc ít nhất viết script kiểm tra **data drift** khi thêm data mới.

#### Baseline Model & Fine-tuning
* Khởi tạo Baseline model bằng pre-trained model. Gói model thành 1 class Python (`ModelInference`) có hàm `predict_single(text)` và `predict_batch(list_texts)`.
* Tích hợp **MLflow** để track metrics (F1-score, Accuracy) và params.
* Áp dụng PEFT/LoRA để fine-tune nhẹ trên tập data dự án nhằm xử lý Sarcasm hoặc đa ngôn ngữ.
* Viết script xử lý Batch Job (đọc từ file CSV/JSON, chạy qua model, xuất ra file kết quả).

#### **[BẮT BUỘC] Aspect-Based Sentiment Analysis (ABSA)**
* **KHÔNG chỉ phân loại cảm xúc chung chung.** Phải trích xuất cả *khía cạnh* (aspect) và *cảm xúc tương ứng*.
* Ví dụ output ABSA: `"Món ăn ngon nhưng phục vụ chậm"` → `[{"aspect": "food", "sentiment": "positive"}, {"aspect": "service", "sentiment": "negative"}]`
* Implement logic ABSA: có thể dùng rule-based (keyword matching cho domain restaurant/e-commerce) kết hợp model hoặc fine-tune riêng một ABSA head.
* Đảm bảo output ABSA có trong Response schema của API (đã định nghĩa ở Giai đoạn 1).

#### **[BẮT BUỘC] Tối ưu hóa Latency (Inference Optimization)**
* *(Đã chuyển từ Optional → Bắt buộc)* Áp dụng ít nhất **một** trong các kỹ thuật sau:
  * **Quantization INT8** (giảm kích thước model, tăng tốc inference).
  * **ONNX Runtime** (export model sang định dạng ONNX để inference nhanh hơn).
  * **TensorRT** (nếu chạy trên GPU NVIDIA).
* Benchmark trước/sau tối ưu: đo latency với N=1 và N=100 samples, ghi vào MLflow.

#### Responsible AI
* Viết code chạy **SHAP hoặc LIME** để giải thích mô hình (Explainability — Yêu cầu bắt buộc của Rubric). Xuất plot ảnh đưa cho Long.
* **[BẮT BUỘC] Phân tích Fairness & Bias Detection:**
  * Kiểm tra xem model có bias với ngôn ngữ không (VD: tiếng Anh vs. tiếng Việt)?
  * Kiểm tra bias theo domain (restaurant vs. e-commerce).
  * Tạo báo cáo nhỏ (có thể là Jupyter notebook) để Long dùng trong Report/Slide.

#### **[BẮT BUỘC] Model Validation Tests**
* Viết **model validation tests** (dùng pytest):
  * Test với tập dữ liệu cố định (golden set) đảm bảo F1 ≥ threshold.
  * Test ABSA output đúng schema.
  * Test latency ≤ 200ms với single request.

---

### **2\. Quân (Backend, DevOps & MLOps) — *Trọng số Rubric: 40% (Deployment, Monitoring, CI/CD)***

**Nhiệm vụ trọng tâm:** Xây dựng cầu nối, triển khai hệ thống và giám sát.

#### API & Containerization
* Khởi tạo FastAPI project. Viết các routers `/predict` và `/batch_predict` trả về Mock Data (theo API Contract) để Long làm UI.
* Viết Dockerfile tối ưu cho FastAPI và `docker-compose.yml` gồm các service: `fastapi_app`, `prometheus`, `grafana`, `mlflow`.
* *(Nâng cao)* Cân nhắc thêm **Kafka/RabbitMQ** service vào `docker-compose.yml` cho luồng real-time processing.

#### Integration & Monitoring
* Tích hợp class `ModelInference` của Trung vào FastAPI. Chạy thử luồng thật, nghiệm thu Latency.
* Gắn middleware vào FastAPI để export metrics (Request count, Error rate, Model Inference Latency) ra định dạng cho Prometheus.
* Cấu hình Dashboard trên **Grafana** (Vẽ biểu đồ request/giây, độ trễ, phân phối cảm xúc). Thiết lập Alert rule cơ bản.
* **[BẮT BUỘC] Data Drift & Model Drift Tracking:**
  * Thêm metric theo dõi sự thay đổi phân phối input (VD: % negative sentiment theo thời gian).
  * Viết script kiểm tra **data drift** (có thể dùng Evidently AI hoặc tự viết thống kê đơn giản).
  * Thêm panel Drift vào Grafana Dashboard.

#### **[BẮT BUỘC] Testing — Hướng tới Coverage > 80%**
Phải có đủ **4 loại test** sau:

1. **Unit Tests** (pytest): Test từng hàm xử lý logic, helper functions.
2. **Integration Tests** (pytest + httpx/TestClient): Test toàn bộ API endpoints (`/predict`, `/batch_predict`, `/health`). Kiểm tra response schema, HTTP status codes, edge cases.
3. **Data Quality Tests**: Test script preprocessing — đảm bảo output đúng format, không có null sau xử lý, đúng encoding.
4. **Model Validation Tests**: Phối hợp với Trung để chạy golden set test trong CI pipeline.

**Target: Test coverage ≥ 80%** (dùng `pytest-cov` để đo và export report).

#### CI/CD Pipeline (GitHub Actions)
* Viết GitHub Actions file (`.github/workflows/main.yml`) chạy **tất cả 4 loại test** mỗi khi có push/PR.
* Pipeline bao gồm: lint (flake8/black) → unit tests → integration tests → data tests → model tests → build Docker image (nếu tất cả pass).
* Publish coverage report (badge hoặc artifact) lên PR.

---

### **3\. Long (Frontend, Documentation & Report) — *Trọng số Rubric: 35% (Problem Def, Design, UI, Docs)***

**Nhiệm vụ trọng tâm:** Giao diện người dùng, Tài liệu hóa hệ thống và Slide thuyết trình.

#### Problem Definition & Architecture Design
* Viết phần **Problem Statement** và **Requirements** chi tiết.
* **[BẮT BUỘC] Định nghĩa Success Metrics đa tầng** (Business / System / Model) — xem bảng ở đầu file.
* Vẽ **Data Flow Diagram** (dùng Draw.io/Lucidchart): bao gồm luồng real-time (đồng bộ qua API) và luồng batch.
* **[BẮT BUỘC] Trade-off Analysis Document:** Viết tài liệu giải thích lý do lựa chọn từng thành phần Tech Stack và phân tích trade-off:
  * **Scalability:** Tại sao chọn FastAPI thay vì Flask? Tại sao Docker Compose thay vì Kubernetes?
  * **Cost:** Ưu/nhược điểm về chi phí vận hành vs. hiệu năng.
  * **Complexity:** Tại sao chọn Streamlit thay vì React (hoặc ngược lại)?
  * *(Nâng cao)* Tại sao cân nhắc Kafka cho real-time thay vì xử lý đồng bộ trực tiếp?

#### Frontend UI (Streamlit/React)
* Build UI kết nối API Backend của Quân.
* Thiết kế 2 luồng tính năng chính:
  1. *Real-time:* Ô text nhập liệu \-> Nút submit \-> Hiển thị kết quả (bảng cảm xúc, **ABSA aspects breakdown**, biểu đồ SHAP giải thích từ Trung).
  2. *Batch Job:* Nút upload file CSV \-> Nút process \-> Nút download kết quả \+ Hiển thị biểu đồ thống kê tổng quan (Pie chart, **Aspect distribution**).

#### Documentation & Report (LaTeX)
* Tạo cấu trúc repository chuẩn (có `.gitignore`, `requirements.txt`).
* **[BẮT BUỘC] Tạo file `CONTRIBUTING.md`**: ghi rõ vai trò, trách nhiệm của từng thành viên, quy trình tạo branch, commit convention, và cách submit PR.
* Viết `README.md` cực kỳ chi tiết (Hướng dẫn setup từng bước, run docker-compose, link đến API docs).
* **[BẮT BUỘC] Fairness & Data Privacy (trong Report/Slide):**
  * Tổng hợp kết quả Fairness/Bias analysis từ Trung vào Report.
  * Viết phần Data Privacy: dữ liệu người dùng được xử lý như thế nào? Có lưu trữ không? GDPR-aware?
  * Trình bày trong Slide (ít nhất 1 slide riêng về Responsible AI).
* Tổng hợp tất cả nội dung vào template LaTeX thành Report hoàn chỉnh. Đảm bảo format theo chuẩn học thuật/ngành.

---

## **GIAI ĐOẠN 3: TÍCH HỢP, KIỂM THỬ & NGHIỆM THU**

* **End-to-End Testing & Bug Fixing:**
  * Ba người cùng test toàn bộ luồng: Upload CSV \-> FE gọi BE \-> BE gọi Model \-> Trả kết quả FE (bao gồm cả ABSA output).
  * Quân check lại Grafana xem metrics có nhảy không khi test, bao gồm cả Drift panel.
  * Trung đẩy model weights nhẹ nhất lên repo (hoặc load qua HuggingFace) để Docker build không bị lỗi quá tải.
* **Slide & Presentation Rehearsal:**
  * Long hoàn thiện Slide (Canva/PPT) — bao gồm slide về Success Metrics, Trade-off, Fairness/Privacy.
  * Chuẩn bị Kịch bản Live Demo (Tránh rủi ro lúc demo thật).
  * Rehearsal Q&A: Chia nhau trả lời câu hỏi dựa trên Rubric (Trung trả lời về Model/ABSA/Sarcasm/Quantization, Quân trả lời về Latency/Docker/CI/CD/Drift, Long trả lời về UI/Ethics/Business flow/Trade-off).

---

## **LƯU Ý VỀ GIT COMMITS (ĐIỂM CÁ NHÂN)**

> ⚠️ **Rubric chỉ rõ điểm cá nhân có thể dao động ±20%** dựa trên tần suất, chất lượng và sự phân bổ của các Git commits.

* **KHÔNG để một người push code toàn bộ.**
* **Trung** tự push: preprocessing scripts, model training code, ABSA logic, ONNX export, SHAP plots, model validation tests.
* **Quân** tự push: FastAPI routers, Docker files, Prometheus/Grafana config, GitHub Actions, test files, drift tracking scripts.
* **Long** tự push: UI code, Documentation files (README, CONTRIBUTING, LaTeX report), Data Flow diagrams, Trade-off document.
* Sử dụng branch riêng cho từng feature, tạo PR để review chéo trước khi merge vào `main`.

---

## **CHECKLIST KIỂM TRA CHÉO THEO RUBRIC DDM501**

* \[ \] **Problem & Req (10%):** Có Problem Statement, Requirements, và **Success Metrics đa tầng** (Business/System/Model) rõ ràng chưa? (Long)
* \[ \] **Architecture (15%):** Có sơ đồ luồng dữ liệu, **Trade-off Analysis** giải thích lý do chọn Tech Stack (Scalability/Cost/Complexity) chưa? (Long + Quân)
* \[ \] **Implementation — ML (15%):** Có MLflow tracking params/metrics không? Có **ABSA** không? ONNX/Quantization đã làm chưa? (Trung)
* \[ \] **Implementation — Deploy (15%):** Có docker-compose chạy 1 lệnh là lên toàn bộ không? Swagger API docs có không? (Quân)
* \[ \] **Implementation — Monitor (10%):** Có Grafana Dashboard và Alert không? Có **Drift Tracking panel** không? (Quân)
* \[ \] **Testing & CI/CD (15%):** Có GitHub Actions tích xanh và **4 loại tests** (Unit, Integration, Data, Model) với **coverage > 80%** không? (Quân + Trung)
* \[ \] **Responsible AI (10%):** Có biểu đồ SHAP/LIME, **Fairness/Bias analysis**, và **Data Privacy** discussion không? (Trung + Long)
* \[ \] **Docs (10%):** README chuẩn chỉnh không? Có file `ARCHITECTURE.md` và **`CONTRIBUTING.md`** không? (Long)