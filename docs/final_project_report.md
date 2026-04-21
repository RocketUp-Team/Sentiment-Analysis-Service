FPT UNIVERSITY
FSB Institute of Management and Technology
(Vien Quan tri & Cong nghe FSB, Truong Dai hoc FPT)
COURSE
DDM501
AI in Production: From Models to Systems
Duration: 4 weeks · Weight: 40%
FINAL PROJECT REPORT
Sentiment Analysis Service
End-to-End ML System: From Problem Definition to Production
Deployment
Topic: NLP – Topic 8 – Sentiment Analysis Service
Dataset: SemEval-2014 Task 4 Restaurant Reviews
ML Model: RoBERTa + DeBERTa v3 Zero-Shot ABSA
TEAM MEMBERS
# Full Name Student ID Responsibility
1 Duong Hong Quan MSA23236 Monitoring, DevOps / CI-CD
2 Pham Duc Long MSA23233 Backend API, Experiment Tracking
3 Do Quoc Trung MSA23231 ML Pipeline, Data Validation

| Final Project | Report   |             | DDM501      | – Sentiment | Analysis Service |
| ------------- | -------- | ----------- | ----------- | ----------- | ---------------- |
|               |          | Instructor: | Huynh Cong  | Viet Ngu    |                  |
|               |          | Submission: | April 2026  |             |                  |
|               | Academic | Year:       | 2025 – 2026 |             |                  |
2

Final Project Report DDM501 – Sentiment Analysis Service
Abstract
This report documents the complete lifecycle of the Sentiment Analy-
sis Service, a production-grade ML system developed as the capstone project
for the DDM501 course. The system performs real-time sentence-level senti-
ment classification (positive / negative / neutral) as well as Aspect-Based Sen-
timent Analysis (ABSA) using a pre-trained RoBERTa transformer evaluated
on restaurant-domain reviews.
The service is exposed through a RESTful FastAPI application, container-
ised with Docker, orchestrated via Docker Compose, and equipped with a full
observability stack (Prometheus + Grafana), experiment tracking (MLflow),
automated data validation (DVC), a robust test suite, and a GitHub Actions
CI/CD pipeline. A dedicated /explain endpoint delivers token-level SHAP
attributions, fulfilling the Responsible AI explainability requirement.
1

Final Project Report DDM501 – Sentiment Analysis Service
Contents
1 Problem Definition & Requirements (10%) 3
1.1 Problem Statement & Business Context . . . . . . . . . . . . . . . . . 3
1.2 User Requirements & Use Cases . . . . . . . . . . . . . . . . . . . . . . 3
1.3 Success Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.4 Scope & Constraints . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2 System Design & Architecture (15%) 4
2.1 High-Level Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2 Component Design & Responsibilities . . . . . . . . . . . . . . . . . . . 5
2.3 Data Flow . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.4 Technology Stack Justification . . . . . . . . . . . . . . . . . . . . . . . 7
3 Implementation (40%) 7
3.1 ML Pipeline (15%) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.1.1 Datasets & Machine Learning Models . . . . . . . . . . . . . . 8
3.1.2 Reproducible DVC Pipeline . . . . . . . . . . . . . . . . . . . . 8
3.1.3 Experiment Tracking with MLflow . . . . . . . . . . . . . . . . 9
3.2 Deployment (15%) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
3.2.1 REST API Design . . . . . . . . . . . . . . . . . . . . . . . . . 11
3.2.2 Docker Containerisation . . . . . . . . . . . . . . . . . . . . . . 13
3.2.3 Docker Compose Orchestration . . . . . . . . . . . . . . . . . . 14
3.3 Monitoring (10%) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
3.3.1 Prometheus Metrics . . . . . . . . . . . . . . . . . . . . . . . . 14
3.3.2 Grafana Dashboards . . . . . . . . . . . . . . . . . . . . . . . . 15
3.3.3 Alerting Rules . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4 Testing & CI/CD (15%) 15
4.1 Test Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4.1.1 Integration Test Examples . . . . . . . . . . . . . . . . . . . . . 16
4.2 CI/CD Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
5 Responsible AI (10%) 17
5.1 Model Explainability with SHAP . . . . . . . . . . . . . . . . . . . . . 17
5.1.1 Technical Implementation . . . . . . . . . . . . . . . . . . . . . 17
5.2 Fairness Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
5.3 Data Privacy Considerations . . . . . . . . . . . . . . . . . . . . . . . . 20
5.4 Ethical Implications . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
2

| Final Project   | Report |       | DDM501 | – Sentiment | Analysis Service |     |
| --------------- | ------ | ----- | ------ | ----------- | ---------------- | --- |
| 6 Documentation |        | (10%) |        |             |                  | 21  |
6.1 Repository Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
6.2 API Documentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
6.3 Setup & Deployment Guide . . . . . . . . . . . . . . . . . . . . . . . . 22
6.4 Code Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
| 7 Conclusion      |              |      |     |     |     | 23  |
| ----------------- | ------------ | ---- | --- | --- | --- | --- |
| A Full Dependency |              | List |     |     |     | 23  |
| B Individual      | Contribution |      |     |     |     | 24  |
3

| Final Project | Report |            |                | DDM501  | – Sentiment | Analysis Service |
| ------------- | ------ | ---------- | -------------- | ------- | ----------- | ---------------- |
| 1 Problem     |        | Definition | & Requirements |         |             | (10%)            |
| 1.1 Problem   |        | Statement  | & Business     | Context |             |                  |
Customer feedback represents one of the most valuable yet under-utilised assets of any
business. Manual review of large volumes of reviews, social-media posts, and support
| tickets is | both expensive | and slow. |     |     |     |     |
| ---------- | -------------- | --------- | --- | --- | --- | --- |
The Sentiment Analysis Service addresses this gap by providing an automated,
| REST-accessible |     | system that: |     |     |     |     |
| --------------- | --- | ------------ | --- | --- | --- | --- |
a) classifies the overall sentiment of a text snippet (positive · negative · neutral);
b) extractsaspect-level sentiments(food, service, ambiance, price, location, general)
| for | fine-grained | business | insight; |     |     |     |
| --- | ------------ | -------- | -------- | --- | --- | --- |
c) provides SHAP-based token explanations so that end-users can trust and audit
| model | decisions. |     |     |     |     |     |
| ----- | ---------- | --- | --- | --- | --- | --- |
TheprimarydatasetusedforofflineevaluationistheSemEval-2014 Task 4 Restau-
rant Reviews corpus, which provides sentence-level and aspect-level sentiment anno-
tations.
| 1.2 User    | Requirements |             | & Use Cases   |              |     |     |
| ----------- | ------------ | ----------- | ------------- | ------------ | --- | --- |
|             |              | Table       | 1: Functional | Requirements |     |     |
| ID Priority |              | Requirement |               |              |     |     |
FR-01 Must Have Single-text sentiment prediction via POST /predict endpoint.
FR-02 Must Have Aspect-BasedSentimentAnalysis(ABSA)returnedalongsideoverall
prediction.
FR-03 Must Have Batch processing of CSV files via POST /batch_predict.
FR-04 Must Have Model explainability (SHAP values) via POST /explain endpoint.
FR-05 Must Have Health-check endpoint for load-balancer liveness probes.
FR-06 Should Have Language guard: reject non-English inputs with HTTP 400.
FR-07 Should Have Prometheus metrics endpoint GET /metrics for observability.
| FR-08 Could | Have | Angular-based | frontend chatbot | interface. |     |     |
| ----------- | ---- | ------------- | ---------------- | ---------- | --- | --- |
4

| Final Project | Report   |     |             |                   |     | DDM501       | – Sentiment | Analysis | Service |
| ------------- | -------- | --- | ----------- | ----------------- | --- | ------------ | ----------- | -------- | ------- |
|               |          |     | Table       | 2: Non-Functional |     | Requirements |             |          |         |
| ID            | Category |     | Requirement |                   |     |              |             |          |         |
NFR-01 Performance P95 end-to-end latency < 200ms for single-text prediction.
| NFR-02 | Reliability |     | Service |     | uptime ≥ 99.9%. |     |     |     |     |
| ------ | ----------- | --- | ------- | --- | --------------- | --- | --- | --- | --- |
NFR-03 Scalability Stateless API; horizontally scalable via Docker replicas.
NFR-04 Security No secrets in Docker images; model weights baked in at build
time.
NFR-05 Maintainability Typed Python 3.11 codebase, flake8 clean, full docstrings.
NFR-06 Observability All API calls and inference latencies tracked in Prometheus.
| 1.3 | Success | Metrics |     |     |     |     |     |     |     |
| --- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- |
Table 3: Success Metrics at Business, Model, and System Levels
| Level    | Metric     |            |     |     | Target     |         |        | Tool       |         |
| -------- | ---------- | ---------- | --- | --- | ---------- | ------- | ------ | ---------- | ------- |
| Business | Prediction | Throughput |     |     | >100 req/s |         |        | Locust /   | Grafana |
| Business | ABSA       | Coverage   |     |     | Aspects    | on >80% | of in- | Prometheus |         |
puts
| Model  | Accuracy      | (test         | set)    |     | >0.85  |     |     | src.model.evaluate |           |
| ------ | ------------- | ------------- | ------- | --- | ------ | --- | --- | ------------------ | --------- |
| Model  | F1 Macro      | (test         | set)    |     | >0.80  |     |     | MLflow             |           |
| System | P95           | Latency       |         |     | <200ms |     |     | Prometheus         | histogram |
| System | 5xx Error     | Rate          |         |     | <1%    |     |     | Prometheus         | counter   |
| System | Avg Inference |               | Latency |     | <500ms |     |     | Prometheus         | histogram |
| 1.4    | Scope         | & Constraints |         |     |        |     |     |                    |           |
• In Scope: Real-time sentiment + ABSA inference, SHAP explainability, batch
CSV processing, full observability stack, data quality validation, MLflow exper-
| iment | tracking. |     |     |     |     |     |     |     |     |
| ----- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
• Out of Scope: Fine-tuning on custom data, multi-language support, user au-
| thentication |     | / rate | limiting. |     |     |     |     |     |     |
| ------------ | --- | ------ | --------- | --- | --- | --- | --- | --- | --- |
• Constraints: CPU-only Docker image; offline model serving (weights cached
| at       | build time); |        | Python | 3.11. |              |     |       |     |     |
| -------- | ------------ | ------ | ------ | ----- | ------------ | --- | ----- | --- | --- |
| 2 System |              | Design |        | &     | Architecture |     | (15%) |     |     |
5

| Final Project  | Report |              |     |     | DDM501 | – Sentiment |     | Analysis Service |
| -------------- | ------ | ------------ | --- | --- | ------ | ----------- | --- | ---------------- |
| 2.1 High-Level |        | Architecture |     |     |        |             |     |                  |
The system consists of five containerised services orchestrated by Docker Compose
on a shared bridge network (sentiment-network). Figure 1 shows the component
diagram.
RoBERTa
Inference
|     |     |     | Angular |     | FastAPI |     |     |     |
| --- | --- | --- | ------- | --- | ------- | --- | --- | --- |
MLflow
| User / Browser |     |     | Frontend |     | App     |     | logruns |         |
| -------------- | --- | --- | -------- | --- | ------- | --- | ------- | ------- |
|                |     |     | (:80)    |     | (:8000) |     |         | (:5005) |
/metrics
|     |     |     |     |     | Prometheus |     |     | Grafana |
| --- | --- | --- | --- | --- | ---------- | --- | --- | ------- |
|     |     |     |     |     | (:9091)    |     |     | (:3000) |
Figure 1: High-level component diagram – Sentiment Analysis Service.
| 2.2 Component |       | Design | &                | Responsibilities |     |     |     |     |
| ------------- | ----- | ------ | ---------------- | ---------------- | --- | --- | --- | --- |
|               |       |        | Table 4: Service | Responsibilities |     |     |     |     |
| Service       | Image |        |                  | Responsibility   |     |     |     |     |
fastapi_app python:3.11-slim (custom) Core ML inference, REST API, metrics expo-
sure
frontend Angular/Nginx Chatbot-style web UI for sentiment queries
prometheus prom/prometheus:v2.45.0 Time-series metrics scraping, alerting rules
grafana grafana/grafana:10.0.0 Dashboard visualisation, alert notification
mlflow ghcr.io/mlflow:v2.4.1 Experiment run tracking, artifacts storage
|             | Table     | 5: Resource | Limits | per Service | (Docker | Compose) |     |     |
| ----------- | --------- | ----------- | ------ | ----------- | ------- | -------- | --- | --- |
| Service     | CPU Limit |             |        |             | Memory  | Limit    |     |     |
| fastapi_app | 1.50 vCPU |             |        |             | 4GB     |          |     |     |
| frontend    | 0.25 vCPU |             |        |             | 512MB   |          |     |     |
| prometheus  | –         |             |        |             | 256MB   |          |     |     |
| grafana     | –         |             |        |             | 512MB   |          |     |     |
| mlflow      | 0.50 vCPU |             |        |             | 1GB     |          |     |     |
6

| Final Project | Report | DDM501 | – Sentiment | Analysis Service |
| ------------- | ------ | ------ | ----------- | ---------------- |
| 2.3 Data      | Flow   |        |             |                  |
1. User submits text to (or /explain) via the Angular frontend or
POST /predict
| any | HTTP client. |     |     |     |
| --- | ------------ | --- | --- | --- |
2. FastAPI validates the request body through Pydantic schemas
(PredictRequest).
3. Validated text is forwarded to BaselineModelInference.predict_single().
4. The RoBERTa tokenizer encodes the text; softmax output gives sentiment label
| and | confidence. |     |     |     |
| --- | ----------- | --- | --- | --- |
5. The DeBERTa zero-shot ABSA pipeline extracts aspect sentiments above a 0.5
threshold.
6. For /explain, a SHAP Explainer returns per-token attribution scores.
7. The response is serialised by PredictResponse and returned to the client.
8. The monitoring middleware records request counts and latencies for Prometheus.
7

| Final Project  | Report   |               |               | DDM501 |           | – Sentiment |            | Analysis |     | Service |
| -------------- | -------- | ------------- | ------------- | ------ | --------- | ----------- | ---------- | -------- | --- | ------- |
| 2.4 Technology | Stack    | Justification |               |        |           |             |            |          |     |         |
|                | Table 6: | Technology    | Decisions     | and    | Trade-off |             | Analysis   |          |     |         |
| Layer          | Choice   |               | Justification |        |           |             | Trade-off  |          |     |         |
| API            | FastAPI  |               | Async-native, |        |           | auto-       | Node.js    |          |     | higher  |
|                |          |               | generates     |        |           | Ope-        | throughput |          | for | pure    |
|                |          |               | nAPI/Swagger, |        |           |             | I/O        |          |     |         |
|                |          |               | Pydantic      |        |           | integra-    |            |          |     |         |
tion
| ML Model | RoBERTa |     | Pre-trained |           | on     | 58M   | Larger    |     | than | Dis-   |
| -------- | ------- | --- | ----------- | --------- | ------ | ----- | --------- | --- | ---- | ------ |
|          |         |     | tweets;     |           | strong | zero- | tilBERT;  |     |      | slower |
|          |         |     | shot        | sentiment |        | accu- | inference |     |      |        |
racy
| ABSA | DeBERTa | v3  | No        | labelled |          | ABSA     | Higher     |     | latency | than a |
| ---- | ------- | --- | --------- | -------- | -------- | -------- | ---------- | --- | ------- | ------ |
|      |         |     | data      | needed;  |          | flexible | fine-tuned |     | model   |        |
|      |         |     | candidate |          | category | set      |            |     |         |        |
Container Docker multi-stage Smaller image: Longer initial build
|          |        |     | builder      | installs |        | deps, | time   |     |       |        |
| -------- | ------ | --- | ------------ | -------- | ------ | ----- | ------ | --- | ----- | ------ |
|          |        |     | runtime      |          | copies | site- |        |     |       |        |
|          |        |     | packages     |          | only   |       |        |     |       |        |
| Tracking | MLflow |     | Open-source, |          |        | self- | W&B    |     | has a | richer |
|          |        |     | hosted,      |          | native | DVC   | hosted | UI  |       |        |
integration
Observability Prometheus + Grafana Industry standard, Higher operational
|     |     |     | battle-tested, |         |     | no ven- | complexity |     | vs. | SaaS |
| --- | --- | --- | -------------- | ------- | --- | ------- | ---------- | --- | --- | ---- |
|     |     |     | dor            | lock-in |     |         |            |     |     |      |
Data Versioning DVC Git-native, repro- Steeper learning curve
|     |     |     | ducible |         | pipelines, |     | than | plain | scripts |     |
| --- | --- | --- | ------- | ------- | ---------- | --- | ---- | ----- | ------- | --- |
|     |     |     | remote  | storage |            |     |      |       |         |     |
CI/CD GitHub Actions Free for public repos, Jenkins more flexible
|     |     |     | tight | GitHub |     | integra- | for | complex | pipelines |     |
| --- | --- | --- | ----- | ------ | --- | -------- | --- | ------- | --------- | --- |
tion
| 3 Implementation |     | (40%) |     |     |     |     |     |     |     |     |
| ---------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
8

| Final | Project  | Report   |         |          |        | DDM501 | – Sentiment | Analysis Service |
| ----- | -------- | -------- | ------- | -------- | ------ | ------ | ----------- | ---------------- |
| 3.1   | ML       | Pipeline | (15%)   |          |        |        |             |                  |
| 3.1.1 | Datasets | &        | Machine | Learning | Models |        |             |                  |
The system integrates diverse data sources and advanced modelling techniques to
| support | multiple | text-classification |     | tasks: |     |     |     |     |
| ------- | -------- | ------------------- | --- | ------ | --- | --- | --- | --- |
• Datasets:
– ABSA SemEval 2014 (Restaurants): Provides sentence and aspect-
|     | level | sentiment | annotations. |     | Used | for baseline | evaluation. |     |
| --- | ----- | --------- | ------------ | --- | ---- | ------------ | ----------- | --- |
– Sarcasm Dataset (tweet_eval_irony_v1): Identifies irony and sarcasm
|     | in  | social | media contexts. |     |     |     |     |     |
| --- | --- | ------ | --------------- | --- | --- | --- | --- | --- |
– Multilingual Sentiment (multilingual_sentiment_v1): English and
|     | Vietnamese |          | data    | for training | multilingual | capabilities. |     |     |
| --- | ---------- | -------- | ------- | ------------ | ------------ | ------------- | --- | --- |
| •   | Machine    | Learning | Models: |              |              |               |     |     |
– Base Model (xlm-roberta-base): Serves as a unified backbone for su-
|     | perior | cross-lingual |     | understanding. |     |     |     |     |
| --- | ------ | ------------- | --- | -------------- | --- | --- | --- | --- |
– Training Strategy (PEFT/LoRA): Parameter-Eﬀicient Fine-Tuning
via Low-Rank Adaptation (LoRA) is employed. Separate lightweight
adapters are trained for Sentiment Analysis and Sarcasm Detection, allow-
ing multiple adapters to be stacked eﬀiciently during inference to minimise
|     | memory |     | footprint. |     |     |     |     |     |
| --- | ------ | --- | ---------- | --- | --- | --- | --- | --- |
– Hardware-Aware Optimisation: The training pipeline automatically
detects available accelerators (CUDA, MPS, CPU). On supported GPUs
(e.g., L4), itautomaticallyenablesMixedPrecision(bf16), fusedoptimisers
(adamw_torch_fused), pinned memory, and tuned Data Loader workers to
|     | maximise    |     | training     | throughput. |       |                  |     |             |
| --- | ----------- | --- | ------------ | ----------- | ----- | ---------------- | --- | ----------- |
|     | – Inference |     | Optimisation |             | (ONNX | & Quantization): |     | HuggingFace |
models are exported to ONNX format (both FP32 and INT8 quantiza-
tion), significantly reducing inference latency and resource consumption in
production.
| 3.1.2 | Reproducible |     | DVC | Pipeline |     |     |     |     |
| ----- | ------------ | --- | --- | -------- | --- | --- | --- | --- |
The complete lifecycle of data and models is orchestrated via Data Version Control
(DVC). The pipeline is defined as a Directed Acyclic Graph (DAG) in dvc.yaml to
| ensure | strict            | reproducibility: |     |     |     |     |     |     |
| ------ | ----------------- | ---------------- | --- | --- | --- | --- | --- | --- |
| 1.     | Data Engineering: |                  |     |     |     |     |     |     |
• download: Automatically retrieves raw data from external sources.
• preprocess: Executes a configurable transformation chain (label map-
9

| Final Project | Report |     |     |     | DDM501 | – Sentiment | Analysis Service |
| ------------- | ------ | --- | --- | --- | ------ | ----------- | ---------------- |
ping, sentiment derivation, cleaning, deduplication, length filtering, and
|     | Train/Test | splitting | according | to params.yaml). |     |     |     |
| --- | ---------- | --------- | --------- | ---------------- | --- | --- | --- |
• validate: Enforces data quality checks (null ratios, minimum samples,
label distributions) and automatically generates quality_report.json.
2. Modeling:
• download_sarcasm/download_sentiment: Preparestask-specifictraining
datasets.
• finetune_sarcasm / finetune_sentiment: Triggers LoRA adapter train-
ing.
• evaluate_baseline / evaluate_finetuned: Evaluates models on
test splits, generating detailed reports: metrics_finetuned.json,
|               | per_language_f1.json, |              | and | fairness_report.json. |     |     |     |
| ------------- | --------------------- | ------------ | --- | --------------------- | --- | --- | --- |
| 3. Deployment |                       | Preparation: |     |                       |     |     |     |
• export_onnx_sentiment / export_onnx_sarcasm: Merges LoRA weights
|     | into the | base model | and exports | ONNX | graphs. |     |     |
| --- | -------- | ---------- | ----------- | ---- | ------- | --- | --- |
• benchmark_onnx: Measures real-world inference performance, yielding
onnx_benchmark.json.
Figure 2 illustrates the DAG of the data pipeline generated by DVC.
|       |            | Figure   | 2: DVC | pipeline | DAG. |     |     |
| ----- | ---------- | -------- | ------ | -------- | ---- | --- | --- |
| 3.1.3 | Experiment | Tracking | with   | MLflow   |      |     |     |
MLflow is deeply integrated into the source code to track the entire process from data
| validation | to model | training: |     |     |     |     |     |
| ---------- | -------- | --------- | --- | --- | --- | --- | --- |
• Data Quality Tracking: The Data Validation process automatically logs
metrics directly to the data_preprocessing experiment. Metrics include
10

| Final | Project |     | Report |     |     |     |     | DDM501 | – Sentiment | Analysis Service |
| ----- | ------- | --- | ------ | --- | --- | --- | --- | ------ | ----------- | ---------------- |
total_samples, split-specific null ratios, and a passed_quality_checks flag.
|     | The     | quality_report.json |          |           |     | is attached | as an | artifact. |     |     |
| --- | ------- | ------------------- | -------- | --------- | --- | ----------- | ----- | --------- | --- | --- |
|     | • Model |                     | Training | Tracking: |     |             |       |           |     |     |
– Integration with HuggingFace Trainer (report_to=["mlflow"]) automat-
|     |     | ically           | logs | loss, learning |            | rate, and | metrics | per                          | epoch.        |            |
| --- | --- | ---------------- | ---- | -------------- | ---------- | --------- | ------- | ---------------------------- | ------------- | ---------- |
|     |     | – Organisational |      |                | Structure: |           | Runs    |                              | are organised | into task- |
|     |     | specific         |      | experiments    |            | (e.g.,    |         | phase2_finetuning_sentiment, |               |            |
phase2_finetuning_sarcasm).
– Automatic Tagging: Each run is explicitly tagged with task, git_sha
(linking model version to code), device, and dataset_version.
|     |     | – Hyperparameter |     |     | Tracking: |     | Parameters |     | such | as learning_rate, |
| --- | --- | ---------------- | --- | --- | --------- | --- | ---------- | --- | ---- | ----------------- |
batch_size, and epochs are automatically ingested from the centralised
|     |     | params.yaml |        | and       | logged | for comparative |          | analysis. |             |     |
| --- | --- | ----------- | ------ | --------- | ------ | --------------- | -------- | --------- | ----------- | --- |
|     |     |             | Figure | 3: MLflow |        | dashboard       | overview |           | on DagsHub. |     |
11

| Final Project  | Report |             |            |     |     | DDM501  |         | – Sentiment |            | Analysis Service |
| -------------- | ------ | ----------- | ---------- | --- | --- | ------- | ------- | ----------- | ---------- | ---------------- |
|                | Figure | 4: Detailed | experiment |     | run | metrics | tracked |             | in MLflow. |                  |
| 3.2 Deployment |        |             | (15%)      |     |     |         |         |             |            |                  |
| 3.2.1          | REST   | API Design  |            |     |     |         |         |             |            |                  |
The FastAPI application (src/main.py) provides a self-documenting REST interface:
|        |          |     | Table 7: | API Endpoint |     | Reference |     |     |     |     |
| ------ | -------- | --- | -------- | ------------ | --- | --------- | --- | --- | --- | --- |
| Method | Endpoint |     |          | Description  |     |           |     |     |     |     |
GET /health Liveness probe; returns model load status and sup-
ported languages
| POST | /predict |     |     | Single-text |     | sentiment |     | + ABSA | inference |     |
| ---- | -------- | --- | --- | ----------- | --- | --------- | --- | ------ | --------- | --- |
POST /explain SHAP token-level attributions for a single text
POST /batch_predict CSV upload (multipart/form-data); returns job
ID
GET /batch_status/{job_id} Polls processing status of a batch job
| GET | /metrics |     |     | Prometheus-compatible |     |     |     | plain-text |     | metric exposi- |
| --- | -------- | --- | --- | --------------------- | --- | --- | --- | ---------- | --- | -------------- |
tion
Error Handling: Custom exception handlers return appropriate HTTP status codes:
→
| • UnsupportedLanguageError |     |        |                | HTTP |          | 400 plain-text |     | message. |     |     |
| -------------------------- | --- | ------ | -------------- | ---- | -------- | -------------- | --- | -------- | --- | --- |
| • ModelError               |     | → HTTP | 500 plain-text |      | message. |                |     |          |     |     |
→
• Model not yet loaded HTTP 503 via the get_model dependency.
Request and response schemas are enforced via Pydantic v2 with field-level validation:
| 1 class | PredictRequest(BaseModel): |     |     |     |     |     |     |     |     |     |
| ------- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
12

| Final Project | Report                    |     |                  | DDM501 | – Sentiment | Analysis Service |
| ------------- | ------------------------- | --- | ---------------- | ------ | ----------- | ---------------- |
| text:         | str = Field(min_length=1, |     | max_length=2000) |        |             |                  |
2
| lang: | str = "en" |     |     |     |     |     |
| ----- | ---------- | --- | --- | --- | --- | --- |
3
4
| class PredictResponse(BaseModel): |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | --- | --- |
5
| 6 text:       | str                 |                 |             |            |     |     |
| ------------- | ------------------- | --------------- | ----------- | ---------- | --- | --- |
| 7 sentiment:  | Literal["positive", |                 | "negative", | "neutral"] |     |     |
| 8 confidence: | float               | = Field(ge=0.0, | le=1.0)     |            |     |     |
9 aspects: list[AspectSentimentOut] = Field(default_factory=list)
| 10 sarcasm_flag: |       | bool |     |     |     |     |
| ---------------- | ----- | ---- | --- | --- | --- | --- |
| 11 latency_ms:   | float |      |     |     |     |     |
12
| 13 class ExplainResponse(BaseModel): |             |     |     |     |     |     |
| ------------------------------------ | ----------- | --- | --- | --- | --- | --- |
| 14 tokens:                           | list[str]   |     |     |     |     |     |
| 15 shap_values:                      | list[float] |     |     |     |     |     |
| 16 base_value:                       | float       |     |     |     |     |     |
| 17 latency_ms:                       | float       |     |     |     |     |     |
18
@model_validator(mode="after")
19
| def | validate_lengths(self): |     |     |     |     |     |
| --- | ----------------------- | --- | --- | --- | --- | --- |
20
|     | if len(self.tokens) | != len(self.shap_values): |     |     |     |     |
| --- | ------------------- | ------------------------- | --- | --- | --- | --- |
21
raise ValueError("tokens and shap_values must match in length")
22
return self
23
|     | Listing | 1: Core Pydantic | schemas | (contracts/schemas.py) |     |     |
| --- | ------- | ---------------- | ------- | ---------------------- | --- | --- |
Figure 5 shows the chatbot-style web interface in action, demonstrating both the
prediction and SHAP explanation capabilities via the REST API.
13

| Final Project | Report |     |     |     | DDM501 | – Sentiment | Analysis | Service |
| ------------- | ------ | --- | --- | --- | ------ | ----------- | -------- | ------- |
Figure 5: Web UI demonstrating sentiment prediction and SHAP explanation.
| 3.2.2 Docker | Containerisation |     |     |     |     |     |     |     |
| ------------ | ---------------- | --- | --- | --- | --- | --- | --- | --- |
The Dockerfile employs a multi-stage build to minimise the final image size:
• Stage 1 (builder): python:3.11-slim with build-essential; installs Py-
| Torch | (CPU | wheel) and | all requirements.txt |     | packages. |     |     |     |
| ----- | ---- | ---------- | -------------------- | --- | --------- | --- | --- | --- |
• Stage 2 (runtime): copies only site-packages and bin from the builder, plus
| application |     | source. |     |     |     |     |     |     |
| ----------- | --- | ------- | --- | --- | --- | --- | --- | --- |
• Model caching: RUN python src/model/download_models.py bakes the
| HuggingFace           |          | weights      | into the image | at      | build time.            |          |     |     |
| --------------------- | -------- | ------------ | -------------- | ------- | ---------------------- | -------- | --- | --- |
| • Offline             |          | enforcement: |                |         | TRANSFORMERS_OFFLINE=1 |          |     | and |
| HF_DATASETS_OFFLINE=1 |          |              | eliminate      | network | calls at               | runtime. |     |     |
| # Stage               | 1: build |              |                |         |                        |          |     |     |
1
| FROM python:3.11-slim |     | AS  | builder |     |     |     |     |     |
| --------------------- | --- | --- | ------- | --- | --- | --- | --- | --- |
2
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
3
| && pip | install | -r requirements.txt |     |     |     |     |     |     |
| ------ | ------- | ------------------- | --- | --- | --- | --- | --- | --- |
4
5
| # Stage | 2: lean | runtime |     |     |     |     |     |     |
| ------- | ------- | ------- | --- | --- | --- | --- | --- | --- |
6
| FROM python:3.11-slim |     |     |     |     |     |     |     |     |
| --------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
7
COPY --from=builder /usr/local/lib/python3.11/site-packages ...
8
| COPY src/ | contracts/ |     |     |     |     |     |     |     |
| --------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
9
| RUN python | src/model/download_models.py |     |     |     |     |     |     |     |
| ---------- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- |
10
14

| Final Project              | Report |     |     |     |     | DDM501 | – Sentiment | Analysis Service |
| -------------------------- | ------ | --- | --- | --- | --- | ------ | ----------- | ---------------- |
| ENV TRANSFORMERS_OFFLINE=1 |        |     |     |     |     |        |             |                  |
11
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
12
|              |     | Listing | 2: Dockerfile | multi-stage |     | build | summary |     |
| ------------ | --- | ------- | ------------- | ----------- | --- | ----- | ------- | --- |
| 3.2.3 Docker |     | Compose | Orchestration |             |     |       |         |     |
Five services are co-orchestrated in docker-compose.yml on the sentiment-network
bridge:
1 services:
| 2 fastapi_app: |     | # depends_on: | mlflow          |     |          |                |     |     |
| -------------- | --- | ------------- | --------------- | --- | -------- | -------------- | --- | --- |
| 3 frontend:    |     | # depends_on: | fastapi_app     |     |          |                |     |     |
| 4 prometheus:  |     | # mounts      | alert_rules.yml |     | +        | prometheus.yml |     |     |
| 5 grafana:     |     | # depends_on: | prometheus;     |     | mounts   | provisioning/  |     |     |
| mlflow:        |     | # standalone; | experiment      |     | tracking | UI             |     |     |
6
|     |     | Listing | 3: Service | graph | in  | docker-compose.yml |     |     |
| --- | --- | ------- | ---------- | ----- | --- | ------------------ | --- | --- |
All services share sentiment-network; inter-service DNS allows e.g.
http://mlflow:5000.
| 3.3 Monitoring   |     | (10%)   |     |     |     |     |     |     |
| ---------------- | --- | ------- | --- | --- | --- | --- | --- | --- |
| 3.3.1 Prometheus |     | Metrics |     |     |     |     |     |     |
Three Prometheus metric families are defined in src/monitoring/metrics.py:
|                    |     | Table | 8: Prometheus |         | Metric | Definitions  |           |           |
| ------------------ | --- | ----- | ------------- | ------- | ------ | ------------ | --------- | --------- |
| Metric Name        |     |       |               | Type    |        | Description  |           | Labels    |
| api_requests_total |     |       |               | Counter |        | Total        | number of | method,   |
|                    |     |       |               |         |        | API requests |           | endpoint, |
http_status
api_request_latency_seconds Histogram End-to-end re- method, endpoint
|                                 |     |     |     |           |     | quest latency |          |     |
| ------------------------------- | --- | --- | --- | --------- | --- | ------------- | -------- | --- |
| model_inference_latency_seconds |     |     |     | Histogram |     | Model         | forward- | –   |
|                                 |     |     |     |           |     | pass duration |          |     |
The monitor_middleware ASGI middleware wraps every request, computing wall-
clock duration and incrementing both counters before returning the response.
15

| Final | Project | Report |            |     |     |     |     |     | DDM501 | – Sentiment | Analysis Service |
| ----- | ------- | ------ | ---------- | --- | --- | --- | --- | --- | ------ | ----------- | ---------------- |
| 3.3.2 | Grafana |        | Dashboards |     |     |     |     |     |        |             |                  |
Grafana is provisioned automatically via the infra/grafana/provisioning/ volume
| mount, | with: |     |     |     |     |     |     |     |     |     |     |
| ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
• Datasource provisioning pointing to the internal Prometheus service.
• Pre-built dashboard JSON files visualising request rate, latency percentiles, and
|       | error-rate |     | time  | series. |     |     |     |     |     |     |     |
| ----- | ---------- | --- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- |
| 3.3.3 | Alerting   |     | Rules |         |     |     |     |     |     |     |     |
Two alert rules are defined in infra/prometheus/alert_rules.yml:
| 1 - | alert:  | HighErrorRate                                    |     |          |     |     |       |     |     |     |     |
| --- | ------- | ------------------------------------------------ | --- | -------- | --- | --- | ----- | --- | --- | --- | --- |
| 2   | expr:   | rate(api_requests_total{http_status=~"5.."}[5m]) |     |          |     |     |       |     |     |     |     |
| 3   | /       | rate(api_requests_total[5m])                     |     |          |     |     | > 0.1 |     |     |     |     |
| 4   | for: 1m |                                                  |     |          |     |     |       |     |     |     |     |
|     | labels: | { severity:                                      |     | critical |     | }   |       |     |     |     |     |
5
annotations:
6
|     | summary: | "High |     | HTTP 5xx | error | rate | on  | {{ $labels.endpoint |     | }}" |     |
| --- | -------- | ----- | --- | -------- | ----- | ---- | --- | ------------------- | --- | --- | --- |
7
|     | description: |     | "Error | rate | above | 10% | for | 1 minute." |     |     |     |
| --- | ------------ | --- | ------ | ---- | ----- | --- | --- | ---------- | --- | --- | --- |
8
9
| -   | alert: | HighInferenceLatency |     |     |     |     |     |     |     |     |     |
| --- | ------ | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
10
|     | expr: | model_inference_latency_seconds_sum |     |     |     |     |     |     |     |     |     |
| --- | ----- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
11
|     | /   | model_inference_latency_seconds_count |     |     |     |     |     | >   | 0.5 |     |     |
| --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
12
|     | for: 5m |     |     |     |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
13
|     | labels: | { severity: |     | warning | }   |     |     |     |     |     |     |
| --- | ------- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
14
annotations:
15
|     | summary: | "High |     | model inference |     | latency" |     |     |     |     |     |
| --- | -------- | ----- | --- | --------------- | --- | -------- | --- | --- | --- | --- | --- |
16
description: "Average inference latency above 500ms for 5 minutes."
17
|     |            |          |           | Listing |            | 4: Prometheus |        | alerting | rules  |     |     |
| --- | ---------- | -------- | --------- | ------- | ---------- | ------------- | ------ | -------- | ------ | --- | --- |
| 4   | Testing    |          | &         | CI/CD   |            | (15%)         |        |          |        |     |     |
| 4.1 | Test       | Strategy |           |         |            |               |        |          |        |     |     |
| The | test suite |          | in tests/ | is      | structured |               | around | three    | tiers: |     |     |
16

Final Project Report DDM501 – Sentiment Analysis Service
Table 9: Test Coverage Summary
Test Type Location Coverage
Unit – Data Trans- tests/data/ Each transform (TextCleaner, LengthFil-
forms ter, DuplicateRemover, Splitter, Sentiment-
Deriver) tested in isolation with mock
DataFrames
Unit – Model / Con- tests/model/, ModelConfig defaults, schema field con-
tracts tests/contracts/ straints, model_validator logic
Integration – API tests/test_api.py All five endpoints tested via FastAPI Test-
Client
4.1.1 Integration Test Examples
1 def test_predict(client):
2 response = client.post(
3 "/predict",
4 json={"text": "I love this product!", "lang": "en"}
5 )
6 assert response.status_code == 200
7 data = response.json()
8 assert "sentiment" in data # overall sentiment label
9 assert "confidence" in data # float in [0, 1]
10 assert "aspects" in data # ABSA aspect list
11
12 def test_explain(client):
13 response = client.post(
14 "/explain",
15 json={"text": "I love this product!", "lang": "en"}
16 )
17 assert response.status_code == 200
18 data = response.json()
19 assert "tokens" in data
20 assert "shap_values" in data
21 # Schema invariant enforced by Pydantic model_validator
22 assert len(data["tokens"]) == len(data["shap_values"])
Listing 5: Integration tests (tests/test_api.py)
4.2 CI/CD Pipeline
The GitHub Actions workflow (.github/workflows/ci.yml) triggers on every push
to main/develop and on pull-request to main:
1 on:
17

| Final | Project | Report |           |     |        |          | DDM501 | – Sentiment | Analysis Service |
| ----- | ------- | ------ | --------- | --- | ------ | -------- | ------ | ----------- | ---------------- |
| push: |         | {      | branches: |     | [main, | develop] | }      |             |                  |
2
| pull_request: |     | {   | branches: |     | [main] | }   |     |     |     |
| ------------- | --- | --- | --------- | --- | ------ | --- | --- | --- | --- |
3
4
jobs:
5
6 test:
| 7 runs-on: |     | ubuntu-latest |     |     |     |     |     |     |     |
| ---------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
8 steps:
| 9        | - uses:  | actions/checkout@v3     |           |                        |                  |                     |                     |     |     |
| -------- | -------- | ----------------------- | --------- | ---------------------- | ---------------- | ------------------- | ------------------- | --- | --- |
| 10       | - uses:  | actions/setup-python@v4 |           |                        |                  |                     |                     |     |     |
| 11       | with:    | { python-version:       |           |                        |                  | '3.11' }            |                     |     |     |
| 12       | - run:   | pip                     | install   | -r                     | requirements.txt |                     |                     |     |     |
| 13       | - run:   | pytest                  | --cov=src |                        | tests/           |                     |                     |     |     |
| 14       | - run:   | pip                     | install   | flake8                 |                  |                     |                     |     |     |
| 15       | - run:   | flake8                  | .         | --select=E9,F63,F7,F82 |                  |                     |                     |     |     |
| 16       | - run:   | flake8                  | .         | --exit-zero            |                  | --max-complexity=10 |                     |     |     |
|          |          |                         | Listing   | 6:                     | GitHub           | Actions             | CI/CD configuration |     |     |
| Pipeline |          | stages:                 |           |                        |                  |                     |                     |     |     |
| 1.       | Checkout | –                       | fetches   | source                 |                  | code on             | ubuntu-latest.      |     |     |
2. Dependency Install – installs all packages including pytest, pytest-cov,
httpx.
3. Test & Coverage – runs full suite; pytest-cov measures line coverage across
src/.
4. Lint (hard) – flake8 fails the build on syntax errors and undefined names.
5. Lint (soft) – flake8 in warning mode for complexity and line-length metrics.
| 5   | Responsible |                |     | AI  | (10%) |      |      |     |     |
| --- | ----------- | -------------- | --- | --- | ----- | ---- | ---- | --- | --- |
| 5.1 | Model       | Explainability |     |     |       | with | SHAP |     |     |
The /explain endpoint exposes token-level feature attributions using the SHAP li-
(shap≥0.42.0).
brary
| 5.1.1 | Technical |     | Implementation |     |     |     |     |     |     |
| ----- | --------- | --- | -------------- | --- | --- | --- | --- | --- | --- |
1 def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
| 2   | import                     | shap        |                                       |          |                                      |         |           |     |     |
| --- | -------------------------- | ----------- | ------------------------------------- | -------- | ------------------------------------ | ------- | --------- | --- | --- |
| 3   | self._check_language(lang) |             |                                       |          |                                      |         |           |     |     |
| 4   | # Identify                 |             | the predicted                         |          | class                                | index   |           |     |     |
| 5   | predicted_probs            |             |                                       | =        | self._predict_probabilities(text)[0] |         |           |     |     |
| 6   | predicted_class_idx        |             |                                       | =        | int(predicted_probs.argmax().item()) |         |           |     |     |
| 7   | # Wrap                     | HuggingFace |                                       | pipeline |                                      | in SHAP | Explainer |     |     |
| 8   | pipe                       |             | = self._get_classification_pipeline() |          |                                      |         |           |     |     |
18

| Final Project | Report |                      |     | DDM501 | – Sentiment | Analysis Service |
| ------------- | ------ | -------------------- | --- | ------ | ----------- | ---------------- |
| explainer     | =      | shap.Explainer(pipe) |     |        |             |                  |
9
| shap_values | =   | explainer([text]) |     |     |     |     |
| ----------- | --- | ----------------- | --- | --- | --- | --- |
10
# Extract token strings and SHAP scores for the predicted class
11
| raw_tokens | = shap_values.data[0] |     |     |     |     |     |
| ---------- | --------------------- | --- | --- | --- | --- | --- |
12
13 tokens = raw_tokens.tolist() if hasattr(raw_tokens, "tolist") else list(
raw_tokens)
14 values = shap_values.values[0][:, predicted_class_idx].tolist()
15 base = float(shap_values.base_values[0][predicted_class_idx])
16 return SHAPResult(tokens=tokens, shap_values=values, base_value=base)
|                 | Listing            | 7: SHAP explanation | in            | BaselineModelInference |        |     |
| --------------- | ------------------ | ------------------- | ------------- | ---------------------- | ------ | --- |
| The explanation | workflow:          |                     |               |                        |        |     |
| 1. The          | text is classified | to determine        | the predicted | class                  | index. |     |
2. shap.Explainer computes Shapley values by marginalising over permutations
| of input | tokens. |     |     |     |     |     |
| -------- | ------- | --- | --- | --- | --- | --- |
3. Per-token SHAP scores for the predicted class are extracted alongside the base
value.
4. The ExplainResponse schema enforces len(tokens) == len(shap_values)
| via | model_validator. |     |     |     |     |     |
| --- | ---------------- | --- | --- | --- | --- | --- |
Figure 6 and Figure 7 demonstrate SHAP waterfall plots for specific test samples,
| highlighting | the token-level | sentiment | attributions. |     |     |     |
| ------------ | --------------- | --------- | ------------- | --- | --- | --- |
19

| Final Project | Report | DDM501 | – Sentiment | Analysis Service |
| ------------- | ------ | ------ | ----------- | ---------------- |
Figure 6: SHAP explanation highlighting negative aspects regarding food and service.
20

| Final Project | Report |     |     | DDM501 | – Sentiment | Analysis Service |
| ------------- | ------ | --- | --- | ------ | ----------- | ---------------- |
Figure 7: SHAP explanation indicating a negative sentiment prediction.
| 5.2 Fairness | Analysis |     |     |     |     |     |
| ------------ | -------- | --- | --- | --- | --- | --- |
Known Biases. The cardiffnlp/twitter-roberta-base-sentiment-latest
| model was | pre-trained | on Twitter | data, introducing: |     |     |     |
| --------- | ----------- | ---------- | ------------------ | --- | --- | --- |
• Domain bias: informal/social-media language is better handled than formal
text.
• Language bias: English only; non-English inputs are blocked by the language
guard.
• Sarcasm: The sarcasm_flag field is reserved but not yet populated – a known
limitation.
Dataset Bias Mitigation. The DataQualityValidator emits distribution warnings
when any class falls below 5% in a split, enabling early detection of imbalance before
evaluation.
| 5.3 Data | Privacy | Considerations |     |     |     |     |
| -------- | ------- | -------------- | --- | --- | --- | --- |
• No user input data is persisted; the service is stateless – all processing is in-
| memory | per request. |     |     |     |     |     |
| ------ | ------------ | --- | --- | --- | --- | --- |
21

| Final | Project | Report |     |     |     |     | DDM501 |     | – Sentiment | Analysis Service |
| ----- | ------- | ------ | --- | --- | --- | --- | ------ | --- | ----------- | ---------------- |
• NoPIIisloggedbythemonitoringmiddleware; onlyHTTPmethod, path, status
|     | code, | and | timing | are recorded. |     |     |     |     |     |     |
| --- | ----- | --- | ------ | ------------- | --- | --- | --- | --- | --- | --- |
• Model weights are stored inside the Docker image; no external registry calls
|      | occur   | at  | runtime.     |             |              |      |          |     |     |        |
| ---- | ------- | --- | ------------ | ----------- | ------------ | ---- | -------- | --- | --- | ------ |
| 5.4  | Ethical |     | Implications |             |              |      |          |     |     |        |
|      |         |     |              | Table       | 10: Ethical  | Risk | Register |     |     |        |
| Risk |         |     |              | Implication | & Mitigation |      |          |     |     | Status |
Biased training data RoBERTa pre-trained on Twitter; may disadvan- Acknowledged
|     |     |     |     | tageformaltext.   | Futurefine-tuningondiversecor- |     |     |     |     |     |
| --- | --- | --- | --- | ----------------- | ------------------------------ | --- | --- | --- | --- | --- |
|     |     |     |     | pora recommended. |                                |     |     |     |     |     |
Misclassification harm Incorrect label could negatively influence business SHAP done
|     |     |     |     | decisions.  | SHAP             | explanations |     | and | human-in-the- |     |
| --- | --- | --- | --- | ----------- | ---------------- | ------------ | --- | --- | ------------- | --- |
|     |     |     |     | loop review | are recommended. |              |     |     |               |     |
Sarcasm detection Missed sarcasm leads to reversed polarity. Planned
|     |     |     |     | sarcasm_flag | reserved; |     | dedicated | sarcasm | model |     |
| --- | --- | --- | --- | ------------ | --------- | --- | --------- | ------- | ----- | --- |
planned.
Job displacement May reduce manual review roles. System is posi- Discussed
|     |               |     |     | tioned    | as decision-support, |     | not | a replacement. |     |     |
| --- | ------------- | --- | --- | --------- | -------------------- | --- | --- | -------------- | --- | --- |
| 6   | Documentation |     |     |           | (10%)                |     |     |                |     |     |
| 6.1 | Repository    |     |     | Structure |                      |     |     |                |     |     |
Sentiment-Analysis-Service/
1
|     | .github/workflows/ci.yml |     |     |     | # GitHub | Actions | CI  | pipeline |     |     |
| --- | ------------------------ | --- | --- | --- | -------- | ------- | --- | -------- | --- | --- |
2
|     | app/ |     |     |     | # Angular | frontend |     | + Dockerfile |     |     |
| --- | ---- | --- | --- | --- | --------- | -------- | --- | ------------ | --- | --- |
3
|     | contracts/ |     |     |     | # Shared | interfaces, |     | schemas, | errors |     |
| --- | ---------- | --- | --- | --- | -------- | ----------- | --- | -------- | ------ | --- |
4
| 5   | model_interface.py |     |       |             | # Abstract    | ModelInference |         |           | base class |     |
| --- | ------------------ | --- | ----- | ----------- | ------------- | -------------- | ------- | --------- | ---------- | --- |
| 6   | schemas.py         |     |       |             | # Pydantic    | API            | schemas |           |            |     |
| 7   | errors.py          |     |       |             | # Custom      | exceptions     |         |           |            |     |
| 8   | data/              |     |       |             | # DVC-managed |                | data    | artefacts |            |     |
| 9   | external/,         |     | raw/, | processed/, | reports/      |                |         |           |            |     |
10 infra/
| 11  | prometheus/           |     |     |     | # prometheus.yml, |     |             | alert_rules.yml |      |     |
| --- | --------------------- | --- | --- | --- | ----------------- | --- | ----------- | --------------- | ---- | --- |
| 12  | grafana/provisioning/ |     |     |     | # Datasource      |     | + dashboard |                 | JSON |     |
13 src/
| 14  | main.py |        |           |     | # FastAPI  | app      | entrypoint |            |     |     |
| --- | ------- | ------ | --------- | --- | ---------- | -------- | ---------- | ---------- | --- | --- |
| 15  | api/,   | core/, | services/ |     | # Routers, | logic,   |            | helpers    |     |     |
| 16  | data/   |        |           |     | # Data     | pipeline | &          | transforms |     |     |
22

Final Project Report DDM501 – Sentiment Analysis Service
17 model/ # Baseline model, config, evaluate
18 monitoring/metrics.py # Prometheus metrics + middleware
19 tests/ # Pytest test suite
20 Dockerfile # Multi-stage Docker build
21 docker-compose.yml # Five-service orchestration
22 dvc.yaml # Reproducible ML pipeline stages
23 params.yaml # Experiment & pipeline parameters
24 requirements.txt # Python dependencies (pinned ranges)
25 README.md # Setup & usage guide
26 ARCHITECTURE.md # System design documentation
Listing 8: Top-level repository layout
6.2 API Documentation
FastAPI auto-generates a fully interactive Swagger UI at http://localhost:8000
/docs and a ReDoc interface at http://localhost:8000/redoc from the Pydantic
schemas and endpoint decorators, satisfying the OpenAPI/Swagger documentation
requirement without additional tooling.
6.3 Setup & Deployment Guide
1 # Clone the repository
2 git clone <repo-url>
3 cd Sentiment-Analysis-Service
4
5 # Launch full stack
6 docker-compose up --build
7
8 # Services:
9 # API http://localhost:8000/docs
10 # Grafana http://localhost:3000 (admin / admin)
11 # MLflow http://localhost:5005
12 # Prometheus http://localhost:9091
13
14 # Run tests locally (Python 3.11 environment required)
15 pip install -r requirements.txt
16 pytest --cov=src tests/
17
18 # Re-run DVC pipeline (data + preprocess + validate + evaluate)
19 dvc repro
Listing 9: Quick-start commands
23

Final Project Report DDM501 – Sentiment Analysis Service
6.4 Code Quality
• All public functions and classes carry docstrings.
• Type hints throughout (from __future__ import annotations).
• Immutable configuration via @dataclass(frozen=True) in ModelConfig.
• Separation of concerns: contracts/ holds shared interfaces; business logic in
src/; infrastructure in infra/.
• All modules are flake8-clean (E9/F-series hard errors pass in CI).
7 Conclusion
The Sentiment Analysis Service successfully demonstrates a complete, production-
grade ML system lifecycle:
1. Problem Definition: Clearly motivated by business need; success metrics de-
fined at business, model, and system levels.
2. Architecture: Microservices design with full observability and experiment
tracking, all resource-limited via Docker Compose.
3. ML Pipeline: Reproducible data pipeline (DVC) with data-quality validation,
configurable preprocessing transforms, and MLflow-tracked evaluations of the
RoBERTa baseline.
4. Deployment: Multi-stage Docker image with offline model serving; REST API
with Pydantic schemas; health checks and CORS middleware.
5. Monitoring: Three Prometheus metrics via ASGI middleware; Grafana dash-
boards auto-provisioned; two alerting rules for error rate and inference latency.
6. Testing & CI/CD: Three test layers (unit, integration, data quality); GitHub
Actions lints and tests on every commit.
7. Responsible AI:SHAPtokenattributionsvia/explain; biasacknowledgment;
privacy-preserving stateless design; ethical risk register.
A Full Dependency List
1 dvc>=3.0,<4.0
2 mlflow>=2.0,<3.0
3 pydantic>=2.0,<3.0
4 fastapi>=0.110.0
5 uvicorn>=0.27.0
6 prometheus-client>=0.20.0
24

| Final Project | Report |     | DDM501 | – Sentiment | Analysis Service |
| ------------- | ------ | --- | ------ | ----------- | ---------------- |
python-multipart>=0.0.9
7
PyYAML>=6.0,<7.0
8
scikit-learn>=1.0,<2.0
9
pytest>=8.0
10
11 pytest-cov>=5.0
12 httpx>=0.27.0
13 pandas>=2.2.0
14 python-dotenv>=1.0.1
15 torch>=2.0.0
16 transformers>=4.30.0
17 shap>=0.42.0
18 matplotlib>=3.7.0
|     |     | Listing 10: requirements.txt |     |     |     |
| --- | --- | ---------------------------- | --- | --- | --- |
B Individual Contribution
|        | Table 11: | Team Member   | Responsibilities |     |     |
| ------ | --------- | ------------- | ---------------- | --- | --- |
| Member | Primary   | Contributions |                  |     |     |
Dương Hồng Quân Monitoring stack (Prometheus + Grafana), alerting rules, data valida-
|     | tion (src/data/validators.py), |     | testing |     |     |
| --- | ------------------------------ | --- | ------- | --- | --- |
Phạm Đức Long FastAPI application, Pydantic schemas, Docker containerisation,
|     | CI/CD pipeline | (GitHub | Actions) |     |     |
| --- | -------------- | ------- | -------- | --- | --- |
Đỗ Quốc Trung ML pipeline design, model evaluation (src/model/), MLflow integra-
|     | tion, DVC | pipeline configuration |     |     |     |
| --- | --------- | ---------------------- | --- | --- | --- |
25
