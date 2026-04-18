# 🛡️ Credit-Shield MLOps: Hybrid Sentiment & Risk Intelligence

[![MLOps](https://img.shields.io/badge/MLOps-Ready-blueviolet?style=for-the-badge&logo=gitbook)](https://github.com/features/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-121212?style=for-the-badge&logo=chainlink)](https://blog.langchain.dev/)

**Credit-Shield** is a production-grade MLOps ecosystem designed to transform customer feedback into actionable banking intelligence. By combining high-speed local inference with large-scale LLM reasoning, it provides a robust solution for credit risk assessment and churn mitigation.

---

## 🚀 Project Value

In the modern banking industry, understanding the "voice of the customer" is not just about service—it's about risk management. Credit-Shield solves three critical challenges:

1.  **Churn Reduction**: Identified negative sentiment (at a 99%+ confidence level) triggers immediate retention workflows, preventing capital flight to competitors.
2.  **Reputational Risk Mitigation**: Early detection of systemic complaints (e.g., high interest rates) allows the bank to adjust policies before public escalation.
3.  **Actionable Credit Intelligence**: Beyond simple "positive/negative" labels, the system provides professional risk reports with specific commercial proposals.

---

## 🏗️ High-Level Architecture

The system utilizes a **Hybrid AI Architecture** to balance performance, cost, and depth of insight:

1.  **Local Inference Engine (Edge)**:
    *   **Model**: DistilBERT-base-uncased.
    *   **Deployment**: FastAPI in Docker.
    *   **Purpose**: High-throughput, low-latency sentiment classification (POSITIVE/NEGATIVE).
2.  **Intelligence Layer (Orchestration)**:
    *   **Engine**: LangChain + Groq.
    *   **Model**: `llama-3.3-70b-versatile`.
    *   **Purpose**: Expert-level reasoning. It interprets technical sentiment scores to generate executive reports as a "Senior Banking Risk Analyst."

---

## 🛠️ Tech Stack

| Category | Tools | Purpose |
| :--- | :--- | :--- |
| **Microservices** | FastAPI, Uvicorn | High-performance API serving |
| **Inference/ML** | PyTorch, Transformers | Local model execution & manual inference |
| **AI Orchestration** | LangChain, Groq | LLM reasoning & prompt engineering |
| **Testing** | Pytest, TestClient | 100% endpoint & logic validation |
| **DevOps** | Docker, python-dotenv | Containerization & environment management |
| **Observability** | Custom Logging | Performance and model tracking |

---

## 🏎️ Quick Start

### 1. Installation & Environment
Clone the repo and install dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```
Create a `.env` file and add your credentials:
```text
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Download Model Artifacts
Ensure you have the latest DistilBERT weights:
```bash
python src/download_model.py
```

### 3. Containerization (Production)
Build and run the sentiment service:
```bash
docker build -t credit-sentiment-api -f docker/Dockerfile .
docker run -p 8000:8000 credit-sentiment-api
```

### 4. Run the Risk Agent
Execute the agentic pipeline to process a sample customer case:
```bash
python agents/risk_agent.py
```

---

## 🧪 Testing

Reliability is at the core of MLOps. The project includes a comprehensive test suite to ensure the API remains fault-tolerant:
```bash
$env:PYTHONPATH="."
pytest tests/
```
*   **Unit Tests**: Endpoint validation and sentiment logic.
*   **Robustness**: Mocking for model availability and inference failure handling.

---

## 📝 Example Output

The following report demonstrates the agent's ability to transform a 99.95% Negative score into a strategic banking decision:

```text
==================================================
EJECUCIÓN DEL AGENTE DE RIESGO (GROQ ENGINE)
==================================================
**Informe Ejecutivo: Análisis de Riesgo Bancario**

**1. Impacto de Riesgo:**
El comentario indica una insatisfacción crítica con las tasas de interés, lo que genera un riesgo de fuga de capitales. La mención de trasladar fondos a otra institución representa una pérdida directa de LTV (Lifetime Value).

**2. Propuesta de Acción Bancaria:**
- Revisión y ajuste proactivo de la tasa de interés del préstamo.
- Oferta de productos financieros alternativos (X-Sell) que mejoren la percepción de valor.
- Comunicación prioritaria a través del equipo de retención (Wealth Management).
==================================================
```

---

## 🗺️ Roadmap & Future Work
- [ ] **Infrastructure**: CI/CD pipelines with GitHub Actions.
- [ ] **Deployment**: Target production on AWS SageMaker.
- [ ] **Monitoring**: Integration with Prometheus & Grafana for drift detection.
- [ ] **Scale**: Support for multi-lingual sentiment analysis.

---

## 💻 Hardware Optimization Note
This project is optimized to run efficiently on **commercial grade hardware** (e.g., NVIDIA GTX 1660 Super / 16GB RAM) through manual PyTorch inference and optimized Transformer weights, ensuring a cost-effective MLOps development cycle.
