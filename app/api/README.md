# FastAPI Service & REST Endpoint Registry

The `app/api/` module provides asynchronous HTTP REST endpoints for client applications, job management, and operational monitoring.

---

## Endpoint Specifications

### `POST /analyze`
Initiates an asynchronous entity risk research task.

- **Request Body**:
  ```json
  {
    "entity_name": "Tesla Inc",
    "ticker": "TSLA"
  }
  ```
- **Response**:
  ```json
  {
    "run_id": "run-1727210000",
    "status": "PENDING",
    "message": "Analysis initiated for Tesla Inc"
  }
  ```

---

### `GET /runs/{run_id}`
Polls the execution status and retrieves the final compliance risk report.

- **Response (Status: DONE)**:
  ```json
  {
    "id": "run-1727210000",
    "entity_name": "Tesla Inc",
    "status": "DONE",
    "overall_risk_level": "Medium",
    "report": {
      "overall_risk_score": 45,
      "confidence_level": "High",
      "executive_summary": "...",
      "risk_factors": [...],
      "citations": [...]
    },
    "execution_time_seconds": 4.12
  }
  ```

---

### `GET /runs`
Returns historical execution records.

---

### `GET /runs/{run_id}/trace`
Retrieves step-by-step execution traces, latencies, and tool details.

---

### `GET /health`
Liveness and readiness check returning service status.

---

### `GET /metrics`
Exposes Prometheus telemetry counters, histograms, and latency gauges.
