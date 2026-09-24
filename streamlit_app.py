"""
Agentic Compliance & Risk Research Assistant - Enterprise Portal

Architecture: Streamlit UI interacting with FastAPI backend (http://localhost:8000).
Fallback: Direct graph execution if API is unreachable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise Compliance & Risk Portal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    .sidebar-header {
        padding: 1.25rem 0.5rem 1rem 0.5rem;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.25rem;
    }
    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #f8fafc;
        margin: 0;
    }
    .sidebar-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* Enterprise Headers */
    .app-header {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
    }
    .app-description {
        font-size: 0.875rem;
        color: #94a3b8;
        margin: 0;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Card Containers */
    .card-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }

    .card-title {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #cbd5e1;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }

    /* Metric Box */
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.25rem;
    }

    /* Risk Badges */
    .badge-high {
        background-color: #451a1a;
        color: #f87171;
        border: 1px solid #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-medium {
        background-color: #452b1a;
        color: #fbbf24;
        border: 1px solid #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-low {
        background-color: #1a3d2f;
        color: #34d399;
        border: 1px solid #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }

    /* Source Tags */
    .tag-source {
        background-color: #0f172a;
        color: #94a3b8;
        border: 1px solid #334155;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Timeline Trace Box */
    .trace-item {
        background-color: #0f172a;
        border-left: 3px solid #2563eb;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        border-radius: 0 6px 6px 0;
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.75rem;
    }
    .trace-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        font-weight: 600;
        color: #93c5fd;
        margin-bottom: 0.4rem;
    }
    .trace-detail {
        font-size: 0.8rem;
        color: #cbd5e1;
        line-height: 1.4;
    }

    /* Primary Buttons Override */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: background-color 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
    }

    /* Custom Tables */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }
    .data-table th {
        background-color: #0f172a;
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid #1e293b;
    }
    .data-table td {
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #cbd5e1;
        border-bottom: 1px solid #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def run_pipeline_api(entity_name: str, ticker: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    payload = {"entity_name": entity_name}
    if ticker:
        payload["ticker"] = ticker
    headers = {}
    if api_key:
        headers["X-Google-API-Key"] = api_key
    if model_name:
        headers["X-Gemini-Model"] = model_name

    try:
        resp = requests.post(f"{API_BASE}/analyze", json=payload, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.error(f"API Connection Error: {e}")
    return None


def poll_run_status(run_id: str, max_wait: int = 120) -> Optional[Dict[str, Any]]:
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{API_BASE}/runs/{run_id}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") in ["DONE", "FAILED"]:
                    return data
        except Exception:
            pass
        time.sleep(2)
    return None


def run_pipeline_inline(entity_name: str, ticker: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    if model_name:
        os.environ["GEMINI_MODEL"] = model_name

    from app.agents.graph import create_compiled_graph

    graph = create_compiled_graph()
    initial_state = {
        "entity_name": entity_name,
        "ticker": ticker,
        "search_queries": [],
        "raw_filings": [],
        "raw_news": [],
        "matched_policies": [],
        "draft_report": {},
        "verifier_flags": [],
        "retry_count": 0,
        "execution_trace": [],
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_t = time.time()
    config = {"configurable": {"thread_id": f"inline-{int(time.time())}"}}
    try:
        final_state = loop.run_until_complete(graph.ainvoke(initial_state, config=config))
        elapsed = time.time() - start_t
        report = final_state.get("draft_report") or {}
        return {
            "id": f"local-{int(time.time())}",
            "entity_name": entity_name,
            "ticker": ticker,
            "status": "DONE",
            "overall_risk_level": report.get("overall_risk_level", "Medium"),
            "report": report,
            "trace": final_state.get("execution_trace", []),
            "execution_time_seconds": round(elapsed, 2),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    finally:
        loop.close()


def render_risk_badge(level: str) -> str:
    lvl = (level or "Medium").upper()
    if "HIGH" in lvl:
        return f'<span class="badge-high">HIGH RISK</span>'
    elif "LOW" in lvl:
        return f'<span class="badge-low">LOW RISK</span>'
    else:
        return f'<span class="badge-medium">MEDIUM RISK</span>'


def render_source_tag(source_type: str) -> str:
    stype = (source_type or "unknown").lower()
    if "edgar" in stype or "filing" in stype:
        return '<span class="tag-source">SEC EDGAR</span>'
    elif "news" in stype:
        return '<span class="tag-source">FINANCIAL NEWS</span>'
    elif "policy" in stype:
        return '<span class="tag-source">COMPLIANCE POLICY</span>'
    else:
        return f'<span class="tag-source">{stype.upper()}</span>'


# Sidebar Setup
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-title">Risk Assistant</div>
            <div class="sidebar-subtitle">Enterprise Compliance Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_online = check_api_health()
    status_label = "ONLINE" if api_online else "OFFLINE (INLINE MODE)"
    status_color = "#34d399" if api_online else "#fbbf24"
    st.markdown(
        f"""
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:6px; padding:0.75rem; margin-bottom:1.25rem;">
            <div style="font-size:0.7rem; font-weight:600; color:#64748b; text-transform:uppercase;">Backend Service</div>
            <div style="font-size:0.85rem; font-weight:700; color:{status_color}; margin-top:0.25rem;">{status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="font-size:0.75rem; font-weight:600; color:#64748b; text-transform:uppercase; margin-bottom:0.5rem;">API Key Configuration</div>', unsafe_allow_html=True)
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        placeholder="AIzaSy...",
        help="Provide your Google Gemini API Key to run real-time agentic research.",
    )

    st.markdown('<div style="font-size:0.75rem; font-weight:600; color:#64748b; text-transform:uppercase; margin-bottom:0.5rem; margin-top:0.75rem;">Model Selection</div>', unsafe_allow_html=True)
    selected_gemini_model = st.selectbox(
        "Google Gemini Model",
        [
            "models/gemini-3.5-flash-lite",
            "models/gemini-3.5-flash",
            "models/gemini-3.8-flash",
            "models/gemini-3.7-flash",
            "models/gemini-3.6-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-2.5-flash",
            "models/gemini-flash-lite-latest",
            "models/gemini-flash-latest",
            "models/gemini-pro-latest",
        ],
        index=0,
        help="Select the Google Gemini model for risk scoring and verification.",
    )

    navigation_choice = st.radio(
        "Navigation",
        ["Entity Analysis", "Execution History", "Audit Trace", "Documentation"],
        index=0,
    )

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem; font-weight:600; color:#64748b; text-transform:uppercase; margin-bottom:0.5rem;">Sample Entities</div>', unsafe_allow_html=True)
    sample_selected = st.selectbox("Select Quick Test Entity", ["None", "Tesla Inc (TSLA)", "Apple Inc (AAPL)", "Boeing Co (BA)"])
    if sample_selected != "None":
        if "TSLA" in sample_selected:
            st.session_state["entity_input"] = "Tesla Inc"
            st.session_state["ticker_input"] = "TSLA"
        elif "AAPL" in sample_selected:
            st.session_state["entity_input"] = "Apple Inc"
            st.session_state["ticker_input"] = "AAPL"
        elif "BA" in sample_selected:
            st.session_state["entity_input"] = "Boeing Co"
            st.session_state["ticker_input"] = "BA"


# Top Application Header
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Agentic Compliance & Risk Research Portal</div>
        <div class="app-description">
            Automated multi-agent intelligence system for enterprise entity risk research, SEC filings cross-examination, policy compliance validation, and verified risk scoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if navigation_choice == "Entity Analysis":
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown(
            """
            <div class="card-container">
                <div class="card-title">Analysis Request Form</div>
            """,
            unsafe_allow_html=True,
        )

        entity_val = st.session_state.get("entity_input", "Tesla Inc")
        ticker_val = st.session_state.get("ticker_input", "TSLA")

        entity_name = st.text_input("Entity Name", value=entity_val, placeholder="e.g. Acme Corporation")
        ticker = st.text_input("Ticker Symbol (Optional)", value=ticker_val, placeholder="e.g. ACME")

        execute_button = st.button("Run Risk Assessment", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if "last_run_result" in st.session_state:
            res = st.session_state["last_run_result"]
            st.markdown(
                f"""
                <div class="card-container">
                    <div class="card-title">Latest Run Summary</div>
                    <div style="margin-bottom:0.75rem;">
                        <span style="font-size:0.8rem; color:#94a3b8;">Target Entity:</span>
                        <div style="font-size:1rem; font-weight:700; color:#f8fafc;">{res.get('entity_name')}</div>
                    </div>
                    <div style="margin-bottom:0.75rem;">
                        <span style="font-size:0.8rem; color:#94a3b8;">Risk Level:</span><br/>
                        {render_risk_badge(res.get('overall_risk_level', 'Medium'))}
                    </div>
                    <div>
                        <span style="font-size:0.8rem; color:#94a3b8;">Execution Time:</span>
                        <div style="font-size:0.9rem; font-weight:600; color:#cbd5e1;">{res.get('execution_time_seconds', 0)}s</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        if execute_button:
            if not entity_name.strip():
                st.error("Please enter a valid entity name.")
            elif not user_api_key.strip():
                st.error("Please provide a valid Google Gemini API Key in the left sidebar to execute live risk assessment.")
            else:
                with st.spinner("Executing agent pipeline..."):
                    if api_online:
                        init_res = run_pipeline_api(entity_name, ticker, api_key=user_api_key, model_name=selected_gemini_model)
                        if init_res and "run_id" in init_res:
                            run_data = poll_run_status(init_res["run_id"])
                            if run_data:
                                st.session_state["last_run_result"] = run_data
                            else:
                                st.error("Analysis execution timed out.")
                        else:
                            st.error("Failed to initiate analysis via API.")
                    else:
                        run_data = run_pipeline_inline(entity_name, ticker, api_key=user_api_key, model_name=selected_gemini_model)
                        st.session_state["last_run_result"] = run_data

        if "last_run_result" in st.session_state:
            data = st.session_state["last_run_result"]
            report = data.get("report", {})

            st.markdown(
                f"""
                <div class="card-container">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid #334155; padding-bottom:0.75rem;">
                        <div>
                            <span style="font-size:0.75rem; font-weight:600; color:#64748b; text-transform:uppercase;">Compliance Report</span>
                            <h3 style="margin:0; font-size:1.25rem; font-weight:700; color:#f8fafc;">{data.get('entity_name')}</h3>
                        </div>
                        <div>
                            {render_risk_badge(report.get('overall_risk_level', 'Medium'))}
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            metrics_c1, metrics_c2, metrics_c3 = st.columns(3)
            with metrics_c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Risk Score</div>
                        <div class="metric-value">{report.get('overall_risk_score', 0)} / 100</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with metrics_c2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence Level</div>
                        <div class="metric-value">{report.get('confidence_level', 'High')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with metrics_c3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Risk Factors</div>
                        <div class="metric-value">{len(report.get('risk_factors', []))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="margin-top:1.5rem;" class="card-title">Executive Rationale</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.9rem; line-height:1.6; color:#cbd5e1; background:#0f172a; padding:1rem; border-radius:6px; border:1px solid #1e293b;">{report.get("executive_summary", "No executive summary provided.")}</div>', unsafe_allow_html=True)

            risk_factors = report.get("risk_factors", [])
            if risk_factors:
                st.markdown('<div style="margin-top:1.5rem;" class="card-title">Identified Risk Factors</div>', unsafe_allow_html=True)
                for idx, rf in enumerate(risk_factors, 1):
                    rf_level = rf.get("severity", "Medium")
                    st.markdown(
                        f"""
                        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:6px; padding:1rem; margin-bottom:0.75rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                <span style="font-weight:700; font-size:0.95rem; color:#f8fafc;">{idx}. {rf.get('category', 'Risk Factor')}</span>
                                {render_risk_badge(rf_level)}
                            </div>
                            <div style="font-size:0.875rem; color:#cbd5e1; margin-bottom:0.5rem; line-height:1.5;">{rf.get('finding', '')}</div>
                            <div style="font-size:0.8rem; color:#94a3b8; background:#1e293b; padding:0.5rem; border-radius:4px;">
                                <strong>Impact Analysis:</strong> {rf.get('impact', 'N/A')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            citations = report.get("citations", [])
            if citations:
                st.markdown('<div style="margin-top:1.5rem;" class="card-title">Evidence & Provenance Citations</div>', unsafe_allow_html=True)
                for cit in citations:
                    st.markdown(
                        f"""
                        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:6px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                                <span style="font-size:0.8rem; font-weight:600; color:#cbd5e1;">{cit.get('source_doc_id', 'Source Document')}</span>
                                {render_source_tag(cit.get('source_type', ''))}
                            </div>
                            <div style="font-size:0.8rem; color:#94a3b8; font-style:italic;">"{cit.get('passage', '')}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="card-container" style="text-align:center; padding:3rem 1.5rem;">
                    <div style="font-size:1.1rem; font-weight:600; color:#cbd5e1; margin-bottom:0.5rem;">No Assessment Selected</div>
                    <div style="font-size:0.85rem; color:#64748b; max-width:400px; margin:0 auto;">
                        Enter an entity name on the left panel and click "Run Risk Assessment" to generate a detailed compliance report.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


elif navigation_choice == "Execution History":
    st.markdown(
        """
        <div class="card-container">
            <div class="card-title">Historical Execution Records</div>
        """,
        unsafe_allow_html=True,
    )

    if api_online:
        try:
            resp = requests.get(f"{API_BASE}/runs", timeout=5)
            if resp.status_code == 200:
                history_data = resp.json().get("runs", [])
            else:
                history_data = []
        except Exception:
            history_data = []
    else:
        history_data = []

    if "last_run_result" in st.session_state:
        history_data.insert(0, st.session_state["last_run_result"])

    if history_data:
        rows_html = ""
        for item in history_data:
            badge = render_risk_badge(item.get("overall_risk_level", "Medium"))
            rows_html += f"""
            <tr>
                <td style="font-weight:600;">{item.get('id', 'N/A')}</td>
                <td style="font-weight:700; color:#f8fafc;">{item.get('entity_name', 'N/A')}</td>
                <td>{item.get('ticker') or '-'}</td>
                <td><span style="font-weight:600; color:#34d399;">{item.get('status', 'DONE')}</span></td>
                <td>{badge}</td>
                <td>{item.get('created_at', 'N/A')}</td>
            </tr>
            """

        st.markdown(
            f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Run ID</th>
                        <th>Entity Name</th>
                        <th>Ticker</th>
                        <th>Status</th>
                        <th>Risk Assessment</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No execution records found.")

    st.markdown("</div>", unsafe_allow_html=True)


elif navigation_choice == "Audit Trace":
    st.markdown(
        """
        <div class="card-container">
            <div class="card-title">Multi-Agent Execution Trace Log</div>
        """,
        unsafe_allow_html=True,
    )

    if "last_run_result" in st.session_state:
        trace = st.session_state["last_run_result"].get("trace", [])
        if trace:
            for step in trace:
                agent = step.get("agent", "SYSTEM").upper()
                agent_display = f"[{agent}]"
                details = step.get("details", {})
                timestamp = step.get("timestamp", "")

                st.markdown(
                    f"""
                    <div class="trace-item">
                        <div class="trace-header">
                            <span>{agent_display}</span>
                            <span style="color:#64748b;">{timestamp}</span>
                        </div>
                        <div class="trace-detail">
                            <pre style="background:#050811; color:#93c5fd; padding:0.5rem; border-radius:4px; font-size:0.75rem; border:1px solid #1e293b; overflow-x:auto;">{json.dumps(details, indent=2)}</pre>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No detailed trace entries recorded for the current execution.")
    else:
        st.info("Run an analysis first to inspect step-by-step execution traces.")

    st.markdown("</div>", unsafe_allow_html=True)


elif navigation_choice == "Documentation":
    st.markdown(
        """
        <div class="card-container">
            <div class="card-title">System Architecture & Pipeline Documentation</div>
            <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">
                <p>The Enterprise Compliance & Risk Research Assistant relies on a 4-Agent LangGraph State Machine to perform grounded risk evaluation:</p>
                <ol style="margin-left:1.25rem;">
                    <li style="margin-bottom:0.5rem;"><strong>[RESEARCHER AGENT]</strong>: Fetches latest regulatory SEC EDGAR filings and financial news via rate-limited API calls with exponential backoff retries.</li>
                    <li style="margin-bottom:0.5rem;"><strong>[COMPLIANCE CHECKER]</strong>: Queries vector-indexed compliance policy documents using hybrid semantic search and metadata filtering to identify applicable mandates.</li>
                    <li style="margin-bottom:0.5rem;"><strong>[RISK SCORER]</strong>: Synthesizes research findings against compliance rules using structured LLM outputs. Every claim is strictly grounded with source passage citations.</li>
                    <li style="margin-bottom:0.5rem;"><strong>[VERIFIER AGENT]</strong>: Re-examines draft report claims against retrieved source evidence. Flags or strips ungrounded statements and determines final confidence scoring.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
