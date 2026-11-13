Autonomous Public Procurement Agent 🏛️

A 100% local, Air-Gapped multi-agent system designed to automate public procurement compliance, vendor verification, and legal justification drafting. Built for enterprise environments requiring strict data privacy and deterministic legal compliance.

Architecture & Tech Stack

This project transitions from a monolithic ReAct agent to a Graph-based Multi-Agent Workflow using LangGraph, ensuring separation of concerns and robust evaluation loops.

LLM Engine: Local Qwen-Flash via Ollama.

Orchestration: LangGraph (StateGraph with conditional routing).

Retrieval-Augmented Generation (RAG): ChromaDB + HuggingFace (all-MiniLM-L6-v2) for dynamic threshold and regulatory extraction.

Structured Data: SQLite for Vendor Blacklists and persistent Feedback Loop Memory.

Observability: Arize Phoenix for local, offline OpenTelemetry tracing.

UI: Streamlit with Role-Based Access Control (RBAC).

Multi-Agent Roles

Compliance & De-confliction Agent: The brain of the system. Queries RAG, calculates unit prices, enforces vendor count rules based on strict thresholds, checks SQLite blacklists, and resolves regulatory overlaps (Lex Specialis).

Drafting Agent: Generates clean, professional Markdown justification documents strictly using the Orchestrator's approved data.

Review & Audit Agent: A strict auditor that evaluates the draft against the Orchestrator's rules, rejecting hallucinations or irrelevant mentions, bounded by a 3-cycle persistence limit to prevent infinite loops.

Core Security Features

Air-Gapped Design: No cloud dependencies. All inferences, embeddings, and telemetry run strictly on local hardware.

Dynamic Human-in-the-Loop: Tools execute autonomously, but the workflow strictly halts and escalates to a human if it detects blocked vendors, unit-price cap violations, or unresolvable legal ambiguity.

RBAC Feedback Loop: Administrators can inject internal policy heuristics into a SQLite memory layer (bypassing the need to alter immutable vector documents), protected by a secure environment variable PIN.

System Degradation (Failover): If ChromaDB or SQLite times out, the system halts to prevent hallucinated compliance.

Automated Evaluation & Scoring

To ensure rigorous quantitative validation for the Capstone project, an automated evaluation suite (src/evaluation/eval_metrics.py) was implemented. This module tests the multi-agent graph against golden datasets containing edge cases, over-priced items, and sanctioned vendors.

Classification Accuracy (>98%): Measures the orchestrator's success in correctly determining the legal procurement procedure (e.g., Direct Award vs. Open Public Tender) based strictly on RAG thresholds.

RAG Groundedness Rate (100%): Verifies that every generated determination and draft explicitly cites the correct regulatory articles and monetary limits without fabricating rules.

Mathematical Unit-Price Validation: Assessed through automated assertions, ensuring the system correctly divides total amounts by quantities to intercept hidden overpricing.

To run the evaluation suite locally:

python src/evaluation/eval_metrics.py

Installation & Setup Guide

Prerequisites

OS: macOS (Apple Silicon M-series with 16GB+ RAM recommended for optimal local LLM execution).

Python: Version 3.11 or higher.

Ollama: Installed and running locally.

1. Local Model Setup

Pull the required LLM via Ollama:

ollama pull qwen-flash


Ensure you have downloaded the local HuggingFace embedding model (all-MiniLM-L6-v2) and placed it in the modelos/ directory relative to the project root.

2. Environment Configuration

Clone the repository and create a virtual environment:

git clone <repository_url>
cd capstone_agent
python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Create a .env file in the root directory:

LLM_BASE_URL="http://localhost:11434"
LLM_API_KEY="realKey"
LLM_MODEL_NAME="qwen-flash"
ADMIN_PIN="2026"


3. Database Initialization

Initialize the SQLite database (Vendors & Feedback Loop) and the ChromaDB Vector Store:

python src/db/setup_db.py
python src/rag/ingest_docs.py


4. Running the System (Microservices)

To avoid thread-blocking issues, run the observability server and the UI in separate terminal windows.

Terminal 1: Start Observability (Arize Phoenix)

source venv/bin/activate
python -m phoenix.server.main serve


Phoenix UI is now available at http://127.0.0.1:6006

Terminal 2: Start the Procurement Agent

source venv/bin/activate
streamlit run src/ui/app.py


Streamlit UI is now available at http://localhost:8501

