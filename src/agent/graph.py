import os
import sqlite3
import datetime
import operator
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List

# --- OBSERVABILIDAD LOCAL DESACOPLADA (Air-Gapped) ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor

# 1. Configurar el proveedor de trazas apuntando al servidor local de Phoenix
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(tracer_provider)

# 2. Instrumentar LangChain de forma pasiva (no bloquea Streamlit)
LangChainInstrumentor().instrument()

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.db.feedback import get_recent_corrections

# --- CONFIGURACIÓN DEL MODELO ---
load_dotenv()
MODEL_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY", "local")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b")

llm = ChatOpenAI(
    base_url=MODEL_URL, 
    api_key=API_KEY,                  
    model=MODEL_NAME,                    
    temperature=0.0 
)

# --- HERRAMIENTAS (TOOLS) ---
@tool
def query_regulations(query: str) -> str:
    """Searches the current regulations (RAG) for monetary thresholds and business rules."""
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db/chroma_db"))
        modelo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modelos/all-MiniLM-L6-v2"))
        
        if not os.path.exists(db_path):
            return "FAILOVER_ACTIVATED: RAG Database unreachable."

        embeddings = HuggingFaceEmbeddings(model_name=modelo_path, model_kwargs={'local_files_only': True})
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        results_user_query = vector_store.similarity_search(query, k=3, filter={"status": "active"})
        results_prohibitions = vector_store.similarity_search("General Provisions Article 2 Universally Prohibited Goods", k=2, filter={"status": "active"})
        
        combined_results = []
        seen_texts = set()
        for doc in results_prohibitions + results_user_query:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                combined_results.append(doc)
        
        return "RECOVERED REGULATIONS:\n" + "\n\n".join([f"Extract:\n{doc.page_content}" for doc in combined_results])
    except Exception as e:
        return f"FAILOVER_ACTIVATED: RAG query error. Detail: {e}"

@tool
def verify_blocked_vendors(vendor_names: list[str]) -> str:
    """Verifies if proposed vendors are on the blacklist."""
    try:
        if not vendor_names:
            return "ERROR: No names provided."
            
        current_date = datetime.date.today().isoformat()
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db/vendors.db"))
        
        if not os.path.exists(db_path):
            return "FAILOVER_ACTIVATED: SQLite Database unreachable."

        connection = sqlite3.connect(db_path, timeout=5.0)
        cursor = connection.cursor()
        
        WORDS_TO_IGNORE = {"of", "the", "and", "inc", "llc", "corp", "ltd", "co"}
        or_clauses = []
        parameters = []
        
        for name in vendor_names:
            for sub_name in name.split(','):
                text = sub_name.strip().lower()
                if not text:
                    continue
                raw_words = text.split()
                key_words = [w for w in raw_words if w not in WORDS_TO_IGNORE and len(w) > 2]
                if key_words:
                    and_clause = " AND ".join(["name LIKE ?"] * len(key_words))
                    or_clauses.append(f"({and_clause})")
                    for w in key_words:
                        parameters.append(f"%{w}%")
        
        if not or_clauses:
            return "ERROR: No valid keywords remaining."

        where_clauses = " OR ".join(or_clauses)
        query = f'''
            SELECT name, block_reason, start_date, end_date 
            FROM blocked_vendors 
            WHERE ({where_clauses}) 
            AND (end_date IS NULL OR end_date > ?)
        '''
        parameters.append(current_date)
        cursor.execute(query, parameters)
        results = cursor.fetchall()
        connection.close()
        
        if results:
            block_details = []
            for r in results:
                db_name, reason, start, end = r
                end_text = "INDEFINITELY" if end is None else f"until {end}"
                block_details.append(f"- {db_name} (Sanctioned from {start} {end_text} Reason: {reason})")
            return "LEGAL ALERT: BLOCKED VENDORS FOUND:\n" + "\n".join(block_details)
            
        return "VERIFICATION SUCCESSFUL: None of the proposed vendors are blocked."
    except Exception as e:
        return f"FAILOVER_ACTIVATED: SQLite query error. Detail: {e}"

@tool
def generate_draft(procedure_type: str, purchase_details: str, threshold_range_cited: str, selected_vendors: str) -> str:
    """Generates the preliminary Markdown legal justification document."""
    return (
        f"\n--- LEGAL DRAFT (PRELIMINARY VERSION) ---\n"
        f"Procedure: {procedure_type}\n"
        f"Cited Threshold Range: {threshold_range_cited}\n"
        f"Vendors: {selected_vendors}\n"
        f"Details: {purchase_details}\n"
        f"Status: Pending strict human validation."
    )

tools = [query_regulations, verify_blocked_vendors, generate_draft]
compliance_llm = llm.bind_tools(tools)

# --- ESTADO DEL GRAFO ---
def add_ints(a: int, b: int) -> int:
    return a + b

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    revision_count: Annotated[int, add_ints]

# --- AGENTES ---
def compliance_agent(state: AgentState):
    print("\n[AGENT] 🛡️ Compliance & De-confliction Agent active")
    today = datetime.date.today()
    correcciones = get_recent_corrections()

    system_prompt = SystemMessage(content=f"""
    You are the Compliance Agent. Date: {today.isoformat()}. Language: English.

    {correcciones}
    
    AUTONOMOUS EXECUTION & ESCALATION RULES:
    1. Call `query_regulations` autonomously to verify law and procedure.
    2. Call `verify_blocked_vendors` autonomously when vendor names are provided.
    
    3. DYNAMIC THRESHOLD EXTRACTION (CRITICAL): Extract procedural rules, monetary thresholds, and minimum vendor requirements SOLELY from the active text returned by `query_regulations`. DO NOT use hardcoded thresholds or assumptions.
    
    4. CONCEPT SEPARATION - AVOID SEMANTIC BLEEDING (CRITICAL LOGIC):
       - PROCEDURE & VENDOR COUNT: Determine the required procedure and the REQUIRED NUMBER OF VENDORS based strictly on the rule that matches the TOTAL AMOUNT. If the text says the procedure for this total amount requires/allows only 1 vendor, then 1 or 2 vendors is fully compliant. Do not demand more.
       - UNIT PRICE CAP: Determine if there is a unit price cap rule (applicable to Total Amount / Quantity).
       - NEVER confuse an Article number or a unit price cap with the required number of vendors. (e.g., an "Article 3" capping prices does NOT mean you need 3 vendors).
       
    5. UNIT PRICE VALIDATION: Mathematically calculate the Unit Price (Total Amount / Quantity). If it exceeds the regulatory cap extracted from the RAG, output `[ESCALATE_TO_HUMAN]`.
    
    6. FILTER IRRELEVANT RULES: Strictly ignore and DO NOT mention any internal memos or rules that do not apply to the current purchase item.
    
    7. ESCALATION (HUMAN-IN-THE-LOOP): If a proposed vendor is blocked, or unit price exceeded, you MUST halt and output EXACTLY: `[ESCALATE_TO_HUMAN]` followed by the explanation.
    
    8. HANDOFF: Once ALL compliance checks pass (clean vendors, valid unit price, sufficient vendor count based strictly on the retrieved text), append EXACTLY: `[APPROVED_FOR_DRAFTING]`.
    
    9. COMPLETION: If the user approves the draft, output EXACTLY: "✅ The procurement analysis is complete and closed. The final justification document is approved."
    """)
    
    response = compliance_llm.invoke([system_prompt] + state["messages"])
    response.name = "ComplianceAgent"
    return {"messages": [response], "revision_count": 0}

def drafting_agent(state: AgentState):
    print("\n[AGENT] 📝 Drafting Agent active")
    system_prompt = SystemMessage(content="""
    You are the Drafting Agent. The Compliance Agent has ALREADY APPROVED this purchase. 
    Your ONLY job is to format their approved verification into a clean Markdown Justification Document.
    
    CRITICAL RULES:
    1. DO NOT RE-EVALUATE COMPLIANCE. Do not state that the process is "pending", "conditionally pending", or that more vendors are required. Assume compliance is 100% satisfied.
    2. DO NOT include 'N/A' tables or mention irrelevant policies. Keep it focused.
    3. If the Review Agent rejected your previous draft, strictly correct the errors they pointed out.
    
    Sign off with EXACTLY: 'Drafting complete. Passing to Review Agent.'
    """)
    response = llm.invoke([system_prompt] + state["messages"])
    response.name = "DraftingAgent"
    return {"messages": [response], "revision_count": 0}

def review_agent(state: AgentState):
    print("\n[AGENT] 🔍 Review & Audit Agent active")
    if state.get("revision_count", 0) >= 3:
        return {"messages": [AIMessage(content="[PERSISTENT_INCONSISTENCY] Review Agent aborted after 3 failed revision cycles. [ESCALATE_TO_HUMAN]", name="ReviewAgent")], "revision_count": 0}

    system_prompt = SystemMessage(content="""
    You are the Review Agent. Audit the draft against the Compliance Agent's instructions.
    
    EVALUATION CRITERIA:
    1. Check for hallucinated vendors, incorrect amounts, or wrong procedures.
    2. Ensure irrelevant rules (like chairs for a laptop purchase) are completely absent.
    
    TOLERANCE RULE (CRITICAL): Do NOT reject the draft for minor formatting, markdown styling, table structures, or phrasing preferences. ONLY reject for CRITICAL legal or factual errors.
    
    If legally and factually clean, output EXACTLY: `[FINAL_DOCUMENT_APPROVED]` followed by the final document. 
    If critical errors exist, output `[REJECTED_DRAFT]` with the specific correction needed.
    """)
    response = llm.invoke([system_prompt] + state["messages"])
    response.name = "ReviewAgent"
    
    aumento = 1 if "[REJECTED_DRAFT]" in response.content else 0
    return {"messages": [response], "revision_count": aumento}

# --- ENRUTADORES ---
def compliance_router(state: AgentState):
    last_msg = state["messages"][-1]
    content = str(last_msg.content).upper()
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print("[ROUTER] 🔀 Directing to Tool Execution (Autonomous)")
        return "tools"
        
    if "APPROVED_FOR_DRAFTING" in content:
        print("[ROUTER] 🔀 Handoff triggered -> Drafting Agent")
        return "drafting_agent"
        
    if "UNRESOLVED_AMBIGUITY" in content or "FAILOVER_ACTIVATED" in content or "ESCALATE_TO_HUMAN" in content:
        print("[ROUTER] 🛑 Halting for Human Intervention")
        return END
        
    print("[ROUTER] ⏸️ Awaiting user input / workflow complete")
    return END

def review_router(state: AgentState):
    return "drafting_agent" if "[REJECTED_DRAFT]" in state["messages"][-1].content else END

# --- CONSTRUCCIÓN DEL GRAFO ---
workflow = StateGraph(AgentState)
workflow.add_node("compliance_agent", compliance_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("drafting_agent", drafting_agent)
workflow.add_node("review_agent", review_agent)

workflow.add_edge(START, "compliance_agent")
workflow.add_conditional_edges("compliance_agent", compliance_router)
workflow.add_edge("tools", "compliance_agent")
workflow.add_edge("drafting_agent", "review_agent")
workflow.add_conditional_edges("review_agent", review_router)

global_memory = MemorySaver()
# NOTA: Se eliminó interrupt_before=["tools"]. El agente ahora es completamente autónomo 
# hasta que decida explícitamente escalar al humano.
agente_app = workflow.compile(checkpointer=global_memory)