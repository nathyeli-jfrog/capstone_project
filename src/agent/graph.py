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


from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.db.feedback import get_recent_corrections

from src.tools.tools import query_regulations, verify_blocked_vendors, generate_draft


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
    
    STEP-BY-STEP EXECUTION:
    1. TOOLS: Call `query_regulations` to retrieve the rules. Call `verify_blocked_vendors` to check vendors.
    
    2. RULE EXTRACTION (READ CAREFULLY FROM RAG):
       - Thresholds: Find the monetary ranges for procedures (e.g., Direct Award vs Restricted Invitation) and the REQUIRED number of vendors for each range.
       - Unit Caps: Find the maximum unit price allowed for the requested item.
       
    3. MATHEMATICAL & LOGICAL ANALYSIS:
       - Total Amount = Quantity * Unit Price.
       - Match the Total Amount to the correct procedural range retrieved in Step 2.
       - If the retrieved procedure for this Total Amount requires ONLY 1 vendor, and the user provided 1 or 2 vendors, the vendor count requirement is OFFICIALLY MET. DO NOT demand more vendors.
       - Verify the Unit Price is equal to or below the retrieved Unit Cap.
       
    4. ESCALATION (HUMAN-IN-THE-LOOP): 
       - If a proposed vendor is blocked, or the unit price cap is exceeded, you MUST halt and output EXACTLY: `[ESCALATE_TO_HUMAN]` followed by the explanation.
       
    5. HANDOFF: 
       - If the unit price is valid, vendors are clean, and the vendor count matches the requirement for the specific Total Amount, append EXACTLY: `[APPROVED_FOR_DRAFTING]`.
       
    6. COMPLETION: 
       - If the user approves the final draft, output EXACTLY: "✅ The procurement analysis is complete and closed. The final justification document is approved."
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