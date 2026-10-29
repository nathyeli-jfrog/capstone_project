import os
import sqlite3
import datetime
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
MODEL_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY", "local")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def query_regulations(query: str) -> str:
    """
    Searches the current regulations (RAG) for monetary thresholds, applicable procedures, 
    prohibited goods, and business rules for a specific purchase.
    """
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db/chroma_db"))
        modelo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modelos/all-MiniLM-L6-v2"))
        
        embeddings = HuggingFaceEmbeddings(
            model_name=modelo_path,
            model_kwargs={'local_files_only': True}
        )
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # 1. Búsqueda semántica para la solicitud específica del usuario
        results_user_query = vector_store.similarity_search(query, k=3)
        
        # 2. Búsqueda GARANTIZADA para las Disposiciones Generales y Bienes Prohibidos
        results_prohibitions = vector_store.similarity_search("General Provisions Article 2 Universally Prohibited Goods alcohol gifts luxury", k=2)
        
        # Combinar los resultados eliminando duplicados
        combined_results = []
        seen_texts = set()
        
        for doc in results_prohibitions + results_user_query:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                combined_results.append(doc)
        
        legal_context = "\n\n".join([f"Law Extract:\n{doc.page_content}" for doc in combined_results])
        return f"RECOVERED REGULATIONS:\n{legal_context}"
    except Exception as e:
        return f"RAG_ERROR: Failed to query the document base. Detail: {e}"


@tool
def verify_blocked_vendors(vendor_names: list[str]) -> str:
    """
    Verifies if one or more proposed vendors are on the blacklist.
    Uses token-based search to ignore variations like 'Inc', 'LLC'.
    """
    try:
        if not vendor_names:
            return "ERROR: No names provided for verification."
            
        current_date = datetime.date.today().isoformat()
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db/vendors.db"))
        
        connection = sqlite3.connect(db_path)
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
            return "ERROR: No valid keywords remaining after cleaning the text."

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
                
            return "LEGAL ALERT: BLOCKED VENDORS FOUND:\n" + "\n".join(block_details) + "\nSTOP. DO NOT generate any document. Demand the user to propose other vendors."
            
        return "VERIFICATION SUCCESSFUL: None of the proposed vendors are blocked. You may proceed."
    except Exception as e:
        return f"INTERNAL_ERROR: SQLite database connection failed. Detail: {e}"

@tool
def generate_draft(procedure_type: str, purchase_details: str, threshold_range_cited: str, selected_vendors: str) -> str:
    """
    Generates the text for the legal justification document.
    `threshold_range_cited` must be the exact monetary range extracted dynamically from the active RAG text.
    """
    return (
        f"\n--- LEGAL DRAFT (PRELIMINARY VERSION) ---\n"
        f"Procedure: {procedure_type}\n"
        f"Cited Threshold Range: {threshold_range_cited}\n"
        f"Vendors: {selected_vendors}\n"
        f"Details: {purchase_details}\n"
        f"Status: Pending strict human validation."
    )

tools = [query_regulations, verify_blocked_vendors, generate_draft]

llm = ChatOpenAI(
    base_url=MODEL_URL, 
    api_key=API_KEY,                  
    model=MODEL_NAME,                    
    temperature=0.0 
)
llm_with_tools = llm.bind_tools(tools)

def agent_reasoning(state: AgentState):
    today = datetime.date.today().isoformat()
    system_prompt = SystemMessage(content=f"""
    You are an Autonomous Public Procurement Agent. Today's date is {today}.
    ALL YOUR RESPONSES AND INTERACTIONS WITH THE USER MUST BE IN ENGLISH.
    
    DYNAMIC REGULATORY WORKFLOW (SINGLE SOURCE OF TRUTH):
    1. MANDATORY FIRST STEP: Always call `query_regulations` with the requested item and estimated amount.
    2. DYNAMIC THRESHOLD EXTRACTION: Extract procedural rules, monetary thresholds, and minimum vendor requirements SOLELY from the active text returned by `query_regulations`. DO NOT use hardcoded thresholds or assumptions.
    3. OBSOLETE REGULATION FILTER: Ignore any document fragment marked as EXPIRED, OBSOLETE, or SUPERSEDED. Enforce ONLY active provisions.
    4. PROHIBITED GOODS AUDIT: If the recovered active text marks the item as prohibited, REJECT the purchase immediately.
    5. VENDOR VERIFICATION: If the recovered active text mandates 2, 3, or more vendors for the identified threshold, demand them from the user and verify each using `verify_blocked_vendors`.
    6. DRAFT GENERATION: Populate the justification draft using ONLY the dynamic values extracted from the active law.
    7. CURRENCY FORMATTING RULE: NEVER use the '$' symbol when displaying monetary amounts (e.g., write 'USD 60,000' or '60,000 USD' instead of '$60,000').
    """)
    
    full_messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("reasoning", agent_reasoning)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "reasoning")
workflow.add_conditional_edges("reasoning", tools_condition)
workflow.add_edge("tools", "reasoning")

global_memory = MemorySaver()
agente_app = workflow.compile(checkpointer=global_memory, interrupt_before=["tools"])