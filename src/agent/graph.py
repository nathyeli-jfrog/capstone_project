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
    Example query: 'Amount of 50000 USD for coffee'
    """
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../db/chroma_db"))
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        results = vector_store.similarity_search(query, k=5)
        
        if not results:
            return "No related regulations found in the current knowledge base."
            
        legal_context = "\n\n".join([f"Law Extract:\n{doc.page_content}" for doc in results])
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
def generate_draft(procedure_type: str, purchase_details: str, selected_vendors: str) -> str:
    """Generates the text for the legal justification document, including the vendors."""
    return (
        f"\n--- LEGAL DRAFT (PRELIMINARY VERSION) ---\n"
        f"Procedure: {procedure_type}\n"
        f"Vendors to invite/award: {selected_vendors}\n"
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
    
    YOUR STRICT WORKFLOW AND RULES:
    1. MANDATORY: Always start by using the `query_regulations` tool sending the amount and the good to purchase. DO NOT use your memory to decide the procedure.
    2. Read the recovered regulations. If the good is explicitly prohibited, REJECT the purchase immediately and stop.
    3. If allowed, tell the user which procedure applies according to the law you read.
    4. If the law requires 3 vendors, demand them from the user and MANDATORILY verify them using `verify_blocked_vendors`.
    5. Do not invent law articles or vendor names.
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