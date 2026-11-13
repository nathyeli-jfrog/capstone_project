import os
import sqlite3
import datetime
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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