import sqlite3
import os

def get_db_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "vendors.db"))

def save_correction(user_query: str, agent_error: str, human_correction: str) -> None:
    """Save a human correction in the database."""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO agent_corrections (user_query, agent_error, human_correction) 
        VALUES (?, ?, ?)
    """, (user_query, agent_error, human_correction))
    conn.commit()
    conn.close()

def get_recent_corrections(limit: int = 5) -> str:
    """Capture the lessons learned and inject them into the System Prompt."""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute("SELECT user_query, human_correction FROM agent_corrections ORDER BY id DESC LIMIT ?", (limit,))
        registros = cursor.fetchall()
        conn.close()
        
        if not registros:
            return "No previous corrections logged."
            
        lecciones = "\n".join([f"- Query: '{r[0]}' -> Rule Learned: {r[1]}" for r in registros])
        return f"LEARNED CORRECTIONS FROM PAST HUMAN OVERRIDES:\n{lecciones}"
    except Exception:
        return "No previous corrections logged."