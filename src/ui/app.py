import streamlit as st
import sys
import os
from langchain_core.messages import AIMessage, HumanMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.agent.graph import agente_app
from src.db.feedback import save_correction

st.set_page_config(page_title="Procurement Multi-Agent System", layout="wide")

# --- SIMULACIÓN DE RBAC (Control de Acceso Validado) ---
admin_mode = False
admin_pin_secret = os.getenv("ADMIN_PIN", "0000") # Respaldo por si falla el .env

with st.sidebar:
    st.header("🔐 User Roles")
    selected_role = st.selectbox("Select your role:", ["Standard User", "Administrator / Compliance Officer"])
    
    if selected_role == "Administrator / Compliance Officer":
        # Campo de contraseña que enmascara los caracteres
        pin_input = st.text_input("Enter Admin PIN:", type="password")
        
        if pin_input == admin_pin_secret:
            st.success("✅ Identity verified. Write-access granted for Feedback Loop.")
            admin_mode = True
        elif pin_input != "":
            st.error("❌ Invalid PIN. Access denied.")
    else:
        st.info("Standard User Role: Read-only access to policies.")
        admin_mode = False
    
    st.markdown("---")
    st.markdown("📊 **Observability:**\nOpen `http://127.0.0.1:6006` in your browser to view the Arize Phoenix traces.")

st.title("Autonomous Public Procurement System")

def limpiar_texto_streamlit(texto: str) -> str:
    return texto.replace("$", r"\$") if texto else ""

if "messages" not in st.session_state:
    st.session_state.messages = []

graph_config = {"configurable": {"thread_id": "procurement_session_final"}}

# 1. DIBUJAR HISTORIAL
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    agent_name = msg.get("agent_name", None)
    
    with st.chat_message(role):
        if agent_name == "ComplianceAgent": st.caption("🛡️ **Compliance Agent**")
        elif agent_name == "DraftingAgent": st.caption("📝 **Drafting Agent**")
        elif agent_name == "ReviewAgent": st.caption("🔍 **Review Agent**")
        st.markdown(limpiar_texto_streamlit(msg["content"]))

# 2. ENTRADA DE CHAT (Flujo Autónomo)
user_input = st.chat_input("Enter your purchase requirement...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🤖 Multi-Agent Processing...", expanded=True) as status:
            try:
                st.write("🛡️ Agents evaluating and executing tools autonomously...")
                
                # El agente se ejecuta automáticamente sin pausas manuales
                result = agente_app.invoke({"messages": [HumanMessage(content=user_input)]}, graph_config)
                
                for m in result["messages"]:
                    if isinstance(m, AIMessage) and m.content:
                        name = getattr(m, "name", "ComplianceAgent")
                        if not any(msg.get("content") == m.content for msg in st.session_state.messages):
                            st.session_state.messages.append({
                                "role": "assistant", "content": m.content, "agent_name": name
                            })
                            
                status.update(label="✅ Workflow Cycle Completed", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"Execution Error: {e}")

# 3. BUCLE DE RETROALIMENTACIÓN (Restringido por RBAC)
if len(st.session_state.messages) >= 2 and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]["content"].upper()
    
    # Alerta visual si el agente escaló un problema
    if "ESCALATE_TO_HUMAN" in last_msg or "UNRESOLVED_AMBIGUITY" in last_msg:
        st.error("🚨 **HUMAN INTERVENTION REQUIRED:** The agent has safely halted the process and escalated this case for your review.")
    
    # Inyector de reglas visible SOLO si es Administrador
    if admin_mode:
        st.markdown("---")
        with st.expander("⚙️ Admin: Inject Internal Policy (Feedback Loop)"):
            with st.form("feedback_form", clear_on_submit=True):
                human_correction = st.text_area("Add a strict heuristic or correction to the system memory:")
                if st.form_submit_button("💾 Save to System Memory"):
                    save_correction(st.session_state.messages[-2]["content"], st.session_state.messages[-1]["content"], human_correction)
                    st.success("✅ Policy saved. Agents will enforce this on the next iteration.")