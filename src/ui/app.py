import streamlit as st
import sys
import os

# Configuración de rutas e importación del grafo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.agent.graph import agente_app
from src.db.feedback import save_correction

st.set_page_config(page_title="Procurement Assistant", layout="wide")
st.title("Autonomous Public Procurement Agent")

# --- FUNCIÓN DE LIMPIEZA PARA PREVENIR ERRORES DE LATEX EN STREAMLIT ---
def limpiar_texto_streamlit(texto: str) -> str:
    """Escapa el símbolo '$' para evitar que Streamlit lo interprete como LaTeX."""
    if not texto:
        return ""
    return texto.replace("$", r"\$")

# Inicialización de la memoria de sesión en la UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# ID de sesión único para el checkpointer de LangGraph
graph_config = {"configurable": {"thread_id": "procurement_session_03"}}

# 1. DIBUJAR HISTORIAL DE CONVERSACIÓN (Aplicando la función de limpieza)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(limpiar_texto_streamlit(msg["content"]))

# 2. VERIFICACIÓN DEL GUARDRAIL (HUMAN-IN-THE-LOOP)
current_state = agente_app.get_state(graph_config)

if current_state.next and current_state.next[0] == "tools":
    st.warning("⚠️ **Security Guardrail (Human-in-the-Loop):** The agent has requested to execute tools (Database/RAG). Your authorization is required to proceed.")
    
    if st.button("✅ Authorize and Execute Tools"):
        with st.spinner("Executing authorized tools..."):
            result = agente_app.invoke(None, graph_config)
            llm_response = result["messages"][-1].content
            
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
            st.rerun()

# 3. ENTRADA DE CHAT DEL USUARIO
user_input = st.chat_input("Enter your purchase requirement...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing requirement..."):
            try:
                result = agente_app.invoke(
                    {"messages": [("user", user_input)]}, 
                    graph_config
                )
                llm_response = result["messages"][-1].content
                st.session_state.messages.append({"role": "assistant", "content": llm_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Execution Error: {e}")

if len(st.session_state.messages) >= 2 and st.session_state.messages[-1]["role"] == "assistant":
    last_user_query = st.session_state.messages[-2]["content"]
    last_agent_response = st.session_state.messages[-1]["content"]
    
    st.markdown("---")
    with st.expander("⚠️ Provide Correction / Override Agent Decision"):
        st.write("If the agent made a regulatory mistake, log the correction here. The agent will memorize it for future requests.")
        
        with st.form("feedback_form", clear_on_submit=True):
            human_correction = st.text_area("Describe the correct procedure or rule the agent missed:")
            submitted = st.form_submit_button("💾 Save Correction to Memory")
            
            if submitted and human_correction:
                try:
                    # Delegamos la responsabilidad de escritura a la capa de datos
                    save_correction(last_user_query, last_agent_response, human_correction)
                    st.success("✅ Correction saved successfully! The agent's reasoning logic has been updated.")
                except Exception as e:
                    st.error(f"Failed to save correction: {e}")                