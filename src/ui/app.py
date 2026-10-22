
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.agent.graph import agente_app

st.set_page_config(page_title="Asistente de Compras", layout="wide")
st.title("Asistente Autónomo de Compras Públicas")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Configuramos el hilo de memoria
config_grafo = {"configurable": {"thread_id": "sesion_compras_01"}}

# Dibujar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

solicitud = st.chat_input("Ejemplo: Necesito comprar 10 computadoras por $45,000 USD.")

if solicitud:
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": solicitud})
    with st.chat_message("user"):
        st.markdown(solicitud)
        
    with st.chat_message("assistant"):
        with st.spinner("Analizando requerimiento y políticas..."):
            try:
                # Invocamos el grafo de forma síncrona y segura
                resultado = agente_app.invoke(
                    {"messages": [("user", solicitud)]}, 
                    config_grafo
                )
                
                respuesta_llm = resultado["messages"][-1].content
                st.markdown(respuesta_llm)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_llm})
                
                # Revisar si el grafo se detuvo por el Guardrail
                estado_grafo = agente_app.get_state(config_grafo)
                if estado_grafo.next and estado_grafo.next[0] == "tools":
                    st.warning("⚠️ El agente ha solicitado usar una herramienta. Requiere validación humana en la consola.")
                    
            except Exception as e:
                st.error(f"Error de ejecución: {e}")