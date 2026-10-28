import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.agent.graph import agente_app

st.set_page_config(page_title="Procurement Assistant", layout="wide")
st.title("Autonomous Public Procurement Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

graph_config = {"configurable": {"thread_id": "procurement_session_01"}}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

current_state = agente_app.get_state(graph_config)

if current_state.next and current_state.next[0] == "tools":
    st.warning("⚠️ **Security Guardrail (Human-in-the-Loop):** The agent has requested to execute tools (Database/RAG). Your authorization is required to proceed.")
    
    if st.button("✅ Authorize and Execute Tools"):
        with st.spinner("Executing authorized tools..."):
            result = agente_app.invoke(None, graph_config)
            llm_response = result["messages"][-1].content
            
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
            st.rerun()

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