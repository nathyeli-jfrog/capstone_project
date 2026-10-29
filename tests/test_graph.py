import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.graph import agente_app

def test_interrupcion_guardrail_human_in_the_loop(mocker):
    """Verifica que LangGraph se pause antes de ejecutar nodos de herramientas."""
    # Creamos una respuesta del LLM solicitando llamar a una herramienta
    mensaje_llm_con_tool = AIMessage(
        content="",
        tool_calls=[{
            "name": "query_regulations",
            "args": {"query": "laptop purchase 60000 USD"},
            "id": "call_abc123"
        }]
    )
    
    # Mocks para evitar la llamada real al modelo local durante la prueba
    mocker.patch("langchain_openai.ChatOpenAI.invoke", return_value=mensaje_llm_con_tool)
    
    config = {"configurable": {"thread_id": "test_session_unit"}}
    
    # Invocamos el grafo
    estado = agente_app.invoke(
        {"messages": [HumanMessage(content="Buy laptops for 60000 USD")]},
        config
    )
    
    # Obtenemos el estado actual del grafo
    estado_siguiente = agente_app.get_state(config)
    
    # Validamos que el próximo nodo a ejecutar sea 'tools' y esté pausado por la interrupción
    assert estado_siguiente.next[0] == "tools"