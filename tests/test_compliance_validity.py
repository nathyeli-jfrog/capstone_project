import pytest
from langchain_core.messages import HumanMessage
from src.agent.graph import agente_app

@pytest.mark.parametrize("query,expected_procedure,forbidden_keywords", [
    (
        "I need to buy office stationery for 20000 USD", 
        "Direct Award", 
        ["Restricted Invitation", "2020", "EXPIRED", "SUPERSEDED"]
    ),
    (
        "I need to buy network servers for 120000 USD", 
        "Restricted Invitation", 
        ["Public Tender", "2020", "EXPIRED", "100,000"]
    ),
])
def test_regulatory_vigilance(query, expected_procedure, forbidden_keywords):
    """Verifica que el agente use únicamente los umbrales vigentes de 2026."""
    config = {"configurable": {"thread_id": "test_compliance_session"}}
    
    # 1. Primera invocación (Búsqueda RAG)
    resultado = agente_app.invoke({"messages": [HumanMessage(content=query)]}, config)
    respuesta_texto = resultado["messages"][-1].content
    
    # 2. Validaciones de cumplimiento
    assert expected_procedure.lower() in respuesta_texto.lower(), f"Se esperaba el procedimiento '{expected_procedure}'"
    
    for word in forbidden_keywords:
        assert word.lower() not in respuesta_texto.lower(), f"Se encontró la palabra/norma obsoleta prohibida: '{word}'"