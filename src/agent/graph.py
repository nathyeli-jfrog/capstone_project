from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

BASE_URL = "http://192.168.1.44:5001/v1"
API_KEY = "EMPTY"  # vLLM no valida la key salvo que lo arranques con --api-key
MODEL = "qwen38"


# 1. Estado y 2. Herramientas Nativas se mantienen igual
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def calcular_umbral(monto_estimado: float) -> str:
    """Calcula el tipo de procedimiento legal aplicable basado en el monto estimado en USD."""
    if monto_estimado < 50000:
        return "Adjudicación Directa Excepcional"
    elif monto_estimado < 200000:
        return "Invitación Restringida"
    return "Licitación Pública Abierta"

@tool
def generar_borrador(tipo_procedimiento: str, datos_compra: str) -> str:
    """Genera el texto estructurado para el documento legal de justificación."""
    return (
        f"\n--- BORRADOR LEGAL GENERADO ---\n"
        f"Procedimiento: {tipo_procedimiento}\n"
        f"Detalles: {datos_compra}\n"
        f"Estado: Pendiente de revisión humana."
    )

herramientas = [calcular_umbral, generar_borrador]

# 3. LLM 
llm = ChatOpenAI(
    base_url=BASE_URL, 
    api_key=API_KEY,                  
    model=MODEL,                    
    temperature=0.1
)

llm_con_herramientas = llm.bind_tools(herramientas)

# 4. Construcción del Grafo (Reasoning Architecture)
def razonamiento_agente(state: AgentState):
    respuesta = llm_con_herramientas.invoke(state["messages"])
    return {"messages": [respuesta]}

workflow = StateGraph(AgentState)
workflow.add_node("razonamiento", razonamiento_agente)

# CAMBIO AQUÍ: Nombramos el nodo exactamente "tools"
workflow.add_node("tools", ToolNode(herramientas))

workflow.add_edge(START, "razonamiento")
workflow.add_conditional_edges("razonamiento", tools_condition)
# CAMBIO AQUÍ: La arista regresa desde "tools"
workflow.add_edge("tools", "razonamiento")

# 5. Seguridad: Guardrail para validación humana
memoria_global = MemorySaver()
agente_app = workflow.compile(
    checkpointer=memoria_global,
    interrupt_before=["tools"] # CAMBIO AQUÍ: Pausa antes del nodo "tools"
)