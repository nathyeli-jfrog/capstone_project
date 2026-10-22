from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

# Definimos la estructura de memoria del agente
class AgentState(TypedDict):
    # 'add_messages' asegura que los mensajes nuevos se sumen al historial en lugar de sobrescribirlo
    messages: Annotated[list, add_messages]