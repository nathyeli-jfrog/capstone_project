import asyncio
import sys
from src.agent.graph import procesar_mensaje_con_mcp

async def probar_comunicacion():
    print("1. Iniciando conexión con el servidor MCP de forma aislada...")
    solicitud = "Necesito comprar 10 computadoras por un total de $120,000 USD. ¿Qué procedimiento aplica?"
    
    try:
        print("2. Enviando solicitud al agente...")
        resultado = await procesar_mensaje_con_mcp(solicitud, thread_id="prueba_aislada")
        
        print("\n✅ ¡ÉXITO! RESPUESTA DEL AGENTE:")
        print("--------------------------------------------------")
        print(resultado["messages"][-1].content)
        print("--------------------------------------------------")
        
    except Exception as e:
        print(f"\n❌ OCURRIÓ UN ERROR TÉCNICO: {e}")

if __name__ == "__main__":
    # Ejecutamos el ciclo asíncrono de forma limpia
    asyncio.run(probar_comunicacion())