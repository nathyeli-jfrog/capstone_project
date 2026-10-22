import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mcp.server.fastmcp import FastMCP
from tools import calcular_umbral_procedimiento, generar_borrador_justificacion

mcp = FastMCP("ServidorComprasPublicas")

@mcp.tool()
def calcular_umbral(monto_estimado: float) -> str:
    return calcular_umbral_procedimiento(monto_estimado)

@mcp.tool()
def generar_borrador(tipo_procedimiento: str, datos_compra: str) -> str:
    return generar_borrador_justificacion(tipo_procedimiento, datos_compra)

if __name__ == "__main__":
    print("Iniciando Servidor de Herramientas MCP en HTTP (SSE) puerto 8000...")
    # Cambiamos stdio por sse
    mcp.run(transport='sse')