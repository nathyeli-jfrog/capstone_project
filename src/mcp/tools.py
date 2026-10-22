def calcular_umbral_procedimiento(monto_estimado: float) -> str:
    """Calcula el tipo de procedimiento legal aplicable basado en el monto estimado en USD."""
    if monto_estimado < 50000:
        return "Adjudicación Directa Excepcional"
    elif monto_estimado < 200000:
        return "Invitación Restringida"
    return "Licitación Pública Abierta"

def generar_borrador_justificacion(tipo_procedimiento: str, datos_compra: str) -> str:
    """Genera el texto estructurado para el documento legal de justificación."""
    return (
        f"--- BORRADOR LEGAL ---\n"
        f"Procedimiento: {tipo_procedimiento}\n"
        f"Detalles: {datos_compra}\n"
        f"Estado: Pendiente de firma y validación."
    )