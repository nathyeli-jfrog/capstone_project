import pytest
from src.agent.graph import generate_draft

def test_generate_draft_formato():
    """Valida la generación correcta del borrador preliminar."""
    resultado = generate_draft.invoke({
        "procedure_type": "Direct Award",
        "purchase_details": "20 ergonomic chairs",
        "selected_vendors": "OfficeSupply Corp"
    })
    
    assert "LEGAL DRAFT (PRELIMINARY VERSION)" in resultado
    assert "Procedure: Direct Award" in resultado
    assert "OfficeSupply Corp" in resultado
    assert "Pending strict human validation" in resultado