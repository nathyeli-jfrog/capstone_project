import pytest
import sqlite3
import os
import datetime
from src.agent.graph import verify_blocked_vendors

@pytest.fixture
def db_prueba_temporal(tmp_path):
    """Crea una base de datos SQLite temporal aislada para pruebas unitarias."""
    db_file = tmp_path / "test_vendors.db"
    conexion = sqlite3.connect(db_file)
    cursor = conexion.cursor()
    
    cursor.execute('''
    CREATE TABLE blocked_vendors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        block_reason TEXT NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE
    )
    ''')
    
    # Insertar proveedores de prueba
    sancionados = [
        ("CompusGub", "Incumplimiento contractual", "2024-01-01", "2028-10-10"),
        ("Luxury Coffee Inc", "Corrupción", "2025-06-01", None), # Indefinido
    ]
    cursor.executemany('''
    INSERT INTO blocked_vendors (name, block_reason, start_date, end_date)
    VALUES (?, ?, ?, ?)
    ''', sancionados)
    
    conexion.commit()
    conexion.close()
    return str(db_file)


def test_verificar_proveedor_sancionado_con_sufijos(mocker, db_prueba_temporal):
    """Prueba que el tokenizador atrape 'CompusGub SA' aunque en la BD esté registrado como 'CompusGub'."""
    # Parcheamos la ruta de la BD dentro de la herramienta
    mocker.patch("os.path.abspath", return_value=db_prueba_temporal)
    
    # Invocamos la herramienta con variaciones de nombre
    resultado = verify_blocked_vendors.invoke({"vendor_names": ["CompusGub SA de CV"]})
    
    assert "LEGAL ALERT: BLOCKED VENDORS FOUND" in resultado
    assert "CompusGub" in resultado
    assert "Incumplimiento contractual" in resultado


def test_verificar_proveedor_sancion_indefinida(mocker, db_prueba_temporal):
    """Prueba que detecte correctamente a un proveedor con sanción indefinida (end_date = NULL)."""
    mocker.patch("os.path.abspath", return_value=db_prueba_temporal)
    
    resultado = verify_blocked_vendors.invoke({"vendor_names": ["Luxury Coffee"]})
    
    assert "LEGAL ALERT: BLOCKED VENDORS FOUND" in resultado
    assert "INDEFINITELY" in resultado


def test_verificar_proveedor_habilitado(mocker, db_prueba_temporal):
    """Prueba que un proveedor limpio devuelva confirmación exitosa."""
    mocker.patch("os.path.abspath", return_value=db_prueba_temporal)
    
    resultado = verify_blocked_vendors.invoke({"vendor_names": ["TechSolutions LLC"]})
    
    assert "VERIFICATION SUCCESSFUL" in resultado
    assert "None of the proposed vendors are blocked" in resultado