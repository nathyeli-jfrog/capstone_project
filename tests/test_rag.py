import pytest
import os
from langchain_huggingface import HuggingFaceEmbeddings
from src.agent.graph import query_regulations

def test_carga_embeddings_local_offline():
    """Verifica que el modelo MiniLM físico se pueda cargar de forma local."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    modelo_path = os.path.join(base_dir, "modelos/all-MiniLM-L6-v2")
    
    assert os.path.exists(modelo_path), "La carpeta del modelo local no existe."
    
    embeddings = HuggingFaceEmbeddings(
        model_name=modelo_path,
        model_kwargs={'local_files_only': True}
    )
    
    vector = embeddings.embed_query("test query")
    assert isinstance(vector, list)
    assert len(vector) == 384  # Dimensión estándar de MiniLM-L6-v2


def test_herramienta_query_regulations(mocker):
    """Verifica la ejecución de la herramienta RAG con ChromaDB simulada."""
    # Mock del similarity_search de ChromaDB
    mock_doc = mocker.MagicMock()
    mock_doc.page_content = "Article 2: Corporate gifts are strictly prohibited."
    
    mock_vectorstore = mocker.MagicMock()
    mock_vectorstore.similarity_search.return_value = [mock_doc]
    
    mocker.patch("src.agent.graph.Chroma", return_value=mock_vectorstore)
    
    resultado = query_regulations.invoke({"query": "corporate gifts"})
    
    assert "RECOVERED REGULATIONS" in resultado
    assert "Corporate gifts are strictly prohibited" in resultado