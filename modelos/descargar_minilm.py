import os
from huggingface_hub import snapshot_download

def aislar_modelo():
    ruta_destino = os.path.abspath(os.path.join(os.path.dirname(__file__), "modelos/all-MiniLM-L6-v2"))
    
    print(f"Descargando MiniLM (Versión Optimizada ~90MB) en:\n{ruta_destino}")
    
    # Filtramos la descarga para bajar SOLAMENTE los pesos modernos y la configuración
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2", 
        local_dir=ruta_destino,
        allow_patterns=["*.safetensors", "*.json", "*.txt"] # Ignora ONNX, TensorFlow, CoreML, etc.
    )
    print("\n✅ ¡Modelo descargado y aislado con éxito! Ya no dependes de internet.")

if __name__ == "__main__":
    aislar_modelo()