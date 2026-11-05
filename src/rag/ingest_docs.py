import os
import shutil
import datetime
# Agregamos TextLoader a la importación
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Rutas
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
knowledge_base_dir = os.path.join(base_dir, "knowledge_base")
db_path = os.path.join(base_dir, "src/db/chroma_db")

año_actual = datetime.date.today().year

def crear_base_vectorial():
    print("1. Cargando documentos normativos...")
    # OPTIMIZACIÓN CLAVE: Usar loader_cls=TextLoader evita errores de dependencias pesadas
    loader = DirectoryLoader(
        knowledge_base_dir, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        show_progress=True
    )
    documentos = loader.load()

    for doc in documentos:
        filename = os.path.basename(doc.metadata["source"])
        if filename.startswith("archive_") or "EXPIRED" in doc.page_content:
            doc.metadata["status"] = "expired"
        else:
            doc.metadata["status"] = "active"
            doc.metadata["effective_year"] = año_actual # <-- Dinámico
    
    print(f"-> Se cargaron {len(documentos)} documentos.")
    
    print("2. Segmentando (Chunking) el texto legal...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documentos)
    print(f"-> Se generaron {len(chunks)} fragmentos (chunks).")
    
    print("3. Generando Embeddings locales (Modo Offline)...")
    # Apuntamos a la carpeta física que acabamos de crear
    ruta_modelo = os.path.join(base_dir, "modelos/all-MiniLM-L6-v2")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=ruta_modelo,
        model_kwargs={'local_files_only': True} # Bloquea cualquier intento de conexión a internet
    )
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    print("4. Guardando en ChromaDB...")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)
    
    print("✅ Base de datos vectorial creada con éxito en src/db/chroma_db/")

if __name__ == "__main__":
    crear_base_vectorial()