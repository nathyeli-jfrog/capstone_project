import requests
from langchain_openai import ChatOpenAI

BASE_URL = "http://192.168.1.44:5001/v1"
API_KEY = "EMPTY"  # vLLM no valida la key salvo que lo arranques con --api-key

# 1) Verificar que el servidor responde y ver el nombre exacto del modelo
r = requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=10)
r.raise_for_status()
modelos = [m["id"] for m in r.json()["data"]]
print("Modelos servidos:", modelos)

# Usa el nombre tal cual lo reporta vLLM (o el de --served-model-name)
MODEL = modelos[0]

# 2) Cliente LangChain
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0.7,
    max_tokens=256,
    timeout=60,
    # Qwen3: desactiva el modo "thinking" para respuestas directas.
    # Quita esta línea si quieres ver el razonamiento <think>.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# 3) Llamada simple
resp = llm.invoke("Responde solo con 'OK' si me recibes.")
print("Respuesta:", resp.content)
print("Uso de tokens:", resp.response_metadata.get("token_usage"))

# 4) Streaming
print("Streaming: ", end="")
for chunk in llm.stream("Cuenta del 1 al 5."):
    print(chunk.content, end="", flush=True)
print()