import streamlit as st
import openai
from pinecone import Pinecone
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Configurar API Keys
openai.api_key = "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"
pc = Pinecone(api_key="pcsk_3RbqBk_PtTDkRu1JVqPAcYv7Ld6ueH9HjAn6rREqhRPVvb5URRMJn64k7eDgpNkTENfvdC", environment="us-east-1")

firebase_cert = "service_account.json"  # Asegúrate de tener este archivo en la misma carpeta

# Inicializar Firebase si no está ya inicializado
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cert)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Conectar al índice de Pinecone
index_name = "aiie-index"
index = pc.Index(index_name)

def procesar_consulta(query_text):
    try:
        # Buscar en Pinecone usando el método antiguo que no requiere embeddings
        results = index.search_records(
            namespace="values",
            query={"inputs": {"text": query_text}, "top_k": 2},
            fields=["category", "chunk_text"]
        )
        if not results["result"]["hits"]:
            return "No se encontraron resultados en Pinecone.", "No hay respuesta generada."

        # Extraer contexto
        context = "\n".join(hit["fields"]["chunk_text"] for hit in results["result"]["hits"][:2])
        prompt = f"Context: {context}\nQuestion: {query_text}"

        # Generar respuesta con OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            request_timeout=60
        )
        answer = response.choices[0].message["content"]

        # Guardar en Firebase
        db.collection("pinecone_logs").add({
            "query_text": query_text,
            "context": context,
            "response": answer
        })
        return context, answer

    except Exception as e:
        return f"Error: {str(e)}", "No se pudo generar una respuesta."

# Interfaz con Streamlit
st.title("Test Fine-Tuning con Pinecone, OpenAI y Firebase")

# Campo de texto para la consulta
query = st.text_input("Ingresa una consulta:")

if st.button("Enviar"):
    context, answer = procesar_consulta(query)
    st.write("**Contexto:**")
    st.write(context)
    st.write("**Respuesta:**")
    st.write(answer)
