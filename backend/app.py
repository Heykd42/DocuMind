import streamlit as st
import requests
import uuid

# Base URL for the FastAPI server
API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG & Document QA Interface", layout="wide")
st.title("RAG & Document QA Interface")
st.markdown("""
This interface lets you interact with our document indexing and retrieval system powered by a
Retrieval Augmented Generation (RAG) pipeline. You can upload PDFs, ask questions, and manage chat sessions.
""")

operation = st.sidebar.radio("Select Operation", ["Upload PDF", "Query Chat", "Reset Chat"])

if operation == "Upload PDF":
    st.header("📄 Upload PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        st.info(f"File selected: {uploaded_file.name}")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        with st.spinner("Uploading and processing PDF..."):
            resp = requests.post(f"{API_BASE}/upload_pdf/", files=files)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"PDF uploaded successfully! Document ID: {data['document_id']}")
            st.write("Indexed documents count:", data["indexed_documents"])
        else:
            st.error("PDF upload failed. " + resp.text)

elif operation == "Query Chat":
    st.header("💬 Query Chat with LLM")
    session_id = st.text_input(
        "Session ID",
        value=str(uuid.uuid4())[:8],
        help="Use the same session ID to maintain conversation context"
    )
    document_id = st.text_input(
        "Document ID (optional)",
        help="Provide a document ID to restrict search to a specific document"
    )
    st.markdown("#### Conversation History (Optional)")
    previous_conversations = st.text_area(
        "Enter previous conversation history here...",
        height=150
    )
    query = st.text_input("Enter your query", placeholder="E.g., What is the treatment for Chickenpox?")

    if st.button("Submit Query"):
        if not query.strip():
            st.error("Please enter a query.")
        else:
            payload = {
                "query": query,
                "document_id": document_id.strip() or None,
                "session_id": session_id,
                "previous_context": previous_conversations.strip() or None
            }
            with st.spinner("Generating response..."):
                resp = requests.post(f"{API_BASE}/rag", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                st.subheader("Response")
                st.markdown(data["response"])
                st.subheader("Chat History")
                st.text(data.get("chat_history", "No history available."))
            else:
                st.error("Error: " + resp.text)

elif operation == "Reset Chat":
    st.header("🔄 Reset Chat Session")
    session_id = st.text_input("Enter Session ID to Reset", value=str(uuid.uuid4())[:8])
    if st.button("Reset Chat"):
        payload = {"session_id": session_id}
        resp = requests.post(f"{API_BASE}/reset_chat/", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            st.success(data.get("message", "Chat reset."))
        else:
            st.error("Error: " + resp.text)

st.markdown("---")
st.caption("Powered by Together.AI, Meta‑Llama, and Streamlit")