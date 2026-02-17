import os, io
import streamlit as st
from dotenv import load_dotenv
import PyPDF2

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Setup ----------
st.set_page_config(page_title="Chat With PDF", page_icon="📄")
st.title("📄 Chat With PDF")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY missing in .env")
    st.stop()

# ---------- Upload ----------
file = st.file_uploader("Upload PDF", type=["pdf"])

if not file:
    st.info("Upload a PDF to start.")
    st.stop()

# ---------- Read PDF ----------
reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
text = "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
if not text:
    st.error("No text found in PDF (maybe scanned).")
    st.stop()

# ---------- Split ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_text(text)
if not chunks:
    st.error("Could not create chunks from PDF.")
    st.stop()

st.write(f"Chunks: **{len(chunks)}**")

# ---------- Embeddings + FAISS (HuggingFace) ----------
@st.cache_resource(show_spinner="Building vector store...")
def build_vectorstore(docs):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(docs, emb)

vectordb = build_vectorstore(chunks)
retriever = vectordb.as_retriever()

# ---------- Prompt + LLM ----------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant. Answer ONLY using the context.\n\n"
        'If answer is not in context, reply exactly: "Answer not available in the provided PDF."\n\n'
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
    ),
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=api_key,
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# ---------- UI: Question ----------
q = st.text_input("Ask something about this PDF:")

if st.button("Get Answer") and q:
    with st.spinner("Thinking..."):
        resp = rag_chain.invoke(q)
        st.subheader("Answer:")
        st.write(resp.content)
